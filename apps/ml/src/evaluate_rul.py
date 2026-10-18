"""
Walk-forward evaluation of RUL models — RF-30.

Entry point:
    python -m src.evaluate_rul [--stride 20]

Per walk-forward fold (same splitter/purge as RF-28):
* TRAINING labels are censored at the training cut-off (no future failure
  date leaks into training — see ``rul.survival_labels``).
* TEST labels use the true next-failure time inside the test segment.

Models:
* ``xgb_aft``  — XGBoost Accelerated Failure Time on the 80 features.
* ``renewal``  — calendar baseline: mean inter-failure gap seen in training
                 minus time since the last training failure (what a
                 time-based maintenance plan implicitly assumes).

Outputs: models/eval_rul.json, docs/figures-ml/rul_fold*.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ML_ROOT))

from src.evaluate_temporal import EMBARGO_S, PURGE_S, load_features  # noqa: E402
from src.ingest_metropt import FAILURE_INTERVALS  # noqa: E402
from src.rul import (  # noqa: E402
    SurvivalLabels,
    concordance_index,
    mae_near_failure,
    survival_labels,
)
from src.stat_tests import bootstrap_ci  # noqa: E402
from src.temporal_validation import (  # noqa: E402
    PurgedWalkForwardSplit,
    events_from_intervals,
)  # noqa: E402

OUT_JSON: Path = _ML_ROOT / "models" / "eval_rul.json"
FIG_DIR: Path = _ML_ROOT.parent.parent / "docs" / "figures-ml"
HORIZONS_H: tuple[float, ...] = (24.0, 72.0, 168.0)

_AFT_PARAMS: dict[str, Any] = {
    "objective": "survival:aft",
    "eval_metric": "aft-nloglik",
    "aft_loss_distribution": "normal",
    "aft_loss_distribution_scale": 1.2,
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log: logging.Logger = logging.getLogger("evaluate_rul")


def _renewal_prediction(ts_test: pd.Series, train_events: list[Any]) -> np.ndarray:
    starts = sorted(ev.start for ev in train_events)
    if len(starts) >= 2:
        gap_h = float(np.mean(np.diff([s.value for s in starts])) / 3.6e12)
    else:  # one failure: assume the observed time from data start to it
        gap_h = (
            float((starts[0] - ts_test.min()).total_seconds() / 3600)
            if starts
            else 24.0 * 30
        )
    last = starts[-1] if starts else ts_test.min()
    since_h = (ts_test - last).dt.total_seconds().to_numpy() / 3600
    return np.maximum(gap_h - since_h, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stride", type=int, default=20, help="training subsample stride"
    )
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pooled",
        action="store_true",
        help="MetroPT-3 + MetroPT 2022, harmonised sensors (7 failures, 6 folds)",
    )
    args = parser.parse_args()

    if args.pooled:
        import src.evaluate_cross_dataset as xd
        from src.ingest_metropt2022 import FAILURE_INTERVALS_2022

        xd.configure(["Reservoirs", "Oil_level"])
        X, _, ts, is22 = xd.load_pooled()
        events = events_from_intervals([*FAILURE_INTERVALS, *FAILURE_INTERVALS_2022])
    else:
        X, _, ts = load_features()
        is22 = np.zeros(len(ts), dtype=bool)
        events = events_from_intervals(FAILURE_INTERVALS)

    def labels(idx: np.ndarray, end: pd.Timestamp | None = None) -> SurvivalLabels:
        """Per-dataset survival labels: a 2020 row is censored at the end of
        the 2020 recording, never 'failing' at a 2022 failure 1.5 years later."""
        parts = []
        for flag in (False, True):
            sub = idx[is22[idx] == flag]
            if sub.size == 0:
                continue
            t = ts.iloc[sub]
            seg_end = t.max() if end is None else min(t.max(), end)
            evs = [e for e in events if t.min() <= e.start <= t.max()]
            parts.append((sub, survival_labels(t, evs, end=seg_end)))
        order = np.argsort(np.concatenate([p[0] for p in parts]))
        cat = {
            k: np.concatenate([getattr(p[1], k) for p in parts])[order]
            for k in ("lower", "upper", "observed", "valid")
        }
        return SurvivalLabels(**cat)

    all_idx = np.arange(len(ts))
    truth = labels(all_idx)

    folds_out: list[dict[str, Any]] = []
    for fold in PurgedWalkForwardSplit(events, PURGE_S, EMBARGO_S).split(ts):
        cutoff: pd.Timestamp = ts.iloc[fold.train_idx].max()
        train_events = [ev for ev in events if ev.start <= cutoff]
        tr_lab = labels(fold.train_idx, end=cutoff)
        keep = np.flatnonzero(tr_lab.valid)[:: args.stride]
        tr_idx = fold.train_idx[keep]

        dtrain = xgb.DMatrix(X.iloc[tr_idx])
        dtrain.set_float_info("label_lower_bound", tr_lab.lower[keep])
        dtrain.set_float_info("label_upper_bound", tr_lab.upper[keep])
        booster = xgb.train(
            {**_AFT_PARAMS, "seed": args.seed}, dtrain, num_boost_round=args.rounds
        )

        te_valid = fold.test_idx[truth.valid[fold.test_idx]]
        t_true = truth.lower[te_valid]
        e_true = truth.observed[te_valid]
        ts_te = ts.iloc[te_valid].reset_index(drop=True)
        pred_aft = booster.predict(xgb.DMatrix(X.iloc[te_valid]))
        pred_ren = _renewal_prediction(ts_te, train_events)

        res: dict[str, Any] = {
            "fold": fold.fold,
            "test_event": [str(fold.test_event.start), str(fold.test_event.end)],
            "n_train": int(tr_idx.size),
            "n_train_observed": int(tr_lab.observed[keep].sum()),
            "n_test": int(te_valid.size),
            "models": {},
        }
        for name, pred in (("xgb_aft", pred_aft), ("renewal", pred_ren)):
            res["models"][name] = {
                "c_index": concordance_index(t_true, e_true, pred, seed=args.seed),
                **{
                    f"mae_h_within_{int(h)}h": mae_near_failure(t_true, pred, e_true, h)
                    for h in HORIZONS_H
                },
            }
        folds_out.append(res)
        log.info("fold=%d %s", fold.fold, json.dumps(res["models"]))
        _plot(
            ts_te,
            t_true,
            e_true,
            pred_aft,
            pred_ren,
            f"pooled_{fold.fold}" if args.pooled else str(fold.fold),
        )

    summary: dict[str, Any] = {}
    for name in ("xgb_aft", "renewal"):
        summary[name] = {
            k: bootstrap_ci(
                np.array([f["models"][name][k] for f in folds_out])
            ).to_dict()
            for k in folds_out[0]["models"][name]
        }
    out = {
        "schema_version": "1.0",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "requirements": ["RF-30"],
        "data": (
            "MetroPT-3 + MetroPT 2022 (pooled, harmonised)"
            if args.pooled
            else "MetroPT-3"
        ),
        "config": {
            "stride": args.stride,
            "rounds": args.rounds,
            "seed": args.seed,
            **_AFT_PARAMS,
        },
        "folds": folds_out,
        "summary": summary,
    }
    out_json = OUT_JSON.with_name("eval_rul_pooled.json") if args.pooled else OUT_JSON
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)


def _plot(
    ts_te: pd.Series,
    t_true: np.ndarray,
    e_true: np.ndarray,
    aft: np.ndarray,
    ren: np.ndarray,
    fold: int | str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step = max(len(ts_te) // 4000, 1)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    obs = e_true[::step]
    ax.plot(
        ts_te[::step][obs],
        t_true[::step][obs],
        color="black",
        lw=1.2,
        label="true RUL (observed)",
    )
    ax.plot(ts_te[::step], aft[::step], lw=0.8, label="XGBoost AFT")
    ax.plot(ts_te[::step], ren[::step], lw=0.8, ls="--", label="renewal baseline")
    ax.set_yscale("log")
    ax.set_ylabel("RUL (hours, log)")
    ax.set_title(f"RF-30 — walk-forward fold {fold}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"rul_fold{fold}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
