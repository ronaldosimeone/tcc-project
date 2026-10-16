"""
Honest temporal evaluation of the classical models — RF-28 / RF-29 / RNF-79.

Entry point:
    python -m src.evaluate_temporal [--seeds 5] [--neg-stride 5]

Protocols
---------
* ``legacy_random_split`` — reproduces the original stratified random 80/20
  split (one seed) to quantify how much it inflated the metrics.
* ``walk_forward``        — PurgedWalkForwardSplit, PRIMARY protocol: the model
  never sees data after the test segment (3 folds = failures 2, 3, 4).
* ``blocked``             — PurgedEventBlockKFold, SECONDARY protocol
  (4 folds, trains on the future, purge + embargo).

Decision thresholds (never tuned on the test fold)
-------------------------------------------------
* ``fixed``  — a-priori threshold 0.5.
* ``inner``  — RN-02 cost-optimal threshold chosen on out-of-fold scores of the
  TRAINING data (inner leave-one-event-out). Falls back to 0.5 when the
  training data contains fewer than two failures.
* ``oracle`` — cost-optimal on the test fold itself. Reported ONLY as an upper
  bound; never used to compare or select models.

Training rows: every positive sample plus every ``neg_stride``-th negative
sample (memory/time budget; test folds are always evaluated at full
resolution). Logistic regression is the simple baseline required by the
Friedman comparison (k >= 3 models).

Outputs:
    models/eval_temporal.json
    docs/figures-ml/cd_*.png, docs/figures-ml/scores_fold*.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ML_ROOT))

from src.event_metrics import (  # noqa: E402
    cost_optimal_threshold,
    evaluate_events,
    persistent_alarms,
)
from src.ingest_metropt import FAILURE_INTERVALS  # noqa: E402
from src.preprocessing import MetroPTPreprocessor  # noqa: E402
from src.stat_tests import (  # noqa: E402
    bootstrap_ci,
    friedman_nemenyi,
    plot_cd_diagram,
    wilcoxon_paired,
)
from src.uncertainty import (  # noqa: E402
    MondrianConformalClassifier,
    calibration_report,
    isotonic_recalibrate,
)
from src.temporal_validation import (  # noqa: E402
    FailureEvent,
    PurgedEventBlockKFold,
    PurgedWalkForwardSplit,
    TemporalFold,
    events_from_intervals,
)
from src.train_random_forest import FEATURE_COLS, TARGET_COL  # noqa: E402

PARQUET_PATH: Path = _ML_ROOT / "data" / "processed" / "metropt3.parquet"
OUT_JSON: Path = _ML_ROOT / "models" / "eval_temporal.json"
FIG_DIR: Path = _ML_ROOT.parent.parent / "docs" / "figures-ml"

# Rolling features look back at most 15 samples (150 s): 300 s purge covers it.
PURGE_S: float = 300.0
EMBARGO_S: float = 3600.0
# RN-03: 6 consecutive samples (~60 s) above threshold before an alarm.
PERSISTENCE_K: int = 6
# RN-01: an alarm counts as early warning when it precedes the failure by >= 2 h;
# alarms up to 6 h before a failure are attributed to it (not false alarms).
EARLY_HORIZON_S: float = 6 * 3600.0
MIN_LEAD_S: float = 2 * 3600.0
GRACE_S: float = 3600.0
# RN-02: a missed failure costs 100x a false alarm (assumption; sensitivity
# analysis belongs in the thesis text).
COST_FN: float = 100.0
COST_FP: float = 1.0
FIXED_THRESHOLD: float = 0.5
# RF-31: target miscoverage of the conformal prediction sets (per class).
CONFORMAL_ALPHA: float = 0.1

# XGBoost best Optuna params (docs/model_comparison.md, 100 trials).
_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 598,
    "max_depth": 7,
    "learning_rate": 0.1877,
    "subsample": 0.9201,
    "colsample_bytree": 0.7351,
    "min_child_weight": 1,
    "gamma": 0.0023,
}

EVAL_KW: dict[str, float] = {
    "early_horizon_s": EARLY_HORIZON_S,
    "grace_s": GRACE_S,
    "min_lead_s": MIN_LEAD_S,
    "cost_fn": COST_FN,
    "cost_fp": COST_FP,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log: logging.Logger = logging.getLogger("evaluate_temporal")

ModelFactory = Callable[[int, float], Any]


def _rf(seed: int, pos_weight: float) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


def _xgb(seed: int, pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        **_XGB_PARAMS,
        scale_pos_weight=pos_weight,
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        eval_metric="logloss",
    )


def _logreg(seed: int, pos_weight: float) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
    )


MODELS: dict[str, ModelFactory] = {
    "random_forest": _rf,
    "xgboost": _xgb,
    "logistic_regression": _logreg,
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_features() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Features are computed once over the full series: every rolling/lag
    feature is backward-looking, and the purge removes the overlap."""
    df: pd.DataFrame = pd.read_parquet(PARQUET_PATH)
    df = df.sort_values("timestamp").reset_index(drop=True)
    X: pd.DataFrame = MetroPTPreprocessor().fit_transform(df[FEATURE_COLS].copy())
    X = X.astype(np.float32)
    return X, df[TARGET_COL].astype(int), pd.to_datetime(df["timestamp"])


def _subsample(idx: np.ndarray, y: pd.Series, neg_stride: int) -> np.ndarray:
    y_sub: np.ndarray = y.to_numpy()[idx]
    pos: np.ndarray = idx[y_sub == 1]
    neg: np.ndarray = idx[y_sub == 0][::neg_stride]
    return np.sort(np.concatenate([pos, neg]))


def _fit_predict(
    name: str,
    seed: int,
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    y_tr: np.ndarray = y.to_numpy()[train_idx]
    pos: int = int(y_tr.sum())
    pos_weight: float = (len(y_tr) - pos) / max(pos, 1)
    model = MODELS[name](seed, pos_weight)
    model.fit(X.iloc[train_idx], y_tr)
    return model.predict_proba(X.iloc[test_idx])[:, 1]


# ---------------------------------------------------------------------------
# Inner threshold selection (never sees the test fold)
# ---------------------------------------------------------------------------


def _inner_threshold(
    name: str,
    seed: int,
    X: pd.DataFrame,
    y: pd.Series,
    ts: pd.Series,
    train_idx: np.ndarray,
    events: list[FailureEvent],
    neg_stride: int,
    candidates: Sequence[float] | None = None,
) -> tuple[float, str, np.ndarray | None, np.ndarray | None]:
    """Returns ``(threshold, source, oof_scores, oof_labels)``. The out-of-fold
    training scores are reused as the conformal calibration set (RF-31).
    ``candidates`` overrides the default threshold grid (0.05…0.95)."""
    t_lo, t_hi = ts.iloc[train_idx].min(), ts.iloc[train_idx].max()
    inner_events: list[FailureEvent] = [
        ev for ev in events if ev.start >= t_lo and ev.end <= t_hi
    ]
    if len(inner_events) < 2:
        return (
            FIXED_THRESHOLD,
            "fallback_fixed (<2 failures in training data)",
            None,
            None,
        )

    ts_tr: pd.Series = ts.iloc[train_idx].reset_index(drop=True)
    oof: np.ndarray = np.full(len(train_idx), np.nan)
    for inner in PurgedEventBlockKFold(inner_events, PURGE_S, EMBARGO_S).split(ts_tr):
        tr_abs: np.ndarray = _subsample(train_idx[inner.train_idx], y, neg_stride)
        # In the blocked protocol the training rows have a hole (the outer
        # test segment + purge/embargo), so an inner segment can be empty or
        # its complement can lack failures — skip it instead of fitting on 0 rows.
        if inner.test_idx.size == 0 or y.to_numpy()[tr_abs].sum() == 0:
            continue
        oof[inner.test_idx] = _fit_predict(
            name, seed, X, y, tr_abs, train_idx[inner.test_idx]
        )

    mask: np.ndarray = ~np.isnan(oof)
    scored_events: list[FailureEvent] = [
        ev
        for ev in inner_events
        if ((ts_tr[mask] >= ev.start) & (ts_tr[mask] <= ev.end)).any()
    ]
    if len(scored_events) < 1:
        return (FIXED_THRESHOLD, "fallback_fixed (no usable inner fold)", None, None)
    thr, _ = cost_optimal_threshold(
        ts_tr[mask].reset_index(drop=True),
        oof[mask],
        scored_events,
        k=PERSISTENCE_K,
        candidates=candidates,
        **EVAL_KW,
    )
    return (thr, "inner_oof_cost_optimal", oof[mask], y.to_numpy()[train_idx][mask])


def _uncertainty_block(
    scores: np.ndarray,
    y_te: np.ndarray,
    oof: np.ndarray | None,
    oof_y: np.ndarray | None,
) -> dict[str, Any]:
    """RF-31: calibration of the raw test scores, and Mondrian conformal sets
    calibrated on the training out-of-fold scores (never on the test fold)."""
    rep = calibration_report(y_te, scores)
    out: dict[str, Any] = {
        "ece": rep.ece,
        "mce": rep.mce,
        "brier": rep.brier,
        "conformal": None,
    }
    if oof is None or oof_y is None or len(set(oof_y.tolist())) < 2:
        out["conformal"] = {
            "skipped": "no out-of-fold calibration scores with both classes"
        }
        return out
    iso = isotonic_recalibrate(oof, oof_y)
    out["ece_isotonic"] = calibration_report(y_te, iso.predict(scores)).ece
    cp = MondrianConformalClassifier(alpha=CONFORMAL_ALPHA).fit(oof, oof_y)
    out["conformal"] = {
        "alpha": CONFORMAL_ALPHA,
        **cp.empirical_coverage(cp.predict_sets(scores), y_te),
    }
    return out


# ---------------------------------------------------------------------------
# Fold evaluation
# ---------------------------------------------------------------------------


def _event_block(
    ts_test: pd.Series, scores: np.ndarray, thr: float, events: list[FailureEvent]
) -> dict[str, Any]:
    alarms: np.ndarray = persistent_alarms(scores, thr, PERSISTENCE_K)
    rep = evaluate_events(ts_test, alarms, events, **EVAL_KW).to_dict()
    rep["threshold"] = thr
    return rep


def evaluate_fold(
    name: str,
    seed: int,
    fold: TemporalFold,
    X: pd.DataFrame,
    y: pd.Series,
    ts: pd.Series,
    events: list[FailureEvent],
    neg_stride: int,
) -> dict[str, Any]:
    t0: float = time.perf_counter()
    tr: np.ndarray = _subsample(fold.train_idx, y, neg_stride)
    scores: np.ndarray = _fit_predict(name, seed, X, y, tr, fold.test_idx)
    y_te: np.ndarray = y.to_numpy()[fold.test_idx]
    ts_te: pd.Series = ts.iloc[fold.test_idx].reset_index(drop=True)

    inner_thr, inner_src, oof, oof_y = _inner_threshold(
        name, seed, X, y, ts, fold.train_idx, events, neg_stride
    )
    oracle_thr, _ = cost_optimal_threshold(
        ts_te, scores, events, k=PERSISTENCE_K, **EVAL_KW
    )

    return {
        "model": name,
        "seed": seed,
        "fold": fold.fold,
        "test_event": [str(fold.test_event.start), str(fold.test_event.end)],
        "n_train_rows_used": int(tr.size),
        "n_train_positives": int(y.to_numpy()[tr].sum()),
        "n_test_rows": int(fold.test_idx.size),
        "n_test_positives": int(y_te.sum()),
        "point": {
            "roc_auc": float(roc_auc_score(y_te, scores)),
            "pr_auc": float(average_precision_score(y_te, scores)),
            "f1_at_0.5": float(
                f1_score(y_te, (scores >= 0.5).astype(int), zero_division=0)
            ),
        },
        "event_fixed": _event_block(ts_te, scores, FIXED_THRESHOLD, events),
        "event_inner": {
            **_event_block(ts_te, scores, inner_thr, events),
            "threshold_source": inner_src,
        },
        "event_oracle_upper_bound": _event_block(ts_te, scores, oracle_thr, events),
        "uncertainty": _uncertainty_block(scores, y_te, oof, oof_y),
        "seconds": round(time.perf_counter() - t0, 1),
        "_scores": scores,
        "_ts": ts_te,
        "_y": y_te,
    }


def legacy_random_split(
    X: pd.DataFrame, y: pd.Series, neg_stride: int
) -> list[dict[str, Any]]:
    """The original protocol (stratified, shuffled 80/20) — leakage baseline."""
    idx: np.ndarray = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    tr = _subsample(np.sort(tr), y, neg_stride)
    rows: list[dict[str, Any]] = []
    for name in MODELS:
        scores: np.ndarray = _fit_predict(name, 42, X, y, tr, te)
        y_te: np.ndarray = y.to_numpy()[te]
        rows.append(
            {
                "model": name,
                "roc_auc": float(roc_auc_score(y_te, scores)),
                "pr_auc": float(average_precision_score(y_te, scores)),
                "f1_at_0.5": float(
                    f1_score(y_te, (scores >= 0.5).astype(int), zero_division=0)
                ),
            }
        )
        log.info("[legacy] %s %s", name, rows[-1])
    return rows


# ---------------------------------------------------------------------------
# Aggregation + statistics
# ---------------------------------------------------------------------------

_METRICS: dict[str, tuple[Callable[[dict[str, Any]], float], bool]] = {
    "pr_auc": (lambda r: r["point"]["pr_auc"], True),
    "roc_auc": (lambda r: r["point"]["roc_auc"], True),
    "event_recall_inner": (lambda r: r["event_inner"]["event_recall"], True),
    "range_f1_inner": (lambda r: r["event_inner"]["range_f1"], True),
    "false_alarms_per_day_inner": (
        lambda r: r["event_inner"]["false_alarms_per_day"],
        False,
    ),
    "expected_cost_inner": (lambda r: r["event_inner"]["expected_cost"], False),
}


def _per_block(
    rows: list[dict[str, Any]], getter: Callable[[dict[str, Any]], float]
) -> dict[str, np.ndarray]:
    """Mean over seeds for each (model, fold) -> one value per fold (block)."""
    out: dict[str, np.ndarray] = {}
    folds: list[int] = sorted({r["fold"] for r in rows})
    for name in MODELS:
        out[name] = np.array(
            [
                np.mean(
                    [getter(r) for r in rows if r["model"] == name and r["fold"] == f]
                )
                for f in folds
            ]
        )
    return out


def summarise(rows: list[dict[str, Any]], protocol: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "per_model": {},
        "friedman": {},
        "wilcoxon_rf_vs_xgb": {},
    }
    for name in MODELS:
        mrows = [r for r in rows if r["model"] == name]
        summary["per_model"][name] = {
            metric: bootstrap_ci(np.array([g(r) for r in mrows])).to_dict()
            for metric, (g, _) in _METRICS.items()
        }
        leads = [
            o["lead_time_s"]
            for r in mrows
            for o in r["event_inner"]["outcomes"]
            if o["lead_time_s"] is not None
        ]
        summary["per_model"][name]["lead_time_s_inner"] = bootstrap_ci(
            np.array(leads)
        ).to_dict()

    for metric, (getter, higher) in _METRICS.items():
        blocks = _per_block(rows, getter)
        try:
            res = friedman_nemenyi(blocks, higher_is_better=higher)
        except ValueError as exc:
            summary["friedman"][metric] = {"error": str(exc)}
            continue
        summary["friedman"][metric] = res.to_dict()
        if metric in ("pr_auc", "expected_cost_inner"):
            plot_cd_diagram(
                res, FIG_DIR / f"cd_{protocol}_{metric}.png", f"{protocol} — {metric}"
            )
        summary["wilcoxon_rf_vs_xgb"][metric] = dict(
            zip(
                ("statistic", "p_value"),
                wilcoxon_paired(blocks["random_forest"], blocks["xgboost"]),
            )
        )
    return summary


def plot_scores(
    rows: list[dict[str, Any]], protocol: str, events: list[FailureEvent]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seed0: int = min(r["seed"] for r in rows)
    for f in sorted({r["fold"] for r in rows}):
        fig, ax = plt.subplots(figsize=(11, 3.2))
        for r in rows:
            if r["fold"] != f or r["seed"] != seed0:
                continue
            ax.plot(r["_ts"], r["_scores"], lw=0.6, label=r["model"])
        for ev in events:
            ax.axvspan(ev.start, ev.end, color="tab:red", alpha=0.15)
        ts_any = next(r["_ts"] for r in rows if r["fold"] == f)
        ax.set_xlim(ts_any.min(), ts_any.max())
        ax.axhline(FIXED_THRESHOLD, color="0.5", ls="--", lw=0.8)
        ax.set_ylabel("P(failure)")
        ax.set_title(f"{protocol} — fold {f} (red = real failure)")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / f"scores_{protocol}_fold{f}.png", dpi=150)
        plt.close(fig)


def _strip(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--neg-stride", type=int, default=5)
    parser.add_argument("--skip-legacy", action="store_true")
    args = parser.parse_args()

    started: float = time.perf_counter()
    X, y, ts = load_features()
    events: list[FailureEvent] = events_from_intervals(FAILURE_INTERVALS)
    log.info(
        "Features %s | positives %d | events %d", X.shape, int(y.sum()), len(events)
    )

    result: dict[str, Any] = {
        "schema_version": "1.0",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "requirements": [
            "RF-28",
            "RF-29",
            "RF-31",
            "RNF-79",
            "RN-01",
            "RN-02",
            "RN-03",
            "RN-05",
        ],
        "config": {
            "purge_s": PURGE_S,
            "embargo_s": EMBARGO_S,
            "persistence_k": PERSISTENCE_K,
            "early_horizon_s": EARLY_HORIZON_S,
            "min_lead_s": MIN_LEAD_S,
            "grace_s": GRACE_S,
            "cost_fn": COST_FN,
            "cost_fp": COST_FP,
            "fixed_threshold": FIXED_THRESHOLD,
            "conformal_alpha": CONFORMAL_ALPHA,
            "seeds": list(range(args.seeds)),
            "neg_stride": args.neg_stride,
            "n_features": int(X.shape[1]),
            "models": list(MODELS),
        },
        "protocols": {},
    }

    if not args.skip_legacy:
        result["legacy_random_split"] = legacy_random_split(X, y, args.neg_stride)

    splitters = {
        "walk_forward": PurgedWalkForwardSplit(events, PURGE_S, EMBARGO_S),
        "blocked": PurgedEventBlockKFold(events, PURGE_S, EMBARGO_S),
    }
    for protocol, splitter in splitters.items():
        rows: list[dict[str, Any]] = []
        for fold in splitter.split(ts):
            for name in MODELS:
                for seed in range(args.seeds):
                    row = evaluate_fold(
                        name, seed, fold, X, y, ts, events, args.neg_stride
                    )
                    log.info(
                        "[%s] fold=%d %s seed=%d pr_auc=%.3f recall_ev=%.2f fa/day=%.2f thr=%.2f (%.0fs)",
                        protocol,
                        fold.fold,
                        name,
                        seed,
                        row["point"]["pr_auc"],
                        row["event_inner"]["event_recall"],
                        row["event_inner"]["false_alarms_per_day"],
                        row["event_inner"]["threshold"],
                        row["seconds"],
                    )
                    rows.append(row)
        plot_scores(rows, protocol, events)
        result["protocols"][protocol] = {
            "n_folds": splitter.get_n_splits(),
            "summary": summarise(rows, protocol),
            "runs": _strip(rows),
        }

    result["elapsed_s"] = round(time.perf_counter() - started, 1)
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    log.info("Wrote %s (%.0fs)", OUT_JSON, result["elapsed_s"])


if __name__ == "__main__":
    main()
