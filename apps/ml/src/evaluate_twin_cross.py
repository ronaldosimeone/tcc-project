"""
Digital twin + late fusion across recording periods (MetroPT-3 2020 +
MetroPT 2022) — does physics generalise where pure ML does not?

Entry point:
    python -m src.evaluate_twin_cross [--seeds 3] [--neg-stride 5]

Harmonised sensors only (``Reservoirs`` and ``Oil_level`` dropped — same
name, different signal across datasets; see docs/cross_dataset_evaluation.md).
The twin's mass balance uses ``TP3`` (identical to MetroPT-3 ``Reservoirs``,
valid on 2022).

ML hygiene (same rules as every other evaluation in the thesis)
--------------------------------------------------------------
* Class balance is handled ONLY in training: every failure row + every
  ``neg_stride``-th normal row, plus class weights (RF ``balanced_subsample``,
  XGBoost ``scale_pos_weight``, LogReg ``balanced``). Test data keeps the real
  prevalence — balancing it would inflate every metric.
* Thresholds are cost-optimal (RN-02) on training out-of-fold scores; the
  test period is never used to choose anything.
* The twin is fitted ONLY on healthy rows of the training period, separately
  per dataset (per-asset calibration). Rolling windows never cross datasets.
* Alarms need k = 6 consecutive samples (RN-03); event metrics (RF-29).

Setting A — ``transfer``: target 2022 with NO failure labels from it.
    Training = all of 2020 + the first weeks of 2022 (``CALIB_END``, normal
    operation only, ends two weeks before the first 2022 failure). Test =
    the rest of 2022 (3 failures). 2×2 ablation:
        ml_2020_only          XGBoost trained on 2020 only
        ml_only               XGBoost trained on 2020 + 2022 normal weeks
        physics_uncalibrated  LogReg on residuals, 2022 twin = 2020 twin
        physics_only          LogReg on residuals, 2022 twin fitted on the
                              2022 normal weeks
        hybrid                XGBoost on features + residuals
        fusion                ml_only OR physics_only (own thresholds)
Setting B — ``pooled_walk_forward``: purged walk-forward over the joint
    timeline (7 failures, 6 folds); per fold the twin of each dataset is
    fitted on that dataset's healthy TRAINING rows. Variants ml_only,
    physics_only, hybrid, fusion. RN-05 gate fusion × ml_only.

Output: models/eval_twin_cross.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ML_ROOT))

import src.evaluate_cross_dataset as xd  # noqa: E402
from src.digital_twin import TWIN_FEATURES, PhysicsTwin  # noqa: E402
from src.evaluate_temporal import (  # noqa: E402
    EMBARGO_S,
    EVAL_KW,
    PERSISTENCE_K,
    PURGE_S,
    evaluate_fold,
)
from src.event_metrics import evaluate_events, persistent_alarms  # noqa: E402
from src.ingest_metropt import FAILURE_INTERVALS  # noqa: E402
from src.ingest_metropt2022 import FAILURE_INTERVALS_2022  # noqa: E402
from src.promotion_gate import decide  # noqa: E402
from src.stat_tests import bootstrap_ci  # noqa: E402
from src.temporal_validation import (  # noqa: E402
    PurgedWalkForwardSplit,
    TemporalFold,
    events_from_intervals,
)

OUT_JSON: Path = _ML_ROOT / "models" / "eval_twin_cross.json"
DROPPED: list[str] = ["Reservoirs", "Oil_level"]
PRESSURE_COL: str = "TP3"
CALIB_END: pd.Timestamp = pd.Timestamp("2022-02-14 00:00")

MODEL_OF: dict[str, str] = {
    "ml_2020_only": "xgboost",
    "ml_only": "xgboost",
    "hybrid": "xgboost",
    "physics_only": "logistic_regression",
    "physics_uncalibrated": "logistic_regression",
}


def _twin_residuals(
    raw: pd.DataFrame,
    is22: np.ndarray,
    fit20: np.ndarray,
    fit22: np.ndarray | None,
) -> pd.DataFrame:
    """Residuals for the whole joint series. ``fit20``/``fit22`` are boolean
    masks of the healthy TRAINING rows used to calibrate each dataset's twin;
    ``fit22=None`` reuses the 2020 twin on 2022 (no calibration)."""
    twin20 = PhysicsTwin(pressure_col=PRESSURE_COL).fit(raw[fit20])
    twin22 = (
        PhysicsTwin(pressure_col=PRESSURE_COL).fit(raw[fit22])
        if fit22 is not None and fit22.sum() > 0
        else twin20
    )
    parts = [
        twin20.transform(raw[~is22].reset_index(drop=True)),
        twin22.transform(raw[is22].reset_index(drop=True)),
    ]
    out = pd.concat(parts, ignore_index=True).astype(np.float32)
    return out


def _fusion_row(
    ml: dict[str, Any], ph: dict[str, Any], events: list[Any], seed: int, fold: int
) -> dict[str, Any]:
    alarms = persistent_alarms(
        ml["_scores"], ml["event_inner"]["threshold"], PERSISTENCE_K
    ) | persistent_alarms(ph["_scores"], ph["event_inner"]["threshold"], PERSISTENCE_K)
    rep = evaluate_events(ml["_ts"], alarms, events, **EVAL_KW).to_dict()
    return {
        "model": "fusion",
        "variant": "fusion",
        "seed": seed,
        "fold": fold,
        "point": {"pr_auc": float("nan")},
        "event_inner": rep,
    }


def _summary(runs: list[dict[str, Any]], variants: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for v in variants:
        rs = [r for r in runs if r["variant"] == v]
        if not rs:
            continue
        leads = [
            o["lead_time_s"]
            for r in rs
            for o in r["event_inner"]["outcomes"]
            if o["lead_time_s"] is not None
        ]
        out[v] = {
            "pr_auc": bootstrap_ci(
                np.array([r["point"]["pr_auc"] for r in rs], dtype=float)
            ).to_dict(),
            "event_recall": bootstrap_ci(
                np.array([r["event_inner"]["event_recall"] for r in rs])
            ).to_dict(),
            "false_alarms_per_day": bootstrap_ci(
                np.array([r["event_inner"]["false_alarms_per_day"] for r in rs])
            ).to_dict(),
            "expected_cost": bootstrap_ci(
                np.array([r["event_inner"]["expected_cost"] for r in rs])
            ).to_dict(),
            "lead_time_s": (
                bootstrap_ci(np.array(leads, dtype=float)).to_dict() if leads else None
            ),
            "detected_per_event": _detected_per_event(rs),
        }
    return out


def _detected_per_event(rs: list[dict[str, Any]]) -> dict[str, float]:
    """Share of runs (seeds) that detected each failure, keyed by its start."""
    hits: dict[str, list[bool]] = {}
    for r in rs:
        for o in r["event_inner"]["outcomes"]:
            hits.setdefault(str(o["event_start"]), []).append(bool(o["detected"]))
    return {k: float(np.mean(v)) for k, v in sorted(hits.items())}


def _physics_z(
    resid: pd.DataFrame,
    y: pd.Series,
    ts: pd.Series,
    fold: TemporalFold,
    events: list[Any],
) -> dict[str, Any]:
    """Mean z-score of each residual inside every test failure, relative to
    the healthy training rows — does the physics move when it fails?"""
    tr = fold.train_idx[y.to_numpy()[fold.train_idx] == 0]
    mu, sd = resid.iloc[tr].mean(), resid.iloc[tr].std() + 1e-9
    out: dict[str, Any] = {}
    t_te = ts.iloc[fold.test_idx]
    for ev in events:
        inside = fold.test_idx[((t_te >= ev.start) & (t_te <= ev.end)).to_numpy()]
        if inside.size:
            z = ((resid.iloc[inside] - mu) / sd).mean()
            out[str(ev.start)] = {k: round(float(v), 2) for k, v in z.items()}
    return out


def setting_transfer(
    X: pd.DataFrame,
    y: pd.Series,
    ts: pd.Series,
    is22: np.ndarray,
    raw: pd.DataFrame,
    events: list[Any],
    seeds: int,
    neg_stride: int,
) -> dict[str, Any]:
    i20 = np.flatnonzero(~is22)
    calib = np.flatnonzero(is22 & (ts < CALIB_END).to_numpy())
    test = np.flatnonzero(
        is22 & (ts >= CALIB_END + pd.Timedelta(seconds=PURGE_S)).to_numpy()
    )
    if y.to_numpy()[calib].sum() != 0:
        raise RuntimeError("calibration window must contain only normal rows")
    healthy = y.to_numpy() == 0
    resid_cal = _twin_residuals(
        raw, is22, ~is22 & healthy, np.isin(np.arange(len(raw)), calib)
    )
    resid_unc = _twin_residuals(raw, is22, ~is22 & healthy, None)
    ev22 = events_from_intervals(FAILURE_INTERVALS_2022)

    def fold(train: np.ndarray) -> TemporalFold:
        return TemporalFold(
            fold=0,
            train_idx=train,
            test_idx=test,
            test_event=ev22[0],
            test_start=ts.iloc[test].min(),
            test_end=ts.iloc[test].max(),
        )

    plan: dict[str, tuple[pd.DataFrame, np.ndarray]] = {
        "ml_2020_only": (X, i20),
        "ml_only": (X, np.r_[i20, calib]),
        "hybrid": (pd.concat([X, resid_cal], axis=1), np.r_[i20, calib]),
        "physics_only": (resid_cal, np.r_[i20, calib]),
        "physics_uncalibrated": (resid_unc, np.r_[i20, calib]),
    }
    runs: list[dict[str, Any]] = []
    for seed in range(seeds):
        full: dict[str, dict[str, Any]] = {}
        for variant, (matrix, train) in plan.items():
            r = evaluate_fold(
                MODEL_OF[variant], seed, fold(train), matrix, y, ts, events, neg_stride
            )
            r["variant"] = variant
            full[variant] = r
            print(
                f"[transfer] {variant} seed={seed} pr_auc={r['point']['pr_auc']:.3f} "
                f"recall_ev={r['event_inner']['event_recall']:.2f} "
                f"fa/day={r['event_inner']['false_alarms_per_day']:.2f}",
                flush=True,
            )
        runs += [
            {k: v for k, v in r.items() if not k.startswith("_")} for r in full.values()
        ]
        f = _fusion_row(full["ml_only"], full["physics_only"], events, seed, 0)
        runs.append(f)
        print(
            f"[transfer] fusion seed={seed} recall_ev={f['event_inner']['event_recall']:.2f} "
            f"fa/day={f['event_inner']['false_alarms_per_day']:.2f}",
            flush=True,
        )
    z = _physics_z(resid_cal, y, ts, fold(np.r_[i20, calib]), ev22)
    return {
        "calibration_window": [str(ts.iloc[calib].min()), str(CALIB_END)],
        "test_window": [str(ts.iloc[test].min()), str(ts.iloc[test].max())],
        "n_calibration_rows": int(calib.size),
        "physics_z_in_2022_failures": z,
        "summary": _summary(runs, [*MODEL_OF, "fusion"]),
        "runs": runs,
    }


def setting_pooled(
    X: pd.DataFrame,
    y: pd.Series,
    ts: pd.Series,
    is22: np.ndarray,
    raw: pd.DataFrame,
    events: list[Any],
    seeds: int,
    neg_stride: int,
) -> dict[str, Any]:
    healthy = y.to_numpy() == 0
    runs: list[dict[str, Any]] = []
    z_by_fold: dict[str, Any] = {}
    splitter = PurgedWalkForwardSplit(events, PURGE_S, EMBARGO_S)
    for fold in splitter.split(ts):
        in_train = np.zeros(len(raw), bool)
        in_train[fold.train_idx] = True
        fit22 = in_train & is22 & healthy
        resid = _twin_residuals(
            raw, is22, in_train & ~is22 & healthy, fit22 if fit22.any() else None
        )
        z_by_fold[str(fold.fold)] = _physics_z(resid, y, ts, fold, events)
        matrices = {
            "ml_only": X,
            "hybrid": pd.concat([X, resid], axis=1),
            "physics_only": resid,
        }
        for seed in range(seeds):
            full: dict[str, dict[str, Any]] = {}
            for variant, matrix in matrices.items():
                r = evaluate_fold(
                    MODEL_OF[variant], seed, fold, matrix, y, ts, events, neg_stride
                )
                r["variant"] = variant
                r["test_dataset"] = "2022" if is22[fold.test_idx].all() else "2020"
                full[variant] = r
            runs += [
                {k: v for k, v in r.items() if not k.startswith("_")}
                for r in full.values()
            ]
            f = _fusion_row(
                full["ml_only"], full["physics_only"], events, seed, fold.fold
            )
            f["test_dataset"] = full["ml_only"]["test_dataset"]
            runs.append(f)
            print(
                f"[pooled] fold={fold.fold} ({f['test_dataset']}) seed={seed} "
                + " ".join(
                    f"{v}={full[v]['event_inner']['event_recall']:.2f}" for v in full
                )
                + f" fusion={f['event_inner']['event_recall']:.2f}",
                flush=True,
            )

    # RN-05 gate needs runs keyed by "model": use the variant name.
    eval_like = {
        "protocols": {
            "pooled_twin": {"runs": [{**r, "model": r["variant"]} for r in runs]}
        }
    }
    gate = {
        f"{c}_vs_{b}": decide(
            eval_like, c, b, metric="expected_cost_inner", protocol="pooled_twin"
        ).to_dict()
        for c, b in (
            ("fusion", "ml_only"),
            ("hybrid", "ml_only"),
            ("fusion", "physics_only"),
        )
    }
    return {
        "n_folds": splitter.get_n_splits(),
        "physics_z_in_failures_by_fold": z_by_fold,
        "summary": _summary(runs, ["ml_only", "physics_only", "hybrid", "fusion"]),
        "summary_2022_folds": _summary(
            [r for r in runs if r.get("test_dataset") == "2022"],
            ["ml_only", "physics_only", "hybrid", "fusion"],
        ),
        "promotion_gate_rn05": gate,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--neg-stride", type=int, default=5)
    parser.add_argument(
        "--only",
        nargs="*",
        choices=["transfer", "pooled"],
        default=["transfer", "pooled"],
    )
    args = parser.parse_args()
    started = time.perf_counter()

    xd.configure(DROPPED)
    X, y, ts, is22 = xd.load_pooled()
    raw = pd.concat([xd._read(xd.P2020, []), xd._read(xd.P2022, [])], ignore_index=True)
    events = events_from_intervals([*FAILURE_INTERVALS, *FAILURE_INTERVALS_2022])

    result: dict[str, Any] = {}
    if OUT_JSON.exists():
        result = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    result.update(
        {
            "schema_version": "1.0",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "requirements": ["RF-33", "RN-05", "RNF-79"],
            "config": {
                "seeds": list(range(args.seeds)),
                "neg_stride": args.neg_stride,
                "dropped_sensors": DROPPED,
                "features": list(xd.FEATURES),
                "twin_pressure_col": PRESSURE_COL,
                "twin_features": list(TWIN_FEATURES),
                "class_balance": "train only: all positives + every neg_stride-th "
                "negative, class weights; test at real prevalence",
            },
        }
    )
    if "transfer" in args.only:
        result["transfer"] = setting_transfer(
            X, y, ts, is22, raw, events, args.seeds, args.neg_stride
        )
    if "pooled" in args.only:
        result["pooled_walk_forward"] = setting_pooled(
            X, y, ts, is22, raw, events, args.seeds, args.neg_stride
        )
    result["elapsed_s"] = round(time.perf_counter() - started, 1)
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {OUT_JSON} ({result['elapsed_s']:.0f}s)")


if __name__ == "__main__":
    main()
