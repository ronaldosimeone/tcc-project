"""
Cross-dataset generalisation and data pooling — RF-33.

MetroPT-3 (2020, 4 failures) + MetroPT 2022 (3 failures), same APU compressor
family, harmonised by ``src.ingest_metropt2022``.

Entry point:
    python -m src.evaluate_cross_dataset [--seeds 3] [--neg-stride 5]

Questions and the experiment that answers each
----------------------------------------------
Q1  How different are the two periods?            ``shift``
    Per-sensor PSI / KS / mean ± std on NORMAL rows, and adversarial
    validation: a classifier separating 2020 from 2022 normal rows, scored by
    ROC-AUC on time-blocked folds (0.5 = indistinguishable).
Q2  Does a model trained on one period work on the other?   ``cross``
    Train 2020 → test 2022 and train 2022 → test 2020. Threshold chosen on
    the TRAINING data only (inner OOF, RN-02) — never on the target period.
Q3  Is pooling worth it?                          ``pooled_walk_forward``
    Purged walk-forward over the joint timeline (7 failures → 6 folds).
    For the 2022 folds that have earlier 2022 failures, the same fold is also
    trained on 2022 history only (``target_only``) — the direct pooled vs
    not-pooled comparison. RN-05: with 6 folds the paired Wilcoxon can reach
    p = 1/64, so the promotion gate is no longer underpowered by design.
Q4  What does dropping ``Flowmeter`` cost?        ``flowmeter_ablation``
    2022 only (MetroPT-3 has no flow meter), blocked k-fold, same models
    with and without the Flowmeter-derived features.

Output: models/eval_cross_dataset.json, docs/figures-ml/cd_pooled_*.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.drift_detectors import psi
from src.evaluate_temporal import (
    EMBARGO_S,
    MODELS,
    PURGE_S,
    _strip,
    evaluate_fold,
    summarise,
)
from src.ingest_metropt import FAILURE_INTERVALS
from src.ingest_metropt2022 import FAILURE_INTERVALS_2022, OUT_PARQUET as P2022
from src.preprocessing import _DEFAULT_SENSOR_COLS, MetroPTPreprocessor
from src.promotion_gate import decide
from src.stat_tests import bootstrap_ci, wilcoxon_paired
from src.temporal_validation import (
    FailureEvent,
    PurgedEventBlockKFold,
    PurgedWalkForwardSplit,
    TemporalFold,
    events_from_intervals,
)
from src.train_random_forest import FEATURE_COLS, TARGET_COL

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
P2020: Path = _ML_ROOT / "data" / "processed" / "metropt3.parquet"
OUT_JSON: Path = _ML_ROOT / "models" / "eval_cross_dataset.json"

ANALOG: list[str] = list(_DEFAULT_SENSOR_COLS)
# Sensors entering every model/shift test. ``--drop-sensors`` removes the ones
# whose MEANING differs between the datasets (same name, different signal):
# found by the shift block — see docs/cross_dataset_evaluation.md.
FEATURES: list[str] = list(FEATURE_COLS)


def configure(drop: list[str]) -> None:
    unknown = sorted(set(drop) - set(FEATURE_COLS))
    if unknown:
        raise SystemExit(f"--drop-sensors: unknown sensors {unknown}")
    FEATURES[:] = [c for c in FEATURE_COLS if c not in drop]
    ANALOG[:] = [c for c in _DEFAULT_SENSOR_COLS if c not in drop]


SHIFT_SAMPLE: int = 60_000
SHIFT_BLOCKS: int = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log: logging.Logger = logging.getLogger("evaluate_cross_dataset")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _read(path: Path, extra: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    keep = ["timestamp", *FEATURES, *extra, TARGET_COL]
    return df[keep].sort_values("timestamp").reset_index(drop=True)


def _features(df: pd.DataFrame, extra: list[str]) -> pd.DataFrame:
    """Rolling features per dataset: a window never spans the 2020→2022 gap."""
    pre = MetroPTPreprocessor(sensor_cols=ANALOG + extra)
    return pre.fit_transform(df[FEATURES + extra].copy()).astype(np.float32)


def load_pooled() -> tuple[pd.DataFrame, pd.Series, pd.Series, np.ndarray]:
    """Joint timeline. Returns X, y, ts and a boolean mask of 2022 rows."""
    d20, d22 = _read(P2020, []), _read(P2022, [])
    X = pd.concat([_features(d20, []), _features(d22, [])], ignore_index=True)
    y = pd.concat([d20[TARGET_COL], d22[TARGET_COL]], ignore_index=True).astype(int)
    ts = pd.concat([d20["timestamp"], d22["timestamp"]], ignore_index=True)
    is22 = np.r_[np.zeros(len(d20), bool), np.ones(len(d22), bool)]
    return X, y, ts, is22


# ---------------------------------------------------------------------------
# Q1 — distribution shift
# ---------------------------------------------------------------------------


def shift_report(seed: int = 0) -> dict[str, Any]:
    d20, d22 = _read(P2020, []), _read(P2022, [])
    n20, n22 = d20[d20[TARGET_COL] == 0], d22[d22[TARGET_COL] == 0]
    per_sensor: dict[str, Any] = {}
    for col in FEATURES:
        a, b = n20[col].to_numpy(float), n22[col].to_numpy(float)
        per_sensor[col] = {
            "mean_2020": float(a.mean()),
            "std_2020": float(a.std()),
            "mean_2022": float(b.mean()),
            "std_2022": float(b.std()),
            "psi": float(psi(a, b)) if col in ANALOG else None,
            "ks": float(ks_2samp(a[::20], b[::20]).statistic),
        }

    # Adversarial validation on raw sensors. Time-blocked folds: each dataset
    # is cut in SHIFT_BLOCKS contiguous blocks, so neighbouring (near-duplicate)
    # rows never sit on both sides of a split.
    rng = np.random.default_rng(seed)
    parts = []
    for label_, frame in ((0, n20), (1, n22)):
        idx = np.sort(
            rng.choice(len(frame), min(SHIFT_SAMPLE, len(frame)), replace=False)
        )
        part = frame.iloc[idx][FEATURES].copy()
        part["_y"] = label_
        part["_block"] = np.minimum(idx * SHIFT_BLOCKS // len(frame), SHIFT_BLOCKS - 1)
        parts.append(part)
    data = pd.concat(parts, ignore_index=True)
    aucs: list[float] = []
    importances = np.zeros(len(FEATURES))
    for b in range(SHIFT_BLOCKS):
        tr, te = data["_block"] != b, data["_block"] == b
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=12, n_jobs=-1, random_state=seed
        )
        clf.fit(data.loc[tr, FEATURES], data.loc[tr, "_y"])
        aucs.append(
            float(
                roc_auc_score(
                    data.loc[te, "_y"],
                    clf.predict_proba(data.loc[te, FEATURES])[:, 1],
                )
            )
        )
        importances += clf.feature_importances_ / SHIFT_BLOCKS
    ranked = sorted(zip(FEATURES, importances.tolist()), key=lambda t: -t[1])
    return {
        "per_sensor_normal_rows": per_sensor,
        "adversarial_validation": {
            "auc_per_block": aucs,
            "auc_mean": float(np.mean(aucs)),
            "top_features": [{"feature": f, "importance": v} for f, v in ranked[:5]],
            "reading": "0.5 = periods indistinguishable; ~1.0 = trivially separable",
        },
    }


# ---------------------------------------------------------------------------
# Q2 — cross-period transfer
# ---------------------------------------------------------------------------


def _whole_period_fold(
    fold_id: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    ts: pd.Series,
    test_events: list[FailureEvent],
) -> TemporalFold:
    return TemporalFold(
        fold=fold_id,
        train_idx=train_idx,
        test_idx=test_idx,
        test_event=test_events[0],
        test_start=ts.iloc[test_idx].min(),
        test_end=ts.iloc[test_idx].max(),
    )


def cross_transfer(
    X: pd.DataFrame,
    y: pd.Series,
    ts: pd.Series,
    is22: np.ndarray,
    events: list[FailureEvent],
    seeds: int,
    neg_stride: int,
) -> dict[str, Any]:
    i20, i22 = np.flatnonzero(~is22), np.flatnonzero(is22)
    ev20 = events_from_intervals(FAILURE_INTERVALS)
    ev22 = events_from_intervals(FAILURE_INTERVALS_2022)
    out: dict[str, Any] = {}
    for direction, tr, te, tev in (
        ("2020_to_2022", i20, i22, ev22),
        ("2022_to_2020", i22, i20, ev20),
    ):
        fold = _whole_period_fold(0, tr, te, ts, tev)
        rows = []
        for name in MODELS:
            for seed in range(seeds):
                row = evaluate_fold(name, seed, fold, X, y, ts, events, neg_stride)
                log.info(
                    "[cross %s] %s seed=%d pr_auc=%.3f recall_ev=%.2f fa/day=%.2f",
                    direction,
                    name,
                    seed,
                    row["point"]["pr_auc"],
                    row["event_inner"]["event_recall"],
                    row["event_inner"]["false_alarms_per_day"],
                )
                rows.append(row)
        out[direction] = {
            "per_model": {
                name: {
                    "pr_auc": bootstrap_ci(
                        np.array(
                            [r["point"]["pr_auc"] for r in rows if r["model"] == name]
                        )
                    ).to_dict(),
                    "event_recall_inner": bootstrap_ci(
                        np.array(
                            [
                                r["event_inner"]["event_recall"]
                                for r in rows
                                if r["model"] == name
                            ]
                        )
                    ).to_dict(),
                    "false_alarms_per_day_inner": bootstrap_ci(
                        np.array(
                            [
                                r["event_inner"]["false_alarms_per_day"]
                                for r in rows
                                if r["model"] == name
                            ]
                        )
                    ).to_dict(),
                    "expected_cost_inner": bootstrap_ci(
                        np.array(
                            [
                                r["event_inner"]["expected_cost"]
                                for r in rows
                                if r["model"] == name
                            ]
                        )
                    ).to_dict(),
                }
                for name in MODELS
            },
            "runs": _strip(rows),
        }
    return out


# ---------------------------------------------------------------------------
# Q3 — pooled walk-forward and pooled vs target-only
# ---------------------------------------------------------------------------


def pooled_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    ts: pd.Series,
    is22: np.ndarray,
    events: list[FailureEvent],
    seeds: int,
    neg_stride: int,
    protocol: str = "pooled_walk_forward",
) -> dict[str, Any]:
    splitter = PurgedWalkForwardSplit(events, PURGE_S, EMBARGO_S)
    rows: list[dict[str, Any]] = []
    target_only: list[dict[str, Any]] = []
    for fold in splitter.split(ts):
        test_is22 = bool(is22[fold.test_idx].all())
        tr22 = fold.train_idx[is22[fold.train_idx]]
        has_22_history = test_is22 and y.to_numpy()[tr22].sum() > 0
        for name in MODELS:
            for seed in range(seeds):
                row = evaluate_fold(name, seed, fold, X, y, ts, events, neg_stride)
                row["test_dataset"] = "2022" if test_is22 else "2020"
                rows.append(row)
                log.info(
                    "[pooled] fold=%d (%s) %s seed=%d pr_auc=%.3f recall_ev=%.2f cost=%.1f",
                    fold.fold,
                    row["test_dataset"],
                    name,
                    seed,
                    row["point"]["pr_auc"],
                    row["event_inner"]["event_recall"],
                    row["event_inner"]["expected_cost"],
                )
                if has_22_history:
                    tfold = TemporalFold(
                        fold=fold.fold,
                        train_idx=tr22,
                        test_idx=fold.test_idx,
                        test_event=fold.test_event,
                        test_start=fold.test_start,
                        test_end=fold.test_end,
                    )
                    trow = evaluate_fold(
                        name, seed, tfold, X, y, ts, events, neg_stride
                    )
                    trow["variant"] = "target_only_2022"
                    target_only.append(trow)

    eval_like = {"protocols": {protocol: {"runs": _strip(rows)}}}
    gate = {
        f"{cand}_vs_{base}": decide(
            eval_like,
            cand,
            base,
            metric="expected_cost_inner",
            protocol=protocol,
        ).to_dict()
        for cand, base in (
            ("xgboost", "random_forest"),
            ("random_forest", "logistic_regression"),
            ("xgboost", "logistic_regression"),
        )
    }

    # Pooled vs target-only on the folds where both exist (same model/seed/fold).
    paired: dict[str, Any] = {}
    for name in MODELS:
        key = {(r["fold"], r["seed"]): r for r in rows if r["model"] == name}
        pairs = [
            (key[(t["fold"], t["seed"])], t)
            for t in target_only
            if t["model"] == name and (t["fold"], t["seed"]) in key
        ]
        if not pairs:
            continue
        pooled_cost = np.array([p["event_inner"]["expected_cost"] for p, _ in pairs])
        target_cost = np.array([t["event_inner"]["expected_cost"] for _, t in pairs])
        paired[name] = {
            "n_pairs": len(pairs),
            "folds": sorted({t["fold"] for _, t in pairs}),
            "pooled_cost_mean": float(pooled_cost.mean()),
            "target_only_cost_mean": float(target_cost.mean()),
            "pooled_pr_auc_mean": float(
                np.mean([p["point"]["pr_auc"] for p, _ in pairs])
            ),
            "target_only_pr_auc_mean": float(
                np.mean([t["point"]["pr_auc"] for _, t in pairs])
            ),
            "pooled_event_recall_mean": float(
                np.mean([p["event_inner"]["event_recall"] for p, _ in pairs])
            ),
            "target_only_event_recall_mean": float(
                np.mean([t["event_inner"]["event_recall"] for _, t in pairs])
            ),
            "wilcoxon_cost": dict(
                zip(("statistic", "p_value"), wilcoxon_paired(pooled_cost, target_cost))
            ),
        }

    return {
        "n_folds": splitter.get_n_splits(),
        "summary": summarise(rows, protocol),
        "per_test_dataset": {
            ds: {
                name: bootstrap_ci(
                    np.array(
                        [
                            r["event_inner"]["expected_cost"]
                            for r in rows
                            if r["model"] == name and r["test_dataset"] == ds
                        ]
                    )
                ).to_dict()
                for name in MODELS
            }
            for ds in ("2020", "2022")
        },
        "promotion_gate_rn05": gate,
        "pooled_vs_target_only": paired,
        "runs": _strip(rows),
        "target_only_runs": _strip(target_only),
    }


# ---------------------------------------------------------------------------
# Q4 — Flowmeter ablation (2022 only)
# ---------------------------------------------------------------------------


def flowmeter_ablation(seeds: int, neg_stride: int) -> dict[str, Any]:
    d22 = _read(P2022, ["Flowmeter"])
    y = d22[TARGET_COL].astype(int)
    ts = d22["timestamp"]
    events = events_from_intervals(FAILURE_INTERVALS_2022)
    variants = {
        "without_flowmeter": _features(d22, []),
        "with_flowmeter": _features(d22, ["Flowmeter"]),
    }
    folds = list(PurgedEventBlockKFold(events, PURGE_S, EMBARGO_S).split(ts))
    out: dict[str, Any] = {"protocol": "blocked (3 folds, 2022 only)", "variants": {}}
    by_variant: dict[str, dict[tuple[str, int, int], dict[str, Any]]] = {}
    for variant, X in variants.items():
        rows = []
        for fold in folds:
            for name in ("random_forest", "xgboost"):
                for seed in range(seeds):
                    row = evaluate_fold(name, seed, fold, X, y, ts, events, neg_stride)
                    rows.append(row)
                    log.info(
                        "[flowmeter %s] fold=%d %s seed=%d pr_auc=%.3f recall_ev=%.2f",
                        variant,
                        fold.fold,
                        name,
                        seed,
                        row["point"]["pr_auc"],
                        row["event_inner"]["event_recall"],
                    )
        by_variant[variant] = {(r["model"], r["fold"], r["seed"]): r for r in rows}
        out["variants"][variant] = {
            "n_features": int(X.shape[1]),
            "runs": _strip(rows),
        }
    deltas: dict[str, Any] = {}
    for name in ("random_forest", "xgboost"):
        keys = [k for k in by_variant["with_flowmeter"] if k[0] == name]
        for metric, getter in (
            ("pr_auc", lambda r: r["point"]["pr_auc"]),
            ("event_recall_inner", lambda r: r["event_inner"]["event_recall"]),
            ("expected_cost_inner", lambda r: r["event_inner"]["expected_cost"]),
        ):
            w = np.array([getter(by_variant["with_flowmeter"][k]) for k in keys])
            wo = np.array([getter(by_variant["without_flowmeter"][k]) for k in keys])
            deltas.setdefault(name, {})[metric] = {
                "with": float(w.mean()),
                "without": float(wo.mean()),
                "delta_with_minus_without": bootstrap_ci(w - wo).to_dict(),
            }
    out["deltas"] = deltas
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--neg-stride", type=int, default=5)
    parser.add_argument(
        "--drop-sensors",
        nargs="*",
        default=[],
        help="sensors excluded from every model (semantic mismatch between datasets)",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="suffix of the output file: eval_cross_dataset_<tag>.json",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=["shift", "cross", "pooled", "flowmeter"],
        default=["shift", "cross", "pooled", "flowmeter"],
    )
    args = parser.parse_args()
    started = time.perf_counter()
    configure(args.drop_sensors)
    out_json = (
        OUT_JSON.with_name(f"eval_cross_dataset_{args.tag}.json")
        if args.tag
        else OUT_JSON
    )

    result: dict[str, Any] = {}
    if out_json.exists():  # --only re-runs one block and keeps the others
        result = json.loads(out_json.read_text(encoding="utf-8"))
    result.update(
        {
            "schema_version": "1.0",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "requirements": ["RF-33", "RN-05", "RNF-79"],
            "datasets": {
                "metropt3_2020": {"failures": len(FAILURE_INTERVALS)},
                "metropt_2022": {
                    "failures": len(FAILURE_INTERVALS_2022),
                    "source": "doi:10.5281/zenodo.6854240",
                },
            },
            "config": {
                "seeds": list(range(args.seeds)),
                "neg_stride": args.neg_stride,
                "purge_s": PURGE_S,
                "embargo_s": EMBARGO_S,
                "models": list(MODELS),
                "features": list(FEATURES),
                "dropped_sensors": list(args.drop_sensors),
            },
        }
    )

    if "shift" in args.only:
        result["shift"] = shift_report()
        log.info(
            "adversarial AUC 2020 vs 2022 = %.3f",
            result["shift"]["adversarial_validation"]["auc_mean"],
        )
    if {"cross", "pooled"} & set(args.only):
        X, y, ts, is22 = load_pooled()
        events = events_from_intervals([*FAILURE_INTERVALS, *FAILURE_INTERVALS_2022])
        log.info(
            "pooled: %s rows, %d positives, %d events",
            X.shape,
            int(y.sum()),
            len(events),
        )
        if "cross" in args.only:
            result["cross"] = cross_transfer(
                X, y, ts, is22, events, args.seeds, args.neg_stride
            )
        if "pooled" in args.only:
            result["pooled_walk_forward"] = pooled_walk_forward(
                X,
                y,
                ts,
                is22,
                events,
                args.seeds,
                args.neg_stride,
                protocol=(
                    f"pooled_walk_forward_{args.tag}"
                    if args.tag
                    else "pooled_walk_forward"
                ),
            )
    if "flowmeter" in args.only:
        result["flowmeter_ablation"] = flowmeter_ablation(args.seeds, args.neg_stride)

    result["elapsed_s"] = round(time.perf_counter() - started, 1)
    out_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    log.info("Wrote %s (%.0fs)", out_json, result["elapsed_s"])


if __name__ == "__main__":
    main()
