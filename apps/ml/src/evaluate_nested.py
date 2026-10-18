"""
Doctoral-grade evaluation of every servable model — nested, calibrated,
multi-seed, out-of-time.

Entry point:
    python -m src.evaluate_nested --models random_forest xgboost ... [--seeds 3]

Outer loop: purged walk-forward by failure event over MetroPT-3 + MetroPT 2022
(harmonised sensors, 7 failures → 6 test folds). Inner loop (``nested_protocol``):
leave-one-failure-out on the training data for hyper-parameter selection,
Platt calibration and the RN-02 threshold. Seed 0 tunes; seeds 1..n−1 reuse
the per-fold choice of seed 0 (they measure training variance, not tuning).

Per model and seed it records, on the out-of-time test rows only:
confusion matrices (per sample at the threshold, per persisted alarm, per
event), PR-AUC, ECE of the raw vs calibrated probability, the selected
hyper-parameters and the inner search trace.

Output: models/nested/<model>.json (one file per model, so CPU and GPU
models can run in parallel processes); figures by ``render_nested_report``.
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

import src.deep_temporal as dt
import src.evaluate_cross_dataset as xd
import src.nested_protocol as npc
from src.confusion import add as _add
from src.confusion import fold_report
from src.confusion import metrics as _metrics
from src.evaluate_temporal import EMBARGO_S, PURGE_S
from src.ingest_metropt import FAILURE_INTERVALS
from src.ingest_metropt2022 import FAILURE_INTERVALS_2022
from src.temporal_validation import PurgedWalkForwardSplit, events_from_intervals
from src.uncertainty import calibration_report

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
OUT_DIR: Path = _ML_ROOT / "models" / "nested"
# Per-run OOF + test raw scores (gitignored, regenerable) for the
# sensitivity analysis (``evaluate_sensitivity``).
SCORES_DIR: Path = OUT_DIR / "scores"
DROPPED: list[str] = ["Reservoirs", "Oil_level"]
ALL_MODELS: list[str] = ["random_forest", "xgboost", "logistic_regression", *dt.ARCHS]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("evaluate_nested")


def build_context() -> tuple[npc.Context, np.ndarray]:
    xd.configure(DROPPED)
    X, y, ts, is22 = xd.load_pooled()
    raw_df = pd.concat(
        [xd._read(xd.P2020, []), xd._read(xd.P2022, [])], ignore_index=True
    )[["timestamp", *xd.FEATURES, "anomaly"]]
    data = dt.build_data(raw_df, X, y, ts, is22)
    events = events_from_intervals([*FAILURE_INTERVALS, *FAILURE_INTERVALS_2022])
    return npc.Context(X=X, y=y, ts=ts, events=events, data=data), is22


def _ece(y: np.ndarray, p: np.ndarray) -> float | None:
    if np.isnan(p).any() or not y.any():
        return None
    return float(calibration_report(y, p).ece)


def run_model(
    ctx: npc.Context, is22: np.ndarray, name: str, seeds: int
) -> dict[str, Any]:
    folds = list(PurgedWalkForwardSplit(ctx.events, PURGE_S, EMBARGO_S).split(ctx.ts))
    y = ctx.y.to_numpy()
    runs: list[dict[str, Any]] = []
    chosen: dict[int, dict[str, Any]] = {}
    for seed in range(seeds):
        for fold in folds:
            t0 = time.perf_counter()
            sel = npc.select(
                ctx,
                name,
                fold.train_idx,
                seed,
                tune=seed == 0,
                fixed_params=chosen.get(fold.fold),
            )
            if seed == 0:
                chosen[fold.fold] = sel.params
            model = npc.fit_model(ctx, name, fold.train_idx, seed, sel.params)
            raw = npc.raw_score(ctx, name, model, fold.test_idx)
            y_te = y[fold.test_idx]
            # Alarm decision on the raw scale == calibrated prob >= threshold.
            decision_score = (raw >= sel.raw_threshold).astype(float)
            rep = fold_report(
                decision_score, y_te, ctx.ts.iloc[fold.test_idx], 0.5, ctx.events
            )
            np.savez_compressed(
                SCORES_DIR / f"{name}_s{seed}_f{fold.fold}.npz",
                oof_rows=sel.oof_rows,
                oof_raw=sel.oof_raw.astype(np.float32),
                test_rows=fold.test_idx,
                test_raw=raw.astype(np.float32),
                raw_threshold=sel.raw_threshold,
                platt=np.array(
                    [sel.calibrator.a, sel.calibrator.b]
                    if sel.calibrator is not None
                    else [np.nan, np.nan]
                ),
            )
            p_cal = npc.calibrated(sel, raw, name)
            rep["pr_auc"] = float(np.nan_to_num(_ap(y_te, raw)))
            rep.update(
                {
                    "seed": seed,
                    "fold": fold.fold,
                    "test_dataset": "2022" if is22[fold.test_idx].all() else "2020",
                    "params": sel.params,
                    "threshold_calibrated": sel.threshold_cal,
                    "threshold_raw": sel.raw_threshold,
                    "threshold_source": sel.source,
                    "ece_raw": _ece(y_te, npc.uncalibrated_prob(name, raw)),
                    "ece_calibrated": (
                        _ece(y_te, p_cal) if sel.calibrator is not None else None
                    ),
                    "search": sel.search,
                    "seconds": round(time.perf_counter() - t0, 1),
                }
            )
            runs.append(rep)
            log.info(
                "%s seed=%d fold=%d params=%s detected=%d/1 fa_ep=%d thr_cal=%.3g "
                "ece %s→%s (%.0fs)",
                name,
                seed,
                fold.fold,
                sel.params,
                rep["events"]["detected"],
                rep["events"]["false_alarm_episodes"],
                sel.threshold_cal,
                _fmt(rep["ece_raw"]),
                _fmt(rep["ece_calibrated"]),
                rep["seconds"],
            )
    return {"model": name, "runs": runs, "summary": summarise(runs)}


def _ap(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s)) if y.any() else float("nan")


def _fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    seeds = sorted({r["seed"] for r in runs})
    per_seed = []
    for s in seeds:
        rs = [r for r in runs if r["seed"] == s]
        tot: dict[str, Any] = {"samples": {}, "alarms": {}, "events": {}}
        for r in rs:
            tot["samples"] = _add(tot["samples"], r["samples"])
            tot["alarms"] = _add(tot["alarms"], r["alarms"])
            tot["events"] = _add(
                tot["events"],
                {
                    k: r["events"][k]
                    for k in ("failures", "detected", "missed", "false_alarm_episodes")
                },
            )
        tot["expected_cost_mean"] = float(
            np.mean([r["events"]["expected_cost"] for r in rs])
        )
        tot["metrics_alarms"] = _metrics(tot["alarms"])
        tot["metrics_samples"] = _metrics(tot["samples"])
        eces = [
            (r["ece_raw"], r["ece_calibrated"])
            for r in rs
            if r["ece_raw"] is not None and r["ece_calibrated"] is not None
        ]
        tot["ece_raw_mean"] = float(np.mean([a for a, _ in eces])) if eces else None
        tot["ece_calibrated_mean"] = (
            float(np.mean([b for _, b in eces])) if eces else None
        )
        per_seed.append({"seed": s, **tot})
    detected = [p["events"]["detected"] for p in per_seed]
    return {
        "per_seed": per_seed,
        "detected_mean": float(np.mean(detected)),
        "detected_min": int(np.min(detected)),
        "detected_max": int(np.max(detected)),
        "expected_cost_mean": float(
            np.mean([p["expected_cost_mean"] for p in per_seed])
        ),
        "false_alarm_episodes_mean": float(
            np.mean([p["events"]["false_alarm_episodes"] for p in per_seed])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    ctx, is22 = build_context()
    for name in args.models:
        t0 = time.perf_counter()
        result = run_model(ctx, is22, name, args.seeds)
        result.update(
            {
                "schema_version": "1.0",
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "protocol": "outer purged walk-forward (6 folds) × inner "
                "leave-one-failure-out; Platt calibration; RN-02 threshold",
                "seeds": args.seeds,
                "search_space": npc.SPACES[name],
                "dropped_sensors": DROPPED,
                "deep_config": dt.describe() if name in dt.ARCHS else None,
                "elapsed_s": round(time.perf_counter() - t0, 1),
            }
        )
        (OUT_DIR / f"{name}.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        log.info("wrote nested/%s.json (%.0fs)", name, result["elapsed_s"])


if __name__ == "__main__":
    main()
