"""
Sensitivity of the conclusions to the evaluation choices — persistence k,
early-warning horizon and cost ratio (RN-01/02/03).

Entry point:
    python -m src.evaluate_sensitivity

Uses the per-run OOF and test scores saved by ``evaluate_nested`` (no
retraining). For every setting the RN-02 threshold is RE-SELECTED on the
calibrated OOF scores of that run with the setting's own k / horizon / cost,
then applied to the out-of-time test rows — exactly what a deployment
configured with those values would do. Fallback runs (fewer than two training
failures) keep their a-priori threshold.

Reported per model and setting: failures detected (mean over seeds, of 6),
false-alarm episodes and expected cost; plus the Kendall τ between the model
ranking (by expected cost) under each setting and under the reference
setting (k = 6, horizon 6 h, C_FN/C_FP = 100) — do the conclusions survive?

Output: models/eval_sensitivity.json
"""

from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import src.nested_protocol as npc
from src.evaluate_nested import ALL_MODELS, SCORES_DIR, build_context
from src.evaluate_temporal import EVAL_KW, PERSISTENCE_K
from src.event_metrics import evaluate_events, persistent_alarms

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
OUT_JSON: Path = _ML_ROOT / "models" / "eval_sensitivity.json"

KS: tuple[int, ...] = (3, 6, 12)
HORIZONS_H: tuple[float, ...] = (2.0, 6.0, 12.0)
COST_RATIOS: tuple[float, ...] = (10.0, 100.0, 1000.0)
REFERENCE: tuple[int, float, float] = (PERSISTENCE_K, 6.0, 100.0)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("evaluate_sensitivity")


# Reduced threshold grid (dense where calibrated thresholds live, ~2 % prevalence).
SENS_GRID: list[float] = sorted(
    {
        0.002,
        0.005,
        *np.round(np.arange(0.01, 0.10, 0.01), 2).tolist(),
        *np.round(np.arange(0.10, 1.00, 0.05), 2).tolist(),
    }
)


def _kw(horizon_h: float) -> dict[str, float]:
    kw = dict(EVAL_KW)
    kw["early_horizon_s"] = horizon_h * 3600.0
    return kw


def _counts(
    ts: pd.Series,
    score: np.ndarray,
    thr: float,
    k: int,
    events: list[Any],
    kw: dict[str, float],
) -> tuple[int, int, int]:
    """(missed, detected, false-alarm episodes) — independent of the cost ratio."""
    rep = evaluate_events(ts, persistent_alarms(score, thr, k), events, **kw).to_dict()
    det = sum(1 for o in rep["outcomes"] if o["detected"])
    return len(rep["outcomes"]) - det, det, int(rep["n_false_alarm_episodes"])


def run_all_settings(
    ctx: npc.Context, run: dict[str, np.ndarray]
) -> dict[tuple[int, float, float], dict[str, float]]:
    """All (k, horizon, ratio) settings for one run. Missed / false-alarm counts
    are computed once per (k, horizon, threshold); the cost of every ratio is a
    linear combination of them, so the ratio never needs a re-evaluation."""
    test_rows, test_raw = run["test_rows"], run["test_raw"].astype(float)
    ts_te = ctx.ts.iloc[test_rows].reset_index(drop=True)
    calibrated = np.isfinite(run["platt"]).all() and run["oof_rows"].size > 0
    out: dict[tuple[int, float, float], dict[str, float]] = {}
    if calibrated:
        cal = npc.Platt(a=float(run["platt"][0]), b=float(run["platt"][1]))
        oof_rows = run["oof_rows"]
        ts_oof = ctx.ts.iloc[oof_rows].reset_index(drop=True)
        p_oof = cal.prob(run["oof_raw"].astype(float))
        p_te = cal.prob(test_raw)
        ev_oof = [
            e for e in ctx.events if e.end >= ts_oof.min() and e.start <= ts_oof.max()
        ]
    for k in KS:
        for h in HORIZONS_H:
            kw = _kw(h)
            if calibrated:
                table = [
                    (t, *_counts(ts_oof, p_oof, t, k, ev_oof, kw)) for t in SENS_GRID
                ]
            for r in COST_RATIOS:
                if calibrated:
                    # min cost; ties → higher threshold (fewer alarms), as RN-02
                    thr = min(table, key=lambda row: (r * row[1] + row[3], -row[0]))[0]
                    missed, det, fa = _counts(ts_te, p_te, thr, k, ctx.events, kw)
                else:
                    alarms_score = (test_raw >= float(run["raw_threshold"])).astype(
                        float
                    )
                    missed, det, fa = _counts(
                        ts_te, alarms_score, 0.5, k, ctx.events, kw
                    )
                out[(k, h, r)] = {
                    "detected": det,
                    "false_alarm_episodes": fa,
                    "expected_cost": r * missed + fa,
                }
    return out


WORKERS: int = 6
_CTX: list[npc.Context] = []


def _init_worker() -> None:
    _CTX.append(build_context()[0])


def _run_file(path: str) -> dict[tuple[int, float, float], dict[str, float]]:
    return run_all_settings(_CTX[0], dict(np.load(path)))


def main() -> None:
    ctx, _ = build_context()
    settings = list(itertools.product(KS, HORIZONS_H, COST_RATIOS))
    results: dict[str, Any] = {}
    for name in ALL_MODELS:
        files = sorted(SCORES_DIR.glob(f"{name}_s*_f*.npz"))
        if not files:
            log.warning("no saved scores for %s — run evaluate_nested first", name)
            continue
        runs = {f.stem: dict(np.load(f)) for f in files}
        seeds = sorted({int(s.split("_s")[1].split("_")[0]) for s in runs})
        with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker) as pool:
            per_run = dict(
                zip(
                    runs,
                    pool.map(_run_file, [str(SCORES_DIR / f"{k}.npz") for k in runs]),
                )
            )
        per_setting: dict[str, Any] = {}
        for k, h, r in settings:
            per_seed = []
            for seed in seeds:
                outs = [
                    v[(k, h, r)]
                    for key, v in per_run.items()
                    if key.startswith(f"{name}_s{seed}_")
                ]
                per_seed.append(
                    {
                        "detected": sum(o["detected"] for o in outs),
                        "false_alarm_episodes": sum(
                            o["false_alarm_episodes"] for o in outs
                        ),
                        "expected_cost_mean": float(
                            np.mean([o["expected_cost"] for o in outs])
                        ),
                    }
                )
            per_setting[f"k={k}|h={h:g}|ratio={r:g}"] = {
                "k": k,
                "horizon_h": h,
                "cost_ratio": r,
                "detected_mean": float(np.mean([p["detected"] for p in per_seed])),
                "false_alarm_episodes_mean": float(
                    np.mean([p["false_alarm_episodes"] for p in per_seed])
                ),
                "expected_cost_mean": float(
                    np.mean([p["expected_cost_mean"] for p in per_seed])
                ),
            }
        results[name] = per_setting
        log.info("%s: %d settings", name, len(per_setting))

    ref_key = "k={}|h={:g}|ratio={:g}".format(*REFERENCE)
    names = list(results)
    stability: dict[str, Any] = {}
    if len(names) >= 3:
        ref = [results[n][ref_key]["expected_cost_mean"] for n in names]
        for key in results[names[0]]:
            cur = [results[n][key]["expected_cost_mean"] for n in names]
            tau = kendalltau(ref, cur).statistic
            best = names[int(np.argmin(cur))]
            stability[key] = {
                "kendall_tau_vs_reference": float(tau),
                "best_model": best,
            }
    OUT_JSON.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "requirements": ["RN-01", "RN-02", "RN-03", "RNF-79"],
                "grid": {"k": KS, "horizon_h": HORIZONS_H, "cost_ratio": COST_RATIOS},
                "threshold_grid": SENS_GRID,
                "reference": ref_key,
                "models": results,
                "ranking_stability": stability,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    log.info("wrote %s", OUT_JSON)


if __name__ == "__main__":
    main()
