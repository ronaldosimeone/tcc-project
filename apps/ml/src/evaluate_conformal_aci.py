"""
Conformal prediction under temporal shift — static Mondrian split conformal
vs class-conditional Adaptive Conformal Inference (ACI; Gibbs & Candès,
NeurIPS 2021) — RF-31.

Entry point:
    python -m src.evaluate_conformal_aci [--alpha 0.1] [--gamma 0.005]

Uses the per-run scores saved by ``evaluate_nested``: the calibration set is
the run's calibrated OUT-OF-FOLD probabilities (training data only); the test
stream is the out-of-time test fold, in time order.

* Static Mondrian: per-class quantile of the non-conformity 1 − p(true class)
  on the calibration set, fixed for the whole test stream (assumes
  exchangeability — the assumption the first round showed to fail:
  failure-class coverage 0.59–0.68 vs the 0.90 target).
* ACI (Mondrian): the miscoverage level of each class is updated online,
  α_t+1 = α_t + γ (α − err_t), using the label of each row once known. This
  restores long-run coverage under distribution shift without exchangeability.
  Assumption stated for the thesis: labels become available right after each
  row (in operation they arrive with the work-order verdict; a delay only
  slows the adaptation).

Reported per model (mean over seeds × folds): coverage overall and per class,
mean set size, share of empty sets. Output: models/eval_conformal_aci.json
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

import src.nested_protocol as npc
from src.evaluate_nested import ALL_MODELS, SCORES_DIR, build_context

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
OUT_JSON: Path = _ML_ROOT / "models" / "eval_conformal_aci.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("evaluate_conformal_aci")


def _quantile(scores: np.ndarray, level: float) -> float:
    """Finite-sample conformal quantile of SORTED ``scores``; level ≥ 1 → +inf.

    The ⌈(n+1)·level⌉-th smallest score (1-based), read by index in O(1) so
    the online ACI loop stays linear in the stream length.
    """
    n = scores.size
    if n == 0 or level >= 1.0:
        return np.inf
    if level <= 0.0:
        return -np.inf
    k = int(np.ceil((n + 1) * level))
    return float(scores[min(k, n) - 1]) if k <= n else np.inf


def static_mondrian(
    p_cal: np.ndarray, y_cal: np.ndarray, p_test: np.ndarray, alpha: float
) -> np.ndarray:
    """Boolean sets (n, 2): column c = class c in the set."""
    probs_cal = np.stack([1 - p_cal, p_cal], axis=1)
    probs = np.stack([1 - p_test, p_test], axis=1)
    sets = np.zeros_like(probs, dtype=bool)
    for c in (0, 1):
        s = np.sort(1 - probs_cal[y_cal == c, c])
        sets[:, c] = (1 - probs[:, c]) <= _quantile(s, 1 - alpha)
    return sets


def aci_mondrian(
    p_cal: np.ndarray,
    y_cal: np.ndarray,
    p_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    gamma: float,
) -> np.ndarray:
    probs_cal = np.stack([1 - p_cal, p_cal], axis=1)
    probs = np.stack([1 - p_test, p_test], axis=1)
    cal_scores = [np.sort(1 - probs_cal[y_cal == c, c]) for c in (0, 1)]
    alpha_t = [alpha, alpha]
    sets = np.zeros_like(probs, dtype=bool)
    for t in range(len(p_test)):
        for c in (0, 1):
            sets[t, c] = (1 - probs[t, c]) <= _quantile(cal_scores[c], 1 - alpha_t[c])
        c = int(y_test[t])  # label revealed after the prediction
        err = 0.0 if sets[t, c] else 1.0
        alpha_t[c] = alpha_t[c] + gamma * (alpha - err)
    return sets


def _report(sets: np.ndarray, y: np.ndarray) -> dict[str, float]:
    covered = sets[np.arange(len(y)), y]
    out = {
        "coverage": float(covered.mean()),
        "set_size": float(sets.sum(axis=1).mean()),
        "empty_sets": float((sets.sum(axis=1) == 0).mean()),
    }
    for c in (0, 1):
        m = y == c
        out[f"coverage_class_{c}"] = (
            float(covered[m].mean()) if m.any() else float("nan")
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.005)
    args = parser.parse_args()
    ctx, _ = build_context()
    y_all = ctx.y.to_numpy().astype(int)
    results: dict[str, Any] = {}
    for name in ALL_MODELS:
        files = sorted(SCORES_DIR.glob(f"{name}_s*_f*.npz"))
        rows: dict[str, list[dict[str, float]]] = {"static": [], "aci": []}
        for f in files:
            run = dict(np.load(f))
            if not (np.isfinite(run["platt"]).all() and run["oof_rows"].size):
                continue  # fallback run: no calibration set
            cal = npc.Platt(a=float(run["platt"][0]), b=float(run["platt"][1]))
            p_cal = cal.prob(run["oof_raw"].astype(float))
            y_cal = y_all[run["oof_rows"]]
            p_test = cal.prob(run["test_raw"].astype(float))
            y_test = y_all[run["test_rows"]]
            rows["static"].append(
                _report(static_mondrian(p_cal, y_cal, p_test, args.alpha), y_test)
            )
            rows["aci"].append(
                _report(
                    aci_mondrian(p_cal, y_cal, p_test, y_test, args.alpha, args.gamma),
                    y_test,
                )
            )
        if not rows["static"]:
            continue
        results[name] = {
            method: {k: float(np.nanmean([r[k] for r in rs])) for k in rs[0]}
            | {"n_runs": len(rs)}
            for method, rs in rows.items()
        }
        log.info(
            "%s: failure-class coverage static %.2f → ACI %.2f (target %.2f)",
            name,
            results[name]["static"]["coverage_class_1"],
            results[name]["aci"]["coverage_class_1"],
            1 - args.alpha,
        )
    OUT_JSON.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "requirements": ["RF-31"],
                "alpha": args.alpha,
                "gamma": args.gamma,
                "calibration_set": "calibrated out-of-fold training scores",
                "label_delay_assumption": "label known right after each row",
                "models": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("wrote %s", OUT_JSON)


if __name__ == "__main__":
    main()
