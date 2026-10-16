"""
Statistical promotion gate — RN-05 (and the new RF-10 selection criterion).

A candidate model replaces the baseline only if it is better with statistical
significance, not by a higher mean:

* metric: expected cost under RN-02 (lower is better) by default — the RF-10
  selection criterion moves from point-wise F1 to the operational cost of the
  alarm policy (missed failures vs false alarms);
* test: one-sided exact Wilcoxon signed-rank on paired blocks;
* blocks: temporal folds, seeds averaged (default). Pairing (fold, seed) runs
  is offered only as an explicit option because seeds of the same fold are
  NOT independent evidence (pseudo-replication).

The gate also reports the smallest p-value achievable with N blocks
(1 / 2^N for the one-sided exact test). With MetroPT-3's 3 walk-forward
folds that is 0.125 — so no model can ever be promoted at α = 0.05 on this
dataset alone, and the gate says so explicitly instead of hiding it.

Usage:
    python -m src.promotion_gate --candidate xgboost --baseline random_forest
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
EVAL_JSON: Path = _ML_ROOT / "models" / "eval_temporal.json"

# metric -> (getter path inside a run, lower_is_better)
METRICS: dict[str, tuple[tuple[str, str], bool]] = {
    "expected_cost_inner": (("event_inner", "expected_cost"), True),
    "pr_auc": (("point", "pr_auc"), False),
    "event_recall_inner": (("event_inner", "event_recall"), False),
    "false_alarms_per_day_inner": (("event_inner", "false_alarms_per_day"), True),
}


@dataclass
class GateDecision:
    candidate: str
    baseline: str
    metric: str
    protocol: str
    pairing: str
    n_blocks: int
    median_improvement: float
    p_value: float
    min_achievable_p: float
    alpha: float
    promote: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _blocks(
    runs: list[dict[str, Any]], model: str, metric: str, pairing: str
) -> dict[Any, float]:
    (section, key), _ = METRICS[metric]
    out: dict[Any, list[float]] = {}
    for r in runs:
        if r["model"] != model:
            continue
        block = r["fold"] if pairing == "fold" else (r["fold"], r["seed"])
        out.setdefault(block, []).append(float(r[section][key]))
    return {b: float(np.mean(v)) for b, v in out.items()}


def decide(
    eval_data: dict[str, Any],
    candidate: str,
    baseline: str,
    metric: str = "expected_cost_inner",
    protocol: str = "walk_forward",
    alpha: float = 0.05,
    pairing: str = "fold",
) -> GateDecision:
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}; choose from {sorted(METRICS)}.")
    if pairing not in {"fold", "fold_seed"}:
        raise ValueError("pairing must be 'fold' or 'fold_seed'.")
    runs = eval_data["protocols"][protocol]["runs"]
    cand = _blocks(runs, candidate, metric, pairing)
    base = _blocks(runs, baseline, metric, pairing)
    keys = sorted(set(cand) & set(base), key=str)
    if not keys:
        raise ValueError("No paired blocks between candidate and baseline.")

    _, lower_better = METRICS[metric]
    # improvement > 0 always means "candidate is better"
    diff = np.array(
        [(base[k] - cand[k]) if lower_better else (cand[k] - base[k]) for k in keys]
    )
    n = len(keys)
    min_p = 1.0 / (2**n)
    nonzero = diff[diff != 0]
    if nonzero.size == 0:
        p = 1.0
    else:
        p = float(wilcoxon(nonzero, alternative="greater", method="exact").pvalue)
    median = float(np.median(diff))

    if min_p >= alpha:
        promote, reason = False, (
            f"Underpowered: with {n} paired blocks the smallest achievable one-sided p is "
            f"{min_p:.3f} >= alpha = {alpha}. More independent evaluation blocks (failures) are required."
        )
    elif p < alpha and median > 0:
        promote, reason = (
            True,
            f"Candidate significantly better (p = {p:.4f} < {alpha}, median improvement {median:+.4f}).",
        )
    else:
        promote, reason = (
            False,
            f"Not significant (p = {p:.4f}) or no median improvement ({median:+.4f}).",
        )

    return GateDecision(
        candidate=candidate,
        baseline=baseline,
        metric=metric,
        protocol=protocol,
        pairing=pairing,
        n_blocks=n,
        median_improvement=median,
        p_value=p,
        min_achievable_p=min_p,
        alpha=alpha,
        promote=promote,
        reason=reason,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument(
        "--metric", default="expected_cost_inner", choices=sorted(METRICS)
    )
    parser.add_argument("--protocol", default="walk_forward")
    parser.add_argument("--pairing", default="fold", choices=["fold", "fold_seed"])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--eval", type=Path, default=EVAL_JSON)
    args = parser.parse_args()

    decision = decide(
        json.loads(args.eval.read_text(encoding="utf-8")),
        args.candidate,
        args.baseline,
        args.metric,
        args.protocol,
        args.alpha,
        args.pairing,
    )
    print(json.dumps(decision.to_dict(), indent=2))
    sys.exit(0 if decision.promote else 1)


if __name__ == "__main__":
    main()
