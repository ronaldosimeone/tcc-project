"""
Unit tests for src/promotion_gate.py — RN-05.

Guarantees verified
-------------------
1.  With 3 folds the gate refuses as underpowered (min one-sided p = 0.125).
2.  With enough blocks and a consistent improvement it promotes.
3.  A higher MEAN driven by one block is not enough (no significance).
4.  Direction is respected for lower-is-better metrics (cost).
5.  Seeds of the same fold are averaged under the default pairing.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.promotion_gate import decide


def _eval(costs: dict[str, list[list[float]]]) -> dict[str, Any]:
    """costs[model][fold][seed] -> expected cost."""
    runs = [
        {
            "model": m,
            "fold": f + 1,
            "seed": s,
            "event_inner": {
                "expected_cost": c,
                "event_recall": 1.0,
                "false_alarms_per_day": 0.0,
            },
            "point": {"pr_auc": 0.5},
        }
        for m, folds in costs.items()
        for f, seeds in enumerate(folds)
        for s, c in enumerate(seeds)
    ]
    return {"protocols": {"walk_forward": {"runs": runs}}}


def test_three_folds_are_underpowered() -> None:
    data = _eval({"cand": [[1.0], [1.0], [1.0]], "base": [[100.0], [100.0], [100.0]]})
    d = decide(data, "cand", "base")
    assert d.n_blocks == 3 and d.min_achievable_p == pytest.approx(0.125)
    assert not d.promote and "Underpowered" in d.reason


def test_consistent_improvement_over_many_blocks_promotes() -> None:
    base = [[10.0 + i] for i in range(8)]
    cand = [[5.0 + i] for i in range(8)]
    d = decide(_eval({"cand": cand, "base": base}), "cand", "base")
    assert d.promote and d.p_value < 0.05 and d.median_improvement == pytest.approx(5.0)


def test_mean_driven_by_one_block_is_not_enough() -> None:
    base = [[10.0]] * 8
    cand = [[11.0]] * 7 + [[-200.0]]  # far lower mean cost, worse in 7/8 folds
    d = decide(_eval({"cand": cand, "base": base}), "cand", "base")
    assert not d.promote


def test_higher_is_better_metric_direction() -> None:
    data = _eval({"cand": [[1.0]] * 8, "base": [[1.0]] * 8})
    for r in data["protocols"]["walk_forward"]["runs"]:
        r["point"]["pr_auc"] = 0.9 if r["model"] == "cand" else 0.5
    d = decide(data, "cand", "base", metric="pr_auc")
    assert d.median_improvement == pytest.approx(0.4) and d.promote


def test_seeds_are_averaged_per_fold() -> None:
    data = _eval({"cand": [[0.0, 10.0]] * 3, "base": [[5.0, 5.0]] * 3})
    d = decide(data, "cand", "base")
    assert d.n_blocks == 3 and d.median_improvement == pytest.approx(0.0)
    d2 = decide(data, "cand", "base", pairing="fold_seed")
    assert d2.n_blocks == 6
