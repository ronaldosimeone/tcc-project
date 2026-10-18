"""ACI recovers coverage under shift where static split conformal does not."""

from __future__ import annotations

import numpy as np

from src.evaluate_conformal_aci import _report, aci_mondrian, static_mondrian


def _shifted_stream(n: int = 6000, seed: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    y_cal = (rng.random(4000) < 0.2).astype(int)
    p_cal = np.clip(0.7 * y_cal + 0.15 + rng.normal(0, 0.1, 4000), 0, 1)
    y = (rng.random(n) < 0.2).astype(int)
    # Shift: failures look much less like failures in the test period.
    p = np.clip(0.35 * y + 0.15 + rng.normal(0, 0.1, n), 0, 1)
    return p_cal, y_cal, p, y


def test_static_conformal_is_valid_without_shift() -> None:
    p_cal, y_cal, _, _ = _shifted_stream()
    rep = _report(static_mondrian(p_cal, y_cal, p_cal, 0.1), y_cal)
    assert rep["coverage_class_1"] >= 0.88


def test_aci_restores_failure_class_coverage_under_shift() -> None:
    p_cal, y_cal, p, y = _shifted_stream()
    static = _report(static_mondrian(p_cal, y_cal, p, 0.1), y)
    aci = _report(aci_mondrian(p_cal, y_cal, p, y, 0.1, 0.01), y)
    assert static["coverage_class_1"] < 0.8
    assert abs(aci["coverage_class_1"] - 0.9) < 0.05
