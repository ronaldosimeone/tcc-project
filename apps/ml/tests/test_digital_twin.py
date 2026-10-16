"""
Unit tests for src/digital_twin.py.

Guarantees verified
-------------------
1.  On healthy synthetic data that follows the fitted physics, residuals ≈ 0.
2.  An air leak (loading but the reservoir does not fill) drives
    ``fill_residual`` strongly negative and raises the duty cycle.
3.  Higher current than expected for the same pressure → positive
    ``current_residual``.
4.  Using the twin before ``fit`` or fitting without both regimes fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.digital_twin import TWIN_FEATURES, PhysicsTwin


def _cycle(
    n_cycles: int,
    fill: float = 0.1,
    leak: float = 0.0,
    current_offset: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Alternating loading (20 samples) / idle (60 samples) cycles."""
    rng = np.random.default_rng(seed)
    rows = []
    p = 8.0
    for _ in range(n_cycles):
        for _ in range(20):  # loading
            p += fill - leak
            rows.append(
                dict(
                    COMP=0.0,
                    Reservoirs=p,
                    TP2=p + 0.3,
                    Oil_temperature=60.0,
                    Motor_current=2.0
                    + 0.4 * (p + 0.3)
                    + current_offset
                    + rng.normal(0, 0.01),
                )
            )
        for _ in range(60):  # idle
            p -= 0.02 + leak
            rows.append(
                dict(
                    COMP=1.0,
                    Reservoirs=p,
                    TP2=0.0,
                    Oil_temperature=60.0,
                    Motor_current=1.0,
                )
            )
    return pd.DataFrame(rows)


def test_healthy_residuals_near_zero() -> None:
    healthy = _cycle(40)
    twin = PhysicsTwin().fit(healthy)
    res = twin.transform(_cycle(10, seed=1)).iloc[200:]
    assert list(res.columns) == list(TWIN_FEATURES)
    assert res["fill_residual"].abs().max() < 0.02
    assert res["current_residual"].abs().max() < 0.05


def test_air_leak_signature() -> None:
    twin = PhysicsTwin().fit(_cycle(40))
    healthy = twin.transform(_cycle(10, seed=2)).iloc[200:]
    leaking = twin.transform(_cycle(10, leak=0.09, seed=3)).iloc[200:]
    assert leaking["fill_residual"].mean() < healthy["fill_residual"].mean() - 0.05
    assert (
        leaking["idle_decay_residual"].mean()
        < healthy["idle_decay_residual"].mean() - 0.05
    )


def test_current_residual_positive_when_motor_works_harder() -> None:
    twin = PhysicsTwin().fit(_cycle(40))
    res = twin.transform(_cycle(10, current_offset=0.8, seed=4)).iloc[200:]
    assert res["current_residual"].mean() == pytest.approx(0.8, abs=0.05)


def test_requires_fit_and_both_regimes() -> None:
    with pytest.raises(RuntimeError):
        PhysicsTwin().transform(_cycle(2))
    only_idle = _cycle(5)
    only_idle = only_idle[only_idle.COMP == 1]
    with pytest.raises(ValueError):
        PhysicsTwin().fit(only_idle)
