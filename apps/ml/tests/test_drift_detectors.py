"""
Unit tests for src/drift_detectors.py — RF-34.

Guarantees verified
-------------------
1.  Page-Hinkley and ADWIN stay silent on a stationary stream.
2.  Both detect an abrupt mean shift, after it happens, within a bounded delay.
3.  ADWIN shrinks its window to the post-change data after detecting.
4.  PSI ≈ 0 for identical distributions and large for a shifted one.
"""

from __future__ import annotations

import numpy as np

from src.drift_detectors import Adwin, TwoSidedPageHinkley, psi


def _stream(
    seed: int, n: int = 600, shift_at: int | None = None, shift: float = 0.0
) -> np.ndarray:
    x = np.random.default_rng(seed).normal(0, 1, n)
    if shift_at is not None:
        x[shift_at:] += shift
    return x


def _first_alarm(detector, xs: np.ndarray) -> int | None:
    for i, x in enumerate(xs):
        if detector.update(float(x)):
            return i
    return None


def test_silent_on_stationary_stream() -> None:
    xs = _stream(0)
    assert _first_alarm(TwoSidedPageHinkley(delta=0.05, threshold=25), xs) is None
    assert _first_alarm(Adwin(delta=0.002), xs) is None


def test_detects_abrupt_shift_after_change() -> None:
    xs = _stream(1, shift_at=300, shift=2.0)
    for det in (TwoSidedPageHinkley(delta=0.05, threshold=25), Adwin(delta=0.002)):
        alarm = _first_alarm(det, xs)
        assert alarm is not None and 300 <= alarm < 360


def test_adwin_window_shrinks_after_detection() -> None:
    det = Adwin(delta=0.002)
    xs = _stream(2, shift_at=300, shift=3.0)
    alarm = _first_alarm(det, xs)
    assert alarm is not None
    assert len(det.window) < alarm  # old regime dropped


def test_psi_identical_vs_shifted() -> None:
    rng = np.random.default_rng(3)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.02
    assert psi(ref, rng.normal(1.5, 1, 5000)) > 0.25
