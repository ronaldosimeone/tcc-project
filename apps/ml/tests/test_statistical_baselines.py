"""Unit tests for the CUSUM/EWMA baselines (statistical_baselines)."""

from __future__ import annotations

import numpy as np
import pandas as pd

import src.statistical_baselines as sb


def _frame(x: np.ndarray, gap_at: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
    t = np.arange(len(x)) * 10.0
    if gap_at is not None:
        t[gap_at:] += 3600.0
    ts = pd.Series(pd.Timestamp("2020-01-01") + pd.to_timedelta(t, unit="s"))
    return pd.DataFrame({"Motor_current": x}), ts


def test_segment_mean_is_causal_and_resets_at_gaps() -> None:
    x = np.r_[np.zeros(100), np.full(100, 10.0)]
    X, ts = _frame(x, gap_at=100)
    s = sb._Series(X, ts)
    m = s.smoothed("Motor_current")
    assert np.allclose(m[:100], 0.0)  # future never leaks backwards
    assert np.allclose(m[100:], 10.0)  # new segment: no mix with the past


def test_cusum_grows_on_shift_and_stays_zero_in_control() -> None:
    seg = np.zeros(200, dtype=np.int64)
    z = np.r_[np.zeros(100), np.full(100, 2.0)]
    out = sb._cusum(z, seg, 0.5)
    assert np.all(out[:100] == 0.0)
    assert out[-1] == np.float64(100 * 1.5)


def test_cusum_resets_at_segment_start() -> None:
    z = np.full(20, 2.0)
    seg = np.r_[np.zeros(10), np.ones(10)].astype(np.int64)
    out = sb._cusum(z, seg, 0.5)
    assert out[10] == 1.5


def test_ewma_converges_to_shift() -> None:
    z = np.full(500, 3.0)
    out = sb._ewma(z, np.zeros(500, dtype=np.int64), 0.1)
    assert abs(out[-1] - 3.0) < 1e-6


def test_rolling_reference_excludes_recent_hour() -> None:
    t = np.arange(0, 3 * 3600, 10, dtype=np.float64)
    x = (t >= 2 * 3600).astype(np.float64)  # step in the last hour
    mu, _, cnt = sb._rolling_ref(t, x, 7 * 86400.0, 3600.0)
    last = len(t) - 1
    # the reference only sees data older than 1 h, so the step is not in it
    assert mu[last] < 0.01
    assert cnt[0] == 0


def test_detector_scores_are_monotone_in_shift() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 0.1, 4000)
    x[3000:] += 1.0
    X, ts = _frame(x)
    y = pd.Series(np.zeros(len(x), dtype=int))
    rows = np.arange(len(x))
    params = {"sensors": "Motor_current", "k": 0.5, "reference": "static"}
    det = sb.fit("cusum", params, X, y, ts, rows[:3000])
    raw = sb.raw_score(det, X, ts, rows)
    assert raw[3500:].mean() > raw[:3000].mean() + 1.0
