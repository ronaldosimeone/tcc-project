"""
Feature engineering respects time gaps and never looks ahead.

1. A gap > 60 s starts a new segment: windows, deltas and lags after the gap
   use only post-gap readings.
2. Causality: changing a FUTURE reading never changes a past feature row.
3. Without a timestamp column (live buffer) the frame is one segment — same
   features as before the change.
4. Imputation is causal: leading NaN becomes 0, never the next reading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing import MetroPTPreprocessor

SENSORS = ["TP2", "TP3", "H1", "DV_pressure", "Oil_temperature", "Motor_current"]


def _frame(n: int = 40, gap_at: int | None = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2022-01-01", periods=n, freq="10s")
    if gap_at is not None:
        ts = ts.where(np.arange(n) < gap_at, ts + pd.Timedelta(hours=2))
    df = pd.DataFrame(rng.normal(5, 1, (n, len(SENSORS))), columns=SENSORS)
    df.insert(0, "timestamp", ts)
    return df


def _pre() -> MetroPTPreprocessor:
    return MetroPTPreprocessor(sensor_cols=SENSORS)


def test_windows_and_lags_restart_after_a_gap() -> None:
    df = _frame()
    out = _pre().transform(df)
    after = df["TP2"].iloc[20:]
    assert out["TP2_ma_15"].iloc[20] == after.iloc[0]  # nothing from before
    assert np.isclose(out["TP2_ma_15"].iloc[25], after.iloc[:6].mean())
    assert out["TP2_lag_15"].iloc[22] == after.iloc[0]  # segment's first reading
    assert out["TP2_delta"].iloc[20] == 0.0
    assert out["TP2_std_5"].iloc[20] == 0.0
    assert "timestamp" not in out.columns


def test_changing_the_future_never_changes_the_past() -> None:
    df = _frame(gap_at=None)
    base = _pre().transform(df)
    future = df.copy()
    future.loc[30:, SENSORS] += 100.0
    moved = _pre().transform(future)
    pd.testing.assert_frame_equal(base.iloc[:30], moved.iloc[:30])


def test_without_timestamp_the_frame_is_one_segment() -> None:
    df = _frame()
    no_ts = _pre().transform(df.drop(columns="timestamp"))
    no_gap = _pre().transform(_frame(gap_at=None))
    # Same readings, no time column → identical to a gap-free series.
    pd.testing.assert_frame_equal(no_ts, no_gap)


def test_imputation_is_causal() -> None:
    df = pd.DataFrame({"TP2": [np.nan, np.nan, 7.0, np.nan], "TP3": [1.0] * 4})
    out = MetroPTPreprocessor(sensor_cols=["TP2", "TP3"])._impute_nulls(df)
    assert out["TP2"].tolist() == [0.0, 0.0, 7.0, 7.0]
