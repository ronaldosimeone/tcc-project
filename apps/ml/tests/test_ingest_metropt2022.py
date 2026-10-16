"""RF-33 — harmonização do MetroPT 2022 com o MetroPT-3."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ingest_metropt import EXPECTED_COLUMNS
from src.ingest_metropt2022 import (
    DROPPED_COLUMNS,
    FAILURE_INTERVALS_2022,
    label,
    point_sample,
)


def _one_hz(start: str, seconds: int) -> pd.DataFrame:
    ts = pd.date_range(start, periods=seconds, freq="1s")
    return pd.DataFrame({"timestamp": ts, "TP2": np.arange(seconds, dtype=float)})


def test_point_sample_keeps_first_reading_of_each_bin() -> None:
    out = point_sample(_one_hz("2022-01-01 06:00:00", 35))
    assert out["timestamp"].diff().dropna().dt.total_seconds().eq(10).all()
    # First reading, not the mean: 0, 10, 20, 30.
    assert out["TP2"].tolist() == [0.0, 10.0, 20.0, 30.0]


def test_point_sample_drops_gaps_instead_of_inventing_rows() -> None:
    a = _one_hz("2022-01-01 06:00:00", 20)
    b = _one_hz("2022-01-01 08:00:00", 20)
    out = point_sample(pd.concat([a, b], ignore_index=True))
    assert len(out) == 4
    assert out["TP2"].notna().all()


def test_label_marks_inclusive_intervals_only() -> None:
    ts = pd.to_datetime(
        [
            "2022-03-23 14:53:50",
            "2022-03-23 14:54:00",
            "2022-03-23 15:24:00",
            "2022-03-23 15:24:10",
        ]
    )
    y = label(pd.DataFrame({"timestamp": ts}), FAILURE_INTERVALS_2022)
    assert y.tolist() == [0, 1, 1, 0]


def test_table2_intervals_are_ordered_and_within_the_published_period() -> None:
    starts = [pd.Timestamp(s) for s, _ in FAILURE_INTERVALS_2022]
    ends = [pd.Timestamp(e) for _, e in FAILURE_INTERVALS_2022]
    assert starts == sorted(starts)
    assert all(s < e for s, e in zip(starts, ends))
    assert starts[0] >= pd.Timestamp("2022-01-01")
    assert ends[-1] <= pd.Timestamp("2022-07-01")


def test_gps_is_dropped_and_metropt3_schema_is_kept() -> None:
    assert not set(DROPPED_COLUMNS) & set(EXPECTED_COLUMNS)
    assert all(c.startswith("gps") for c in DROPPED_COLUMNS)
