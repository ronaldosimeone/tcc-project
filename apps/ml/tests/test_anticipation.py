"""
Anticipation target (src/evaluate_anticipation.py).

1. Positive rows are exactly those in [start − H, start).
2. Rows of the failure in progress and of the hour after it are excluded.
3. Event outcome: only a persistent alarm BEFORE the start counts, and the
   lead is measured to the failure start.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from src.evaluate_anticipation import anticipation_task, outcomes
from src.temporal_validation import FailureEvent


def _ctx() -> Any:
    ts = pd.Series(pd.date_range("2022-01-01 00:00", periods=6 * 60 * 6, freq="10s"))
    ev = FailureEvent(
        pd.Timestamp("2022-01-01 03:00"), pd.Timestamp("2022-01-01 03:30")
    )
    X = pd.DataFrame({"a": np.arange(len(ts), dtype=float)})
    y = pd.Series(((ts >= ev.start) & (ts <= ev.end)).astype(int))
    import src.nested_protocol as npc

    return npc.Context(X=X, y=y, ts=ts, events=[ev], data=SimpleNamespace())


def test_label_and_exclusions() -> None:
    ctx = _ctx()
    sub, keep, windows = anticipation_task(ctx, pd.Timedelta(minutes=30))
    ts = sub.ts
    pos = ts[sub.y.to_numpy() == 1]
    assert pos.min() == pd.Timestamp("2022-01-01 02:30")
    assert pos.max() == pd.Timestamp("2022-01-01 02:59:50")
    # failure (03:00–03:30) and the following hour (until 04:30) are gone
    gone = (ts >= pd.Timestamp("2022-01-01 03:00")) & (
        ts <= pd.Timestamp("2022-01-01 04:30")
    )
    assert not gone.any()
    assert len(keep) == len(sub.ts)
    assert windows[0].start == pd.Timestamp("2022-01-01 02:30")


def test_only_alarms_before_start_count() -> None:
    ctx = _ctx()
    sub, _, windows = anticipation_task(ctx, pd.Timedelta(minutes=30))
    ts = sub.ts
    alarms = (
        (ts >= pd.Timestamp("2022-01-01 02:40"))
        & (ts < pd.Timestamp("2022-01-01 02:50"))
    ).to_numpy()
    out = outcomes(ts, alarms.astype(float), windows)
    assert out["anticipated"] == 1
    assert out["per_failure"][0]["lead_before_start_min"] == 20.0
    late = (
        (ts > pd.Timestamp("2022-01-01 05:00"))
        & (ts < pd.Timestamp("2022-01-01 05:10"))
    ).to_numpy()
    out_late = outcomes(ts, late.astype(float), windows)
    assert out_late["anticipated"] == 0 and out_late["false_alarm_episodes"] == 1


def test_chance_control_and_onset() -> None:
    ctx = _ctx()
    sub, _, windows = anticipation_task(ctx, pd.Timedelta(minutes=30))
    ts = sub.ts
    # ON from before the window opened (02:30) until 02:45: anticipated, but
    # with no OFF->ON onset inside the window.
    early = (ts < pd.Timestamp("2022-01-01 02:45")).to_numpy()
    # A second alarm in normal time (05:00–05:10), the only region outside the
    # 6 h-before / 1 h-after guard: it must hit a reference window.
    later = (
        (ts >= pd.Timestamp("2022-01-01 05:00"))
        & (ts < pd.Timestamp("2022-01-01 05:10"))
    ).to_numpy()
    out = outcomes(ts, (early | later).astype(float), windows)
    assert out["anticipated"] == 1 and out["anticipated_onset_in_window"] == 0
    assert out["reference_windows"] >= 1 and out["reference_hits"] == 1
    assert out["alarm_time_share"] is not None and out["alarm_time_share"] > 0
