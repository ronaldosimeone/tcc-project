"""
False alarms are counted on alarm TIME outside the failure windows.

Closes a loophole found in round 6: an always-on alarm was one episode that
touched the failure — "detected, zero false alarms", cost 0 — and the
cost-driven threshold search of the autoencoder converged to it.

1. An alarm only inside the failure window: detected, no false alarm.
2. An always-on alarm over days: detected, but one false alarm per started
   hour outside the window — expensive, as it should be.
3. A 2.5 h false episode counts 3; a single false sample counts 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.event_metrics import evaluate_events
from src.temporal_validation import FailureEvent

TS = pd.Series(pd.date_range("2022-01-01", periods=3 * 24 * 360, freq="10s"))
EV = FailureEvent(pd.Timestamp("2022-01-02 12:00"), pd.Timestamp("2022-01-02 13:00"))
KW = dict(early_horizon_s=6 * 3600.0, grace_s=3600.0, min_lead_s=0.0)


def _alarms(*spans: tuple[str, str]) -> np.ndarray:
    a = np.zeros(len(TS), bool)
    for lo, hi in spans:
        a |= ((TS >= pd.Timestamp(lo)) & (TS <= pd.Timestamp(hi))).to_numpy()
    return a.astype(float)


def test_alarm_inside_the_window_is_not_a_false_alarm() -> None:
    r = evaluate_events(
        TS, _alarms(("2022-01-02 12:10", "2022-01-02 12:40")), [EV], **KW
    )
    assert r.n_detected == 1 and r.n_false_alarm_episodes == 0
    assert r.false_alarm_time_s == 0.0


def test_always_on_alarm_is_expensive() -> None:
    r = evaluate_events(TS, np.ones(len(TS)), [EV], **KW)
    assert r.n_detected == 1
    # window = [06:00 day 2, 14:00 day 2]; outside ≈ 30 h before + 34 h after
    assert r.n_false_alarm_episodes >= 60
    assert r.expected_cost >= 60


def test_false_episode_counts_per_started_hour() -> None:
    r = evaluate_events(
        TS, _alarms(("2022-01-01 01:00", "2022-01-01 03:30")), [EV], **KW
    )
    assert r.n_false_alarm_episodes == 3
    one = evaluate_events(
        TS, _alarms(("2022-01-01 01:00", "2022-01-01 01:00")), [EV], **KW
    )
    assert one.n_false_alarm_episodes == 1
