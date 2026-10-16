"""
Unit tests for src/temporal_validation.py, src/event_metrics.py and
src/stat_tests.py — RF-28 / RF-29 / RNF-79.

Guarantees verified
-------------------
1.  Walk-forward: every training timestamp is earlier than test_start - purge.
2.  Walk-forward: train and test never share an index; fold count = events - 1.
3.  Blocked K-fold: no training row inside [test_start - purge, test_end + embargo).
4.  Each test segment contains exactly its own failure event.
5.  persistent_alarms needs k consecutive samples (RN-03).
6.  Lead time sign: alarm before failure start is positive (RN-01).
7.  Alarm far from any failure counts as a false alarm and adds C_FP (RN-02).
8.  Missed event adds C_FN; range recall = 0 for a missed event.
9.  Friedman/Nemenyi ranks and CD follow Demšar (2006).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.event_metrics import (
    alarm_episodes,
    cost_optimal_threshold,
    evaluate_events,
    persistent_alarms,
)
from src.stat_tests import bootstrap_ci, friedman_nemenyi, nemenyi_cd, wilcoxon_paired
from src.temporal_validation import (
    FailureEvent,
    PurgedEventBlockKFold,
    PurgedWalkForwardSplit,
    events_from_intervals,
)

_INTERVALS: list[tuple[str, str]] = [
    ("2020-01-03 00:00", "2020-01-03 06:00"),
    ("2020-01-06 00:00", "2020-01-06 06:00"),
    ("2020-01-09 00:00", "2020-01-09 06:00"),
]


@pytest.fixture()
def timeline() -> pd.Series:
    return pd.Series(pd.date_range("2020-01-01", "2020-01-11", freq="10s"))


@pytest.fixture()
def events() -> list[FailureEvent]:
    return events_from_intervals(_INTERVALS)


# ---------------------------------------------------------------------------
# Splitters
# ---------------------------------------------------------------------------


def test_walk_forward_never_trains_on_future(
    timeline: pd.Series, events: list[FailureEvent]
) -> None:
    splitter = PurgedWalkForwardSplit(events, purge_seconds=300, embargo_seconds=3600)
    folds = list(splitter.split(timeline))
    assert len(folds) == splitter.get_n_splits() == 2
    for f in folds:
        assert np.intersect1d(f.train_idx, f.test_idx).size == 0
        assert timeline.iloc[f.train_idx].max() < f.test_start - pd.Timedelta(
            seconds=300
        )
        assert timeline.iloc[f.test_idx].min() >= f.test_start


def test_blocked_kfold_purges_and_embargoes(
    timeline: pd.Series, events: list[FailureEvent]
) -> None:
    splitter = PurgedEventBlockKFold(events, purge_seconds=300, embargo_seconds=3600)
    folds = list(splitter.split(timeline))
    assert len(folds) == 3
    for f in folds:
        train_ts = timeline.iloc[f.train_idx]
        forbidden = (train_ts >= f.test_start - pd.Timedelta(seconds=300)) & (
            train_ts < f.test_end + pd.Timedelta(seconds=3600)
        )
        assert not forbidden.any()


def test_each_test_segment_contains_its_event(
    timeline: pd.Series, events: list[FailureEvent]
) -> None:
    for f in PurgedEventBlockKFold(events).split(timeline):
        test_ts = timeline.iloc[f.test_idx]
        assert test_ts.min() <= f.test_event.start and test_ts.max() >= f.test_event.end
        others = [ev for ev in events if ev != f.test_event]
        for ev in others:
            assert not ((test_ts >= ev.start) & (test_ts <= ev.end)).any()


def test_negative_purge_rejected(events: list[FailureEvent]) -> None:
    with pytest.raises(ValueError):
        PurgedWalkForwardSplit(events, purge_seconds=-1)


# ---------------------------------------------------------------------------
# Event metrics
# ---------------------------------------------------------------------------


def test_persistent_alarms_requires_k_consecutive() -> None:
    scores = np.array([0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9])
    assert persistent_alarms(scores, 0.5, 3).tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]
    assert persistent_alarms(scores, 0.5, 1).tolist() == (scores >= 0.5).tolist()
    with pytest.raises(ValueError):
        persistent_alarms(scores, 0.5, 0)


def test_alarm_episodes_merge_by_gap() -> None:
    ts = pd.Series(pd.date_range("2020-01-01", periods=10, freq="10s"))
    alarms = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1], dtype=bool)
    assert len(alarm_episodes(ts, alarms, max_gap_seconds=30)) == 2
    assert len(alarm_episodes(ts, alarms, max_gap_seconds=100)) == 1


def test_lead_time_positive_when_alarm_precedes_failure(
    timeline: pd.Series, events: list[FailureEvent]
) -> None:
    ev = events[0]
    alarms = (
        (timeline >= ev.start - pd.Timedelta(hours=3)) & (timeline <= ev.end)
    ).to_numpy()
    rep = evaluate_events(
        timeline, alarms, [ev], early_horizon_s=6 * 3600, min_lead_s=2 * 3600
    )
    assert rep.n_detected == 1 and rep.n_false_alarm_episodes == 0
    assert rep.outcomes[0].lead_time_s == pytest.approx(3 * 3600)
    assert rep.outcomes[0].meets_min_lead
    assert rep.range_recall == pytest.approx(1.0)


def test_detection_delay_is_negative(
    timeline: pd.Series, events: list[FailureEvent]
) -> None:
    ev = events[0]
    alarms = (
        (timeline >= ev.start + pd.Timedelta(hours=1)) & (timeline <= ev.end)
    ).to_numpy()
    rep = evaluate_events(timeline, alarms, [ev])
    assert rep.outcomes[0].lead_time_s == pytest.approx(-3600)
    assert not rep.outcomes[0].meets_min_lead


def test_false_alarm_and_missed_event_costs(
    timeline: pd.Series, events: list[FailureEvent]
) -> None:
    far = pd.Timestamp("2020-01-01 12:00")
    alarms = (
        (timeline >= far) & (timeline <= far + pd.Timedelta(minutes=5))
    ).to_numpy()
    rep = evaluate_events(timeline, alarms, events, cost_fn=100, cost_fp=1)
    assert rep.n_detected == 0 and rep.n_false_alarm_episodes == 1
    assert rep.expected_cost == pytest.approx(3 * 100 + 1)
    assert rep.range_recall == 0.0 and rep.event_f1 == 0.0
    assert rep.false_alarms_per_day == pytest.approx(1 / rep.observed_days)


def test_cost_optimal_threshold_prefers_detection(
    timeline: pd.Series, events: list[FailureEvent]
) -> None:
    in_event = np.zeros(len(timeline), dtype=bool)
    for ev in events:
        in_event |= ((timeline >= ev.start) & (timeline <= ev.end)).to_numpy()
    scores = np.where(in_event, 0.7, 0.2)
    thr, cost = cost_optimal_threshold(
        timeline, scores, events, k=3, candidates=[0.1, 0.5, 0.8]
    )
    assert thr == 0.5 and cost == 0.0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_bootstrap_ci_contains_mean() -> None:
    ci = bootstrap_ci(np.array([0.6, 0.7, 0.8, 0.75, 0.65]))
    assert ci.low <= ci.mean <= ci.high and ci.n == 5


def test_wilcoxon_identical_samples() -> None:
    assert wilcoxon_paired(np.ones(5), np.ones(5)) == (0.0, 1.0)


def test_friedman_ranks_and_cd() -> None:
    scores = {
        "best": np.array([0.9, 0.8, 0.85, 0.95, 0.9]),
        "mid": np.array([0.7, 0.6, 0.75, 0.8, 0.7]),
        "worst": np.array([0.5, 0.4, 0.55, 0.6, 0.5]),
    }
    res = friedman_nemenyi(scores)
    assert res.avg_ranks == {"best": 1.0, "mid": 2.0, "worst": 3.0}
    assert res.critical_difference == pytest.approx(nemenyi_cd(3, 5))
    assert res.critical_difference == pytest.approx(2.343 * np.sqrt(3 * 4 / 30))
    assert res.significant
