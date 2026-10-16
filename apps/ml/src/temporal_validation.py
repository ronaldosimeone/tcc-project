"""
Temporal validation with purging and embargo — RF-28.

Why this module exists
----------------------
The original evaluation (``train_test_split(..., stratify=y)``) shuffles
10-second samples of a continuous time series whose features are rolling
windows over the previous 15 samples. A test row and the training row recorded
10 s earlier share almost the same feature vector, so the test set is not
"unseen" data — this is why RF/XGB reached AUC = 1.0000.

This module provides two leakage-free splitters built around the real failure
events of MetroPT-3 (López de Prado, *Advances in Financial Machine Learning*,
2018, ch. 7):

* ``PurgedWalkForwardSplit`` — expanding window. Fold *i* tests the timeline
  segment containing failure *i* and trains only on data strictly before it.
  This is the deployment-faithful protocol: the model never sees the future.
* ``PurgedEventBlockKFold`` — leave-one-event-segment-out. Each fold tests one
  segment and trains on all other segments (past and future). Used only as a
  secondary protocol, because it allows training on the future.

In both, the training rows within ``purge`` seconds BEFORE the test segment
are removed (their rolling features overlap the test window) and, for the
blocked variant, the rows within ``embargo`` seconds AFTER the test segment
are removed too (serial correlation leaking back into training).

Segment boundaries are the midpoints between consecutive failure events, so
every test segment contains exactly one real failure plus the normal operation
around it — which is what allows event-level metrics (RF-29) per fold.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FailureEvent:
    """A real failure interval [start, end] (inclusive) from the incident report."""

    start: pd.Timestamp
    end: pd.Timestamp

    @classmethod
    def from_strings(cls, start: str, end: str) -> "FailureEvent":
        return cls(start=pd.Timestamp(start), end=pd.Timestamp(end))


@dataclass(frozen=True)
class TemporalFold:
    """Positional indices of one fold plus the metadata needed to report it."""

    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_event: FailureEvent
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def events_from_intervals(intervals: Sequence[tuple[str, str]]) -> list[FailureEvent]:
    """Build sorted ``FailureEvent`` objects from ``(start, end)`` strings."""
    events: list[FailureEvent] = [FailureEvent.from_strings(s, e) for s, e in intervals]
    return sorted(events, key=lambda ev: ev.start)


def segment_boundaries(
    timestamps: pd.Series, events: Sequence[FailureEvent]
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Split the timeline into one half-open segment ``[lo, hi)`` per event.

    Boundaries are midpoints between consecutive events; the first segment
    starts at the first timestamp and the last ends just after the last one.
    """
    if not events:
        raise ValueError("At least one failure event is required.")
    t_min: pd.Timestamp = pd.Timestamp(timestamps.min())
    t_max: pd.Timestamp = pd.Timestamp(timestamps.max()) + pd.Timedelta(seconds=1)

    cuts: list[pd.Timestamp] = [t_min]
    for prev, nxt in zip(events[:-1], events[1:]):
        cuts.append(prev.end + (nxt.start - prev.end) / 2)
    cuts.append(t_max)
    return [(cuts[i], cuts[i + 1]) for i in range(len(events))]


class _PurgedSplitBase:
    def __init__(
        self,
        events: Sequence[FailureEvent],
        purge_seconds: float = 300.0,
        embargo_seconds: float = 3600.0,
    ) -> None:
        if purge_seconds < 0 or embargo_seconds < 0:
            raise ValueError("purge_seconds and embargo_seconds must be >= 0.")
        self.events: list[FailureEvent] = sorted(events, key=lambda ev: ev.start)
        self.purge: pd.Timedelta = pd.Timedelta(seconds=purge_seconds)
        self.embargo: pd.Timedelta = pd.Timedelta(seconds=embargo_seconds)

    def _segments(
        self, timestamps: pd.Series
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        return segment_boundaries(timestamps, self.events)

    def get_n_splits(self) -> int:
        raise NotImplementedError

    def split(self, timestamps: pd.Series) -> Iterator[TemporalFold]:
        raise NotImplementedError


class PurgedWalkForwardSplit(_PurgedSplitBase):
    """
    Expanding-window walk-forward. Fold *k* (k >= 1) tests segment *k* and
    trains on every row before ``segment_start - purge``.

    Segment 0 is never a test segment: there is no earlier failure to learn
    from, so a supervised model would be trained without any positive label.
    """

    def get_n_splits(self) -> int:
        return max(len(self.events) - 1, 0)

    def split(self, timestamps: pd.Series) -> Iterator[TemporalFold]:
        ts: np.ndarray = pd.to_datetime(timestamps).to_numpy()
        segments = self._segments(timestamps)
        for k in range(1, len(segments)):
            lo, hi = segments[k]
            test_mask: np.ndarray = (ts >= lo.to_numpy()) & (ts < hi.to_numpy())
            train_mask: np.ndarray = ts < (lo - self.purge).to_numpy()
            yield TemporalFold(
                fold=k,
                train_idx=np.flatnonzero(train_mask),
                test_idx=np.flatnonzero(test_mask),
                test_event=self.events[k],
                test_start=lo,
                test_end=hi,
            )


class PurgedEventBlockKFold(_PurgedSplitBase):
    """
    Leave-one-event-segment-out with purge before and embargo after the test
    segment. Trains on the future — report only as a secondary protocol.
    """

    def get_n_splits(self) -> int:
        return len(self.events)

    def split(self, timestamps: pd.Series) -> Iterator[TemporalFold]:
        ts: np.ndarray = pd.to_datetime(timestamps).to_numpy()
        segments = self._segments(timestamps)
        for k, (lo, hi) in enumerate(segments):
            test_mask: np.ndarray = (ts >= lo.to_numpy()) & (ts < hi.to_numpy())
            before: np.ndarray = ts < (lo - self.purge).to_numpy()
            after: np.ndarray = ts >= (hi + self.embargo).to_numpy()
            yield TemporalFold(
                fold=k,
                train_idx=np.flatnonzero(before | after),
                test_idx=np.flatnonzero(test_mask),
                test_event=self.events[k],
                test_start=lo,
                test_end=hi,
            )
