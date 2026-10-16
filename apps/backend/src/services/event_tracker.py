"""
EventTracker — online alarm-episode detection (RN-03), pure logic.

Per asset, consumes ``(timestamp, probability)`` one prediction at a time:

* an episode OPENS when ``k_open`` consecutive probabilities are ≥ threshold;
  its start is the first sample of that streak (not the k-th), so the
  recorded start time does not lag by k samples;
* while open, every sample updates peak / mean / count;
* it CLOSES when ``k_close`` consecutive probabilities are < threshold; its end
  is the last sample that was still ≥ threshold.

No I/O here: ``FailureEventService`` persists what this returns, and the same
class rebuilds history offline from stored predictions, so online and
offline episodes are identical by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class EpisodeStats:
    started_at: datetime
    last_above_at: datetime
    peak: float
    mean: float
    n: int


@dataclass(frozen=True)
class TrackerUpdate:
    opened: EpisodeStats | None = None  # episode just opened (stats so far)
    extended: EpisodeStats | None = None  # open episode got a new sample
    closed: EpisodeStats | None = None  # episode just closed (final stats)


@dataclass
class _Streak:
    times: list[datetime] = field(default_factory=list)
    probs: list[float] = field(default_factory=list)


class EventTracker:
    def __init__(
        self, threshold: float = 0.7, k_open: int = 6, k_close: int = 6
    ) -> None:
        if k_open < 1 or k_close < 1:
            raise ValueError("k_open and k_close must be >= 1.")
        self.threshold = threshold
        self.k_open = k_open
        self.k_close = k_close
        self._pending = _Streak()
        self._below: int = 0
        self._open: bool = False
        self._start: datetime | None = None
        self._last_above: datetime | None = None
        self._peak: float = 0.0
        self._sum: float = 0.0
        self._n: int = 0

    @property
    def is_open(self) -> bool:
        return self._open

    def _stats(self) -> EpisodeStats:
        assert self._start is not None and self._last_above is not None
        return EpisodeStats(
            self._start, self._last_above, self._peak, self._sum / self._n, self._n
        )

    def update(self, ts: datetime, prob: float) -> TrackerUpdate:
        above = prob >= self.threshold
        if not self._open:
            if not above:
                self._pending = _Streak()
                return TrackerUpdate()
            self._pending.times.append(ts)
            self._pending.probs.append(prob)
            if len(self._pending.probs) < self.k_open:
                return TrackerUpdate()
            self._open = True
            self._start = self._pending.times[0]
            self._last_above = ts
            self._peak = max(self._pending.probs)
            self._sum = sum(self._pending.probs)
            self._n = len(self._pending.probs)
            self._pending = _Streak()
            self._below = 0
            return TrackerUpdate(opened=self._stats())

        if above:
            self._below = 0
            self._last_above = ts
            self._peak = max(self._peak, prob)
            self._sum += prob
            self._n += 1
            return TrackerUpdate(extended=self._stats())

        self._below += 1
        if self._below < self.k_close:
            return TrackerUpdate()
        final = self._stats()
        self._open = False
        self._below = 0
        self._start = self._last_above = None
        self._peak = self._sum = 0.0
        self._n = 0
        return TrackerUpdate(closed=final)


def severity_for(peak: float, critical_at: float = 0.9) -> str:
    """``critico`` when the episode peak reaches ``critical_at``, else ``alerta``."""
    return "critico" if peak >= critical_at else "alerta"
