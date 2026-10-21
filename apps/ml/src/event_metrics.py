"""
Event-level evaluation for predictive maintenance — RF-29 / RN-01 / RN-02 / RN-03.

Point-wise F1 counts 10-second samples. With only four real failures in
MetroPT-3, a model can score F1 = 0.95 while missing an entire failure (the
long June event dominates the positive samples). Operators care about other
questions, which this module answers per failure *event*:

* Was the failure detected at all?                        (event recall)
* How long before the failure started did the alarm fire? (lead time, RN-01)
* How many alarms fired with no failure nearby?           (false alarms / day)
* What is the expected cost of the decisions?             (asymmetric cost, RN-02)

Alarms are not raw thresholded scores: RN-03 requires ``k`` consecutive samples
above the threshold before an alarm is raised (anti-flapping). Consecutive
alarm samples are merged into *episodes*; one episode is one notification.

Range-based precision/recall follow Tatbul et al., "Precision and Recall for
Time Series", NeurIPS 2018 (flat positional bias, reciprocal cardinality).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.temporal_validation import FailureEvent


@dataclass(frozen=True)
class AlarmEpisode:
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass
class EventOutcome:
    event_start: str
    event_end: str
    detected: bool
    first_alarm: str | None
    # Positive = alarm fired BEFORE the failure started (early warning).
    # Negative = detection delay after the failure started.
    lead_time_s: float | None
    meets_min_lead: bool


@dataclass
class EventReport:
    n_events: int
    n_detected: int
    event_recall: float
    n_episodes: int
    n_false_alarm_episodes: int
    episode_precision: float
    false_alarm_time_s: float
    event_f1: float
    observed_days: float
    false_alarms_per_day: float
    mean_lead_time_s: float | None
    n_meeting_min_lead: int
    range_precision: float
    range_recall: float
    range_f1: float
    expected_cost: float
    outcomes: list[EventOutcome] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Alarms (RN-03)
# ---------------------------------------------------------------------------


def persistent_alarms(scores: np.ndarray, threshold: float, k: int) -> np.ndarray:
    """
    Boolean alarm per sample: True only when the current sample AND the
    previous ``k - 1`` samples all have ``score >= threshold``.
    """
    if k < 1:
        raise ValueError("k must be >= 1.")
    above: np.ndarray = np.asarray(scores) >= threshold
    if k == 1:
        return above
    # Running length of the current streak of consecutive True values.
    streak: np.ndarray = np.zeros(above.shape[0], dtype=np.int64)
    run: int = 0
    for i, flag in enumerate(above):
        run = run + 1 if flag else 0
        streak[i] = run
    return streak >= k


def alarm_episodes(
    timestamps: pd.Series, alarms: np.ndarray, max_gap_seconds: float = 60.0
) -> list[AlarmEpisode]:
    """Merge alarm samples separated by at most ``max_gap_seconds`` into episodes."""
    ts: pd.Series = pd.to_datetime(pd.Series(timestamps)).reset_index(drop=True)
    on: np.ndarray = np.flatnonzero(np.asarray(alarms))
    if on.size == 0:
        return []
    gap: pd.Timedelta = pd.Timedelta(seconds=max_gap_seconds)
    episodes: list[AlarmEpisode] = []
    start: pd.Timestamp = ts.iloc[on[0]]
    prev: pd.Timestamp = start
    for idx in on[1:]:
        cur: pd.Timestamp = ts.iloc[idx]
        if cur - prev > gap:
            episodes.append(AlarmEpisode(start=start, end=prev))
            start = cur
        prev = cur
    episodes.append(AlarmEpisode(start=start, end=prev))
    return episodes


# ---------------------------------------------------------------------------
# Range-based precision / recall (Tatbul et al., 2018)
# ---------------------------------------------------------------------------


def _overlap_seconds(
    a0: pd.Timestamp, a1: pd.Timestamp, b0: pd.Timestamp, b1: pd.Timestamp
) -> float:
    lo: pd.Timestamp = max(a0, b0)
    hi: pd.Timestamp = min(a1, b1)
    return max((hi - lo).total_seconds(), 0.0)


def _span(a0: pd.Timestamp, a1: pd.Timestamp) -> float:
    # Inclusive ranges sampled every ~10 s: a single-sample range still has
    # a non-zero extent so overlap fractions stay well defined.
    return max((a1 - a0).total_seconds(), 10.0)


def range_precision_recall(
    episodes: Sequence[AlarmEpisode],
    events: Sequence[FailureEvent],
    alpha: float = 0.5,
) -> tuple[float, float]:
    """
    Range-based recall (existence weight ``alpha``) and precision (alpha = 0),
    flat positional bias and reciprocal cardinality factor.
    """
    if not events:
        return (0.0, 0.0)

    recalls: list[float] = []
    for ev in events:
        overlapping = [
            ep
            for ep in episodes
            if _overlap_seconds(ep.start, ep.end, ev.start, ev.end) > 0
        ]
        existence: float = 1.0 if overlapping else 0.0
        covered: float = sum(
            _overlap_seconds(ep.start, ep.end, ev.start, ev.end) for ep in overlapping
        )
        cardinality: float = 1.0 / len(overlapping) if overlapping else 0.0
        frac: float = min(covered / _span(ev.start, ev.end), 1.0)
        recalls.append(alpha * existence + (1 - alpha) * frac * cardinality)

    if not episodes:
        return (0.0, float(np.mean(recalls)))

    precisions: list[float] = []
    for ep in episodes:
        overlapping_ev = [
            ev
            for ev in events
            if _overlap_seconds(ep.start, ep.end, ev.start, ev.end) > 0
        ]
        covered = sum(
            _overlap_seconds(ep.start, ep.end, ev.start, ev.end)
            for ev in overlapping_ev
        )
        cardinality = 1.0 / len(overlapping_ev) if overlapping_ev else 0.0
        precisions.append(min(covered / _span(ep.start, ep.end), 1.0) * cardinality)

    return (float(np.mean(precisions)), float(np.mean(recalls)))


def _f1(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Event evaluation
# ---------------------------------------------------------------------------


def evaluate_events(
    timestamps: pd.Series,
    alarms: np.ndarray,
    events: Sequence[FailureEvent],
    early_horizon_s: float = 6 * 3600.0,
    grace_s: float = 3600.0,
    min_lead_s: float = 2 * 3600.0,
    cost_fn: float = 100.0,
    cost_fp: float = 1.0,
    max_gap_seconds: float = 60.0,
    renotify_s: float = 3600.0,
) -> EventReport:
    """
    Evaluate alarms against the failure events contained in ``timestamps``.

    An event counts as detected if an alarm episode overlaps the window
    ``[start - early_horizon_s, end]``. Its lead time is ``start - first alarm``
    inside that window.

    False alarms are counted on the alarm TIME outside every
    ``[start - early_horizon_s, end + grace_s]`` window: each contiguous
    out-of-window stretch counts one false alarm per started ``renotify_s``
    (the operator is re-notified while the alarm stays on). Without this an
    always-on alarm is a single episode touching the failure — "detected with
    zero false alarms" — a degenerate optimum a cost-driven threshold search
    can and did find (autoencoder). An episode entirely outside the windows
    still counts at least one.

    ``min_lead_s`` implements RN-01; ``cost_fn``/``cost_fp`` implement RN-02:
    ``expected_cost = cost_fn * missed_events + cost_fp * false_alarm_episodes``.
    """
    ts: pd.Series = pd.to_datetime(pd.Series(timestamps)).reset_index(drop=True)
    t0: pd.Timestamp = ts.min()
    t1: pd.Timestamp = ts.max()
    in_range: list[FailureEvent] = [
        ev for ev in events if ev.end >= t0 and ev.start <= t1
    ]

    episodes: list[AlarmEpisode] = alarm_episodes(ts, alarms, max_gap_seconds)
    horizon: pd.Timedelta = pd.Timedelta(seconds=early_horizon_s)
    grace: pd.Timedelta = pd.Timedelta(seconds=grace_s)

    outcomes: list[EventOutcome] = []
    for ev in in_range:
        win0: pd.Timestamp = ev.start - horizon
        hits = [ep for ep in episodes if ep.end >= win0 and ep.start <= ev.end]
        if not hits:
            outcomes.append(
                EventOutcome(str(ev.start), str(ev.end), False, None, None, False)
            )
            continue
        first: pd.Timestamp = max(min(ep.start for ep in hits), win0)
        lead: float = (ev.start - first).total_seconds()
        outcomes.append(
            EventOutcome(
                str(ev.start), str(ev.end), True, str(first), lead, lead >= min_lead_s
            )
        )

    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = sorted(
        (ev.start - horizon, ev.end + grace) for ev in in_range
    )

    def _outside(ep: AlarmEpisode) -> list[float]:
        """Durations (s) of the parts of ``ep`` outside every window
        (interval difference: episode minus the union of the windows)."""
        segs: list[tuple[pd.Timestamp, pd.Timestamp]] = [(ep.start, ep.end)]
        for w0, w1 in windows:
            kept: list[tuple[pd.Timestamp, pd.Timestamp]] = []
            for a, b in segs:
                if w1 < a or w0 > b:
                    kept.append((a, b))
                    continue
                if a < w0:
                    kept.append((a, w0))
                if b > w1:
                    kept.append((w1, b))
            segs = kept
        return [(b - a).total_seconds() for a, b in segs]

    def _n_false(stretches: list[float]) -> int:
        # one notification per started renotify_s (a 1-sample stretch is one)
        return sum(int(np.ceil(max(d, 10.0) / renotify_s)) for d in stretches)

    out_parts: list[list[float]] = [_outside(ep) for ep in episodes]
    n_false: int = sum(_n_false(p) for p in out_parts)
    false_time: float = float(sum(sum(p) for p in out_parts))
    n_false_eps: int = sum(1 for p in out_parts if p)
    n_detected: int = sum(1 for o in outcomes if o.detected)
    n_events: int = len(in_range)

    event_recall: float = n_detected / n_events if n_events else 0.0
    episode_precision: float = (
        (len(episodes) - n_false_eps) / len(episodes) if episodes else 0.0
    )
    observed_days: float = max((t1 - t0).total_seconds() / 86400.0, 1e-9)
    leads: list[float] = [o.lead_time_s for o in outcomes if o.lead_time_s is not None]
    r_prec, r_rec = range_precision_recall(episodes, in_range)

    return EventReport(
        n_events=n_events,
        n_detected=n_detected,
        event_recall=event_recall,
        n_episodes=len(episodes),
        n_false_alarm_episodes=n_false,
        episode_precision=episode_precision,
        false_alarm_time_s=false_time,
        event_f1=_f1(episode_precision, event_recall),
        observed_days=observed_days,
        false_alarms_per_day=n_false / observed_days,
        mean_lead_time_s=float(np.mean(leads)) if leads else None,
        n_meeting_min_lead=sum(1 for o in outcomes if o.meets_min_lead),
        range_precision=r_prec,
        range_recall=r_rec,
        range_f1=_f1(r_prec, r_rec),
        expected_cost=cost_fn * (n_events - n_detected) + cost_fp * n_false,
        outcomes=outcomes,
    )


def cost_optimal_threshold(
    timestamps: pd.Series,
    scores: np.ndarray,
    events: Sequence[FailureEvent],
    k: int,
    candidates: Sequence[float] | None = None,
    **eval_kwargs: Any,
) -> tuple[float, float]:
    """
    RN-02: pick the threshold that minimises expected cost on a VALIDATION
    slice (never on the test fold). Ties are broken by the higher threshold
    (fewer alarms). Returns ``(threshold, cost)``.
    """
    grid: Sequence[float] = (
        candidates
        if candidates is not None
        else np.round(np.arange(0.05, 1.0, 0.05), 2)
    )
    best: tuple[float, float] = (0.5, float("inf"))
    for thr in grid:
        alarms: np.ndarray = persistent_alarms(scores, float(thr), k)
        cost: float = evaluate_events(
            timestamps, alarms, events, **eval_kwargs
        ).expected_cost
        if cost < best[1] or (cost == best[1] and thr > best[0]):
            best = (float(thr), cost)
    return best
