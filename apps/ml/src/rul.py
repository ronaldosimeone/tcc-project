"""
Remaining Useful Life (RUL) as survival analysis — RF-30.

The classifiers answer "is the compressor failing now?". RUL answers "how
long until the next failure?" — the quantity a maintenance planner needs.

Labels (per 10 s sample at time t):
* next failure starts at s > t      -> event observed, T = s − t (hours)
* no further failure in the dataset -> RIGHT-CENSORED: T ≥ end − t
* t inside a failure interval       -> excluded (RUL is 0 by definition)

Censoring matters: after the last MetroPT-3 failure (July 15) there are
~6 weeks of data whose true RUL is unknown but at least the remaining
observation time. Treating them as "RUL = time to end of data" would teach
the model that failures happen exactly when the recording stops.

Model: XGBoost Accelerated Failure Time (``survival:aft``, Barnwal et al.,
2022) — handles interval/right censoring natively, no extra dependency.

Metrics:
* Harrell's concordance index (C-index) — ranking quality, censoring-aware.
* MAE (hours) on observed samples within ``horizon_h`` before a failure.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.temporal_validation import FailureEvent


@dataclass(frozen=True)
class SurvivalLabels:
    """``lower``/``upper`` bounds in hours (upper = +inf when censored)."""

    lower: np.ndarray
    upper: np.ndarray
    observed: np.ndarray
    valid: np.ndarray  # False for samples inside a failure interval


def survival_labels(
    timestamps: pd.Series,
    events: Sequence[FailureEvent],
    end: pd.Timestamp | None = None,
) -> SurvivalLabels:
    """
    ``end`` = observation cut-off. For TRAINING labels it must be the training
    cut-off and ``events`` only the failures that started before it: a sample
    whose next failure lies after the cut-off is censored there — otherwise
    the label would leak the future failure date into training.
    """
    ts: np.ndarray = pd.to_datetime(timestamps).to_numpy()
    if end is not None:
        events = [ev for ev in events if ev.start <= end]
    n: int = ts.size
    starts: np.ndarray = np.array(
        [ev.start.to_datetime64() for ev in events], dtype="datetime64[ns]"
    )
    ends: np.ndarray = np.array(
        [ev.end.to_datetime64() for ev in events], dtype="datetime64[ns]"
    )
    order: np.ndarray = np.argsort(starts)
    starts, ends = starts[order], ends[order]

    inside: np.ndarray = np.zeros(n, dtype=bool)
    for s, e in zip(starts, ends):
        inside |= (ts >= s) & (ts <= e)

    # Index of the first failure starting strictly after t.
    nxt: np.ndarray = np.searchsorted(starts, ts, side="right")
    observed: np.ndarray = nxt < starts.size
    hours = np.full(n, np.nan)
    hours[observed] = (starts[nxt[observed]] - ts[observed]) / np.timedelta64(1, "h")
    end_t: np.datetime64 = (
        ts.max() if end is None else pd.Timestamp(end).to_datetime64()
    )
    hours[~observed] = (end_t - ts[~observed]) / np.timedelta64(1, "h")

    lower: np.ndarray = np.maximum(hours, 1e-3)
    upper: np.ndarray = np.where(observed, lower, np.inf)
    return SurvivalLabels(lower=lower, upper=upper, observed=observed, valid=~inside)


def concordance_index(
    time: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    max_pairs_n: int = 6000,
    seed: int = 0,
) -> float:
    """
    Harrell's C: among comparable pairs (i fails first, observed), the share
    where the model also predicts a shorter time for i. Ties in prediction
    count 0.5. Subsamples to ``max_pairs_n`` points (O(n²) memory).
    """
    t: np.ndarray = np.asarray(time, dtype=float)
    e: np.ndarray = np.asarray(observed, dtype=bool)
    p: np.ndarray = np.asarray(predicted, dtype=float)
    if t.size > max_pairs_n:
        idx = np.random.default_rng(seed).choice(t.size, max_pairs_n, replace=False)
        t, e, p = t[idx], e[idx], p[idx]
    ti, tj = t[:, None], t[None, :]
    comparable: np.ndarray = (ti < tj) & e[:, None]
    if not comparable.any():
        return float("nan")
    pi, pj = p[:, None], p[None, :]
    concordant: np.ndarray = (pi < pj) & comparable
    tied: np.ndarray = (pi == pj) & comparable
    return float((concordant.sum() + 0.5 * tied.sum()) / comparable.sum())


def mae_near_failure(
    true_h: np.ndarray, pred_h: np.ndarray, observed: np.ndarray, horizon_h: float
) -> float:
    mask: np.ndarray = observed & (true_h <= horizon_h)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(true_h[mask] - pred_h[mask])))
