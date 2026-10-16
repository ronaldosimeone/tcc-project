"""
Unit tests for src/rul.py — RF-30.

Guarantees verified
-------------------
1.  Observed samples: RUL = hours until the NEXT failure start.
2.  Samples after the last failure are right-censored (upper = +inf).
3.  Samples inside a failure interval are excluded.
4.  C-index = 1 for a perfect ranking, 0 for the inverse, 0.5 for constant.
5.  MAE only counts observed samples within the horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rul import concordance_index, mae_near_failure, survival_labels
from src.temporal_validation import events_from_intervals


@pytest.fixture()
def labels():
    ts = pd.Series(pd.date_range("2020-01-01 00:00", "2020-01-05 00:00", freq="1h"))
    events = events_from_intervals([("2020-01-02 00:00", "2020-01-02 03:00")])
    return ts, survival_labels(ts, events)


def test_observed_rul_counts_hours_to_next_start(labels) -> None:
    ts, lab = labels
    i = int(np.flatnonzero(ts == pd.Timestamp("2020-01-01 06:00"))[0])
    assert (
        lab.observed[i]
        and lab.lower[i] == pytest.approx(18.0)
        and lab.upper[i] == pytest.approx(18.0)
    )


def test_after_last_failure_is_right_censored(labels) -> None:
    ts, lab = labels
    i = int(np.flatnonzero(ts == pd.Timestamp("2020-01-03 00:00"))[0])
    assert not lab.observed[i]
    assert lab.lower[i] == pytest.approx(48.0)  # end of data (Jan 5) − t (Jan 3)
    assert np.isinf(lab.upper[i])


def test_inside_failure_is_invalid(labels) -> None:
    ts, lab = labels
    inside = (ts >= pd.Timestamp("2020-01-02 00:00")) & (
        ts <= pd.Timestamp("2020-01-02 03:00")
    )
    assert not lab.valid[inside.to_numpy()].any()
    assert lab.valid[~inside.to_numpy()].all()


def test_concordance_extremes() -> None:
    t = np.array([1.0, 2.0, 3.0, 4.0])
    e = np.ones(4, dtype=bool)
    assert concordance_index(t, e, t) == pytest.approx(1.0)
    assert concordance_index(t, e, -t) == pytest.approx(0.0)
    assert concordance_index(t, e, np.zeros(4)) == pytest.approx(0.5)


def test_censored_subject_is_never_the_earlier_one() -> None:
    t = np.array([1.0, 5.0])
    e = np.array([False, True])  # first is censored at 1 h: not comparable
    assert np.isnan(concordance_index(t, e, np.array([0.0, 1.0])))


def test_mae_near_failure() -> None:
    true_h = np.array([1.0, 10.0, 100.0, 2.0])
    pred_h = np.array([2.0, 12.0, 0.0, 2.0])
    observed = np.array([True, True, True, False])
    assert mae_near_failure(true_h, pred_h, observed, horizon_h=24) == pytest.approx(
        1.5
    )


def test_training_cutoff_censors_future_failures() -> None:
    """A failure after the training cut-off must not leak into the labels."""
    ts = pd.Series(pd.date_range("2020-01-01 00:00", "2020-01-03 00:00", freq="1h"))
    events = events_from_intervals([("2020-01-05 00:00", "2020-01-05 03:00")])
    lab = survival_labels(ts, events, end=pd.Timestamp("2020-01-03 00:00"))
    assert not lab.observed.any()
    assert np.isinf(lab.upper).all()
    assert lab.lower[0] == pytest.approx(48.0)
