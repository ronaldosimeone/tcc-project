"""
EventTracker — RN-03 (persistência) aplicada online.

Garantias verificadas
---------------------
1.  Menos de k_open amostras acima do limiar nunca abre episódio (anti-flapping).
2.  O início do episódio é a 1ª amostra da sequência, não a k-ésima.
3.  Picos/médias/contagem acompanham o episódio aberto.
4.  O episódio só fecha após k_close amostras abaixo; o fim é a última acima.
5.  Uma queda curta (< k_close) no meio não divide o episódio.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.services.event_tracker import EventTracker, severity_for

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _feed(tracker: EventTracker, probs: list[float]):
    return [
        tracker.update(T0 + timedelta(seconds=10 * i), p) for i, p in enumerate(probs)
    ]


def test_short_streak_never_opens() -> None:
    out = _feed(EventTracker(k_open=3), [0.9, 0.9, 0.1, 0.9, 0.9, 0.1])
    assert all(u.opened is None for u in out)


def test_open_start_is_first_sample_of_streak() -> None:
    out = _feed(EventTracker(k_open=3), [0.1, 0.8, 0.95, 0.85])
    opened = [u.opened for u in out if u.opened]
    assert len(opened) == 1
    assert opened[0].started_at == T0 + timedelta(seconds=10)
    assert opened[0].peak == pytest.approx(0.95) and opened[0].n == 3


def test_close_after_k_close_below_and_end_is_last_above() -> None:
    probs = [0.8] * 3 + [0.9, 0.1, 0.1, 0.1]
    out = _feed(EventTracker(k_open=3, k_close=3), probs)
    closed = [u.closed for u in out if u.closed]
    assert len(closed) == 1
    assert closed[0].last_above_at == T0 + timedelta(seconds=30)
    assert closed[0].n == 4 and closed[0].mean == pytest.approx((0.8 * 3 + 0.9) / 4)


def test_short_dip_does_not_split_episode() -> None:
    probs = [0.8] * 3 + [0.1, 0.1] + [0.8] * 2 + [0.1] * 3
    out = _feed(EventTracker(k_open=3, k_close=3), probs)
    assert sum(1 for u in out if u.opened) == 1
    assert sum(1 for u in out if u.closed) == 1


def test_invalid_k_rejected() -> None:
    with pytest.raises(ValueError):
        EventTracker(k_open=0)


def test_severity() -> None:
    # RF threshold 0.70: 0.95 is risk 0.94, 0.75 is risk 0.71
    assert severity_for(0.95, 0.70) == "critico"
    assert severity_for(0.75, 0.70) == "alerta"
    # same rule on a low-threshold model (MLP 0.06): raw 0.5 is not critical
    assert severity_for(0.50, 0.06) == "alerta"
    assert severity_for(0.80, 0.06) == "critico"
