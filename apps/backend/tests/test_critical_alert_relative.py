"""
Critical notifications (RF-24/RF-25) follow the active model's calibrated
threshold, like the dashboard.

Found after the model-relative dashboard fix: the configurable 85 % critical
threshold compared RAW probability, so models calibrated with a low decision
threshold (MLP 0.06, BiLSTM 0.10) could decide "failure" for hours and never
trigger a Telegram/e-mail alert.

The setting is now a RISK LEVEL on the scale the UI uses: 0.65 = the model's
decision threshold, 1.0 = probability 1. Without a decision threshold (legacy
callers) the original raw-probability rule (> setting) is kept.
"""

from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.risk import CRITICAL_RISK, probability_for_risk, risk_index
from src.services.alert_settings_service import AlertSettingsSnapshot
from src.services.critical_failure_notification_service import (
    CriticalFailureNotificationService,
)


@pytest.mark.parametrize(
    ("p", "thr", "expected"),
    [(0.06, 0.06, 0.65), (0.03, 0.06, 0.325), (1.0, 0.06, 1.0), (0.0, 0.5, 0.0)],
)
def test_risk_index_scale(p: float, thr: float, expected: float) -> None:
    assert risk_index(p, thr) == pytest.approx(expected)


@pytest.mark.parametrize("thr", [0.06, 0.1, 0.694, 0.984])
@pytest.mark.parametrize("r", [0.35, 0.5, 0.65, 0.85, 0.95])
def test_probability_for_risk_inverts_the_index(thr: float, r: float) -> None:
    assert risk_index(probability_for_risk(r, thr), thr) == pytest.approx(r)


def test_critical_level_is_the_model_decision() -> None:
    assert CRITICAL_RISK == 0.65
    assert probability_for_risk(CRITICAL_RISK, 0.1) == pytest.approx(0.1)


def _service(level: float) -> tuple[CriticalFailureNotificationService, MagicMock]:
    settings = MagicMock()
    settings.get_settings = AsyncMock(
        return_value=AlertSettingsSnapshot(
            alert_threshold=level,
            telegram_enabled=True,
            email_enabled=False,
            alert_email=None,
        )
    )
    limiter = MagicMock()
    limiter.try_acquire = AsyncMock(return_value=True)
    svc = CriticalFailureNotificationService(
        adapter=MagicMock(),
        rate_limiter=limiter,
        dashboard_url="http://x",
        settings_service=settings,
    )
    svc.dispatch_channels = AsyncMock()  # type: ignore[method-assign]
    return svc, limiter


def _dispatch(svc: CriticalFailureNotificationService) -> AsyncMock:
    return cast(AsyncMock, svc.dispatch_channels)


async def _notify(
    svc: CriticalFailureNotificationService, p: float, thr: float | None
) -> None:
    await svc.notify_if_critical(
        equipment_id="A",
        equipment_name="APU",
        probability=p,
        timestamp="t",
        decision_threshold=thr,
    )


async def test_low_threshold_model_notifies_at_its_decision() -> None:
    svc, _ = _service(0.65)
    await _notify(svc, 0.30, 0.06)  # MLP: P = 0.30 is far above its 0.06
    _dispatch(svc).assert_awaited_once()


async def test_below_the_level_does_not_notify() -> None:
    svc, _ = _service(0.85)
    # BiLSTM thr 0.10: level 0.85 needs P >= 0.10 + (0.2/0.35)*0.9 ≈ 0.614
    await _notify(svc, 0.40, 0.10)
    _dispatch(svc).assert_not_awaited()
    await _notify(svc, 0.62, 0.10)
    _dispatch(svc).assert_awaited_once()


async def test_legacy_raw_rule_without_threshold() -> None:
    svc, _ = _service(0.85)
    await _notify(svc, 0.85, None)  # strict > 0.85, as before
    _dispatch(svc).assert_not_awaited()
    await _notify(svc, 0.86, None)
    _dispatch(svc).assert_awaited_once()


async def test_alert_service_forwards_the_decision_threshold() -> None:
    from src.services.alert_service import AlertService

    notifier = MagicMock()
    notifier.notify_if_critical = AsyncMock()
    ws = MagicMock()
    ws.broadcast_alert = AsyncMock()
    await AlertService(ws, critical_notifier=notifier).process_prediction(
        {"probability": 0.3, "decision_threshold": 0.06}
    )
    assert notifier.notify_if_critical.call_args.kwargs["decision_threshold"] == 0.06


# ---------------------------------------------------------------------------
# RF-22 — the automatic RAG plan of an opened failure event follows the
# model's decision too (an event opens only when the model decided failure;
# its peak probability can be 0.3-0.5 for low-threshold models).
# ---------------------------------------------------------------------------


async def test_suggestion_follows_the_model_decision_when_threshold_known() -> None:
    from src.schemas.maintenance import MaintenanceSuggestionRequest
    from src.services.maintenance_suggestion_service import (
        MaintenanceSuggestionService,
    )

    mcp = MagicMock()
    mcp.search_maintenance_manual = AsyncMock(return_value={"results": []})
    ollama = MagicMock()
    ollama.generate = AsyncMock(return_value="# Plano")
    svc = MaintenanceSuggestionService(
        mcp_client=mcp, ollama_client=ollama, model="llama3.2:3b"
    )

    low = await svc.suggest(
        MaintenanceSuggestionRequest(failure_probability=0.41, decision_threshold=0.24)
    )
    assert low.triggered is True  # BiLSTM event peak 0.41 >= its 0.24

    below = await svc.suggest(
        MaintenanceSuggestionRequest(failure_probability=0.2, decision_threshold=0.24)
    )
    assert below.triggered is False

    legacy = await svc.suggest(MaintenanceSuggestionRequest(failure_probability=0.41))
    assert legacy.triggered is False  # manual request: RF-22 > 0.7 unchanged
