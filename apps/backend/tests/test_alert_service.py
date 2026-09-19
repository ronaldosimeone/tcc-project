"""
Testes de `AlertService` (RF-14/RF-24, RNF-64/RNF-65).

Auditoria desta task: nenhum arquivo dedicado existia — `AlertService` só
era exercitado indiretamente (`test_full_pipeline.py`,
`test_inference_pipeline.py`) com asserções parciais (`payload["probability"]`
apenas), nunca o dict completo nem os defaults quando campos opcionais do
`prediction` bruto estão ausentes.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import src.services.alert_service as alert_service_module
from src.core.ws_manager import ALERT_PROBABILITY_THRESHOLD
from src.services.alert_service import (
    AlertService,
    get_alert_service,
    get_alert_settings_service,
    get_notification_test_service,
)
from src.services.critical_failure_notification_service import (
    CriticalFailureNotificationService,
)


def _make_service(critical_notifier: Any = None) -> tuple[AlertService, MagicMock]:
    ws_manager = MagicMock()
    ws_manager.broadcast_alert = AsyncMock()
    service = AlertService(ws_manager, critical_notifier=critical_notifier)
    return service, ws_manager


# ---------------------------------------------------------------------------
# process_prediction — payload exato + defaults de campos opcionais
# ---------------------------------------------------------------------------


async def test_payload_has_exactly_the_7_expected_keys() -> None:
    service, _ = _make_service()
    payload = await service.process_prediction({"probability": 0.1})
    assert set(payload.keys()) == {
        "type",
        "message_id",
        "timestamp",
        "probability",
        "label",
        "sensor_id",
        "triggered",
        "inference_latency_ms",
    }


async def test_payload_type_is_exactly_alert() -> None:
    service, _ = _make_service()
    payload = await service.process_prediction({"probability": 0.1})
    assert payload["type"] == "alert"


async def test_probability_defaults_to_0_0_when_key_absent() -> None:
    service, _ = _make_service()
    payload = await service.process_prediction({})
    assert payload["probability"] == 0.0
    assert payload["triggered"] is False


async def test_label_defaults_to_unknown_when_key_absent() -> None:
    service, _ = _make_service()
    payload = await service.process_prediction({"probability": 0.1})
    assert payload["label"] == "unknown"


async def test_label_passes_through_when_present() -> None:
    service, _ = _make_service()
    payload = await service.process_prediction({"probability": 0.1, "label": "failure"})
    assert payload["label"] == "failure"


async def test_sensor_id_defaults_to_none_when_key_absent() -> None:
    service, _ = _make_service()
    payload = await service.process_prediction({"probability": 0.1})
    assert payload["sensor_id"] is None


async def test_inference_latency_ms_is_none_when_key_absent() -> None:
    service, _ = _make_service()
    payload = await service.process_prediction({"probability": 0.1})
    assert payload["inference_latency_ms"] is None


async def test_inference_latency_ms_rounds_to_exactly_2_decimal_places() -> None:
    service, _ = _make_service()
    payload = await service.process_prediction(
        {"probability": 0.1, "inference_latency_ms": 12.34567}
    )
    assert payload["inference_latency_ms"] == 12.35


async def test_inference_latency_ms_present_and_correctly_extracted() -> None:
    """Mata o typo de chave `inference_latency_ms` -> `XXinference_latency_msXX`:
    com a chave errada, `raw_latency` seria sempre `None` mesmo com o valor
    presente no dict de entrada."""
    service, _ = _make_service()
    payload = await service.process_prediction(
        {"probability": 0.1, "inference_latency_ms": 5.0}
    )
    assert payload["inference_latency_ms"] == 5.0


# ---------------------------------------------------------------------------
# triggered — fronteira estrita (`>`, nunca `>=`) — RF-14
# ---------------------------------------------------------------------------


async def test_triggered_is_false_exactly_at_the_threshold() -> None:
    service, ws_manager = _make_service()
    payload = await service.process_prediction(
        {"probability": ALERT_PROBABILITY_THRESHOLD}
    )
    assert payload["triggered"] is False
    ws_manager.broadcast_alert.assert_not_awaited()


async def test_triggered_is_true_just_above_the_threshold() -> None:
    service, ws_manager = _make_service()
    payload = await service.process_prediction(
        {"probability": ALERT_PROBABILITY_THRESHOLD + 0.0001}
    )
    assert payload["triggered"] is True
    ws_manager.broadcast_alert.assert_awaited_once()


# ---------------------------------------------------------------------------
# critical_notifier — kwargs exatos + fallback equipment_id/name (RF-24)
# ---------------------------------------------------------------------------


def _make_spy_notifier() -> Any:
    notifier = MagicMock(spec=CriticalFailureNotificationService)
    notifier.notify_if_critical = AsyncMock()
    return notifier


async def test_critical_notifier_receives_equipment_id_from_prediction_when_present() -> (
    None
):
    notifier = _make_spy_notifier()
    service, _ = _make_service(critical_notifier=notifier)
    await service.process_prediction(
        {"probability": 0.9, "equipment_id": "eq-42", "equipment_name": "Bomba X"}
    )
    notifier.notify_if_critical.assert_awaited_once()
    kwargs = notifier.notify_if_critical.await_args.kwargs
    assert kwargs["equipment_id"] == "eq-42"
    assert kwargs["equipment_name"] == "Bomba X"
    assert kwargs["probability"] == 0.9


async def test_critical_notifier_falls_back_to_default_equipment_id_when_absent() -> (
    None
):
    """RNF-64: `prediction.get("equipment_id") or settings.default_equipment_id`
    — um mutante `or`->`and` faria isso virar `None and default` = `None`
    quando a chave está ausente (`.get()` devolve `None`, falsy), em vez do
    fallback real."""
    from src.core.config import settings

    notifier = _make_spy_notifier()
    service, _ = _make_service(critical_notifier=notifier)
    await service.process_prediction({"probability": 0.9})

    kwargs = notifier.notify_if_critical.await_args.kwargs
    assert kwargs["equipment_id"] == settings.default_equipment_id
    assert kwargs["equipment_name"] == settings.default_equipment_name


async def test_critical_notifier_not_called_when_none() -> None:
    service, _ = _make_service(critical_notifier=None)
    # Não deve lançar mesmo sem notifier configurado.
    await service.process_prediction({"probability": 0.9})


# ---------------------------------------------------------------------------
# Singletons de módulo (RF-24/RF-25) — construídos de verdade no import,
# nunca `None` (composição real das factories `get_*`).
# ---------------------------------------------------------------------------


def test_module_level_telegram_adapter_is_constructed_not_none() -> None:
    assert alert_service_module._telegram_adapter is not None  # noqa: SLF001


def test_module_level_email_adapter_is_constructed_not_none() -> None:
    assert alert_service_module._email_adapter is not None  # noqa: SLF001


def test_module_level_alert_settings_service_is_constructed_not_none() -> None:
    assert alert_service_module._alert_settings_service is not None  # noqa: SLF001


def test_module_level_critical_notifier_is_constructed_not_none() -> None:
    assert alert_service_module._critical_notifier is not None  # noqa: SLF001


def test_notification_test_rate_limiter_ttl_is_exactly_10_seconds() -> None:
    assert (
        alert_service_module._notification_test_rate_limiter._ttl_seconds
        == 10  # noqa: SLF001
    )


def test_get_alert_service_returns_a_wired_service_with_the_module_singletons() -> None:
    service = get_alert_service()
    assert isinstance(service, AlertService)
    assert (
        service._critical_notifier is alert_service_module._critical_notifier
    )  # noqa: SLF001


def test_get_alert_settings_service_returns_the_module_singleton() -> None:
    assert (
        get_alert_settings_service()
        is alert_service_module._alert_settings_service  # noqa: SLF001
    )


def test_get_notification_test_service_returns_the_module_singleton() -> None:
    assert (
        get_notification_test_service()
        is alert_service_module._notification_test_service  # noqa: SLF001
    )
