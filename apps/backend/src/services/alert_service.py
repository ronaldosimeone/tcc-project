"""
Alert service — business logic for RF-14 / RF-24 / RF-25.

Responsible for:
- Enriching raw ML prediction payloads with alert metadata.
- Delegating WebSocket broadcast to ConnectionManager when probability > 0.70.
- Delegating critical-failure Telegram notification (RF-24) when
  probability exceeds the effective threshold, via
  CriticalFailureNotificationService (optional — ``None`` when Telegram
  isn't configured, see ``get_alert_service``). RF-25: that threshold is
  now the global setting from ``AlertSettingsService`` when one has been
  saved via ``PUT /v1/settings/alerts``, falling back to the fixed
  ``CRITICAL_FAILURE_THRESHOLD`` (0.85) otherwise.

Deliberately decoupled from the transport layer so it can be tested without
a real WebSocket connection. RF-24 reuses this SAME integration point
(Inference -> AlertService) instead of a second hook into the ML pipeline —
no second inference, no duplicated threshold-checking call site.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog

from src.core.config import settings
from src.core.ws_manager import (
    ALERT_PROBABILITY_THRESHOLD,
    ConnectionManager,
    manager as _ws_manager,
)
from src.services.alert_settings_service import AlertSettingsService
from src.services.critical_failure_notification_service import (
    CRITICAL_FAILURE_THRESHOLD,
    CriticalFailureNotificationService,
)
from src.services.email_notification_adapter import EmailNotificationAdapter
from src.services.notification_test_service import NotificationTestService
from src.services.telegram_alert_rate_limiter import TelegramAlertRateLimiter
from src.services.telegram_notification_adapter import TelegramNotificationAdapter
from src.tasks.notification_tasks import enqueue_critical_failure_notification

log = structlog.get_logger(__name__)


class AlertService:
    """
    Orchestrates alert creation and push delivery.

    Injected into routers via ``fastapi.Depends`` — never instantiated
    directly outside of tests.
    """

    def __init__(
        self,
        ws_manager: ConnectionManager,
        critical_notifier: CriticalFailureNotificationService | None = None,
    ) -> None:
        self._manager = ws_manager
        # ``None`` é um estado válido em testes/ambientes sem Telegram
        # configurado — RF-24 simplesmente não dispara notificação nesse caso.
        self._critical_notifier = critical_notifier

    async def process_prediction(self, prediction: dict[str, Any]) -> dict[str, Any]:
        """
        Enrich a raw model prediction and broadcast an alert if RF-14 fires.

        Parameters
        ----------
        prediction:
            Raw output from the ML pipeline.  Expected keys:
            ``probability`` (float), ``label`` (str), ``sensor_id`` (str),
            and (RF-24) ``equipment_id``/``equipment_name`` — default to
            ``settings.default_equipment_id``/``default_equipment_name``
            when absent (today's pipeline simulates a single real asset).

        Returns
        -------
        dict
            Enriched alert payload (also persisted by the caller).
        """
        probability: float = float(prediction.get("probability", 0.0))
        # Latência medida no pipeline upstream; arredondada a 2 casas para
        # manter o JSON leve. Opcional — payloads antigos sem latência ainda
        # funcionam (frontend usa fallback).
        raw_latency = prediction.get("inference_latency_ms")
        inference_latency_ms: float | None = (
            round(float(raw_latency), 2) if raw_latency is not None else None
        )
        alert_payload: dict[str, Any] = {
            "type": "alert",
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "probability": probability,
            "label": prediction.get("label", "unknown"),
            "sensor_id": prediction.get("sensor_id"),
            "triggered": probability > ALERT_PROBABILITY_THRESHOLD,
            "inference_latency_ms": inference_latency_ms,
        }

        if alert_payload["triggered"]:
            log.info(
                "alert_triggered",
                message_id=alert_payload["message_id"],
                probability=probability,
                sensor_id=alert_payload["sensor_id"],
            )
            await self._manager.broadcast_alert(alert_payload)
        else:
            log.debug(
                "alert_skipped",
                probability=probability,
                threshold=ALERT_PROBABILITY_THRESHOLD,
            )

        # RF-24 — mesmo `prediction` já usado para o alerta WS acima; nenhuma
        # segunda inferência, nenhum novo hook no pipeline de ML.
        if self._critical_notifier is not None:
            await self._critical_notifier.notify_if_critical(
                equipment_id=prediction.get("equipment_id")
                or settings.default_equipment_id,
                equipment_name=prediction.get("equipment_name")
                or settings.default_equipment_name,
                probability=probability,
                timestamp=alert_payload["timestamp"],
            )

        return alert_payload


# RF-24 — singletons de módulo (mesmo padrão de `_ws_manager` acima):
# construídos uma vez, reaproveitados em toda chamada de `get_alert_service`
# (chamada por request HTTP e a cada tick do InferencePipelineService — não
# reconstruir o adapter/rate limiter a cada vez). O adapter é sempre
# construído mesmo sem token/chat_id configurados — a checagem de
# configuração ausente acontece em `send_critical_failure` (RF-24 §11 N/O),
# não aqui, para não duplicar essa regra em dois lugares.
_telegram_adapter = TelegramNotificationAdapter(
    bot_token=settings.telegram_bot_token,
    chat_id=settings.telegram_chat_id,
    api_base_url=settings.telegram_api_base_url,
    timeout=settings.telegram_client_timeout_seconds,
)
_telegram_rate_limiter = TelegramAlertRateLimiter(
    ttl_seconds=settings.critical_failure_rate_limit_seconds
)
# RF-25 — e-mail via Resend (RNF-49). Auditado antes desta task: nenhuma
# integração de e-mail existia no projeto — primeiro canal de e-mail do
# PredictIQ. Sempre construído mesmo sem API key configurada — a checagem
# de configuração ausente acontece em `_send` (mesmo padrão do Telegram).
_email_adapter = EmailNotificationAdapter(
    api_key=settings.resend_api_key,
    from_email=settings.resend_from_email,
    api_base_url=settings.resend_api_base_url,
    timeout=settings.resend_client_timeout_seconds,
)
# RF-25 — configuração global do limiar + canais, com fallback para o
# comportamento fixo do RF-24 (CRITICAL_FAILURE_THRESHOLD = 0.85, Telegram
# sempre ligado, e-mail sempre desligado) enquanto ninguém configurou nada
# em /v1/settings/alerts.
_alert_settings_service = AlertSettingsService(
    default_threshold=CRITICAL_FAILURE_THRESHOLD
)
# RNF-50/51 — `enqueue_notification` real: `notify_if_critical` decide
# enviar e adquire o rate limit aqui mesmo (Postgres, síncrono, atômico —
# RF-24 inalterado), depois ENFILEIRA via Celery/Redis e retorna
# imediatamente. O envio real (rede) acontece no processo do
# `celery-worker` (src/tasks/notification_tasks.py) — a inferência nunca
# aguarda Telegram/e-mail responderem.
_critical_notifier = CriticalFailureNotificationService(
    adapter=_telegram_adapter,
    rate_limiter=_telegram_rate_limiter,
    dashboard_url=settings.dashboard_url,
    settings_service=_alert_settings_service,
    email_adapter=_email_adapter,
    enqueue_notification=enqueue_critical_failure_notification,
)
# RF-25 — rate limiter PRÓPRIO do botão "Testar Notificação", TTL curto
# (10s, só anti-duplo-clique) e totalmente separado do `_telegram_rate_limiter`
# de 15 min acima — um teste manual nunca deve consumir a janela que
# protegeria um alerta crítico real do mesmo equipamento logo em seguida.
_notification_test_rate_limiter = TelegramAlertRateLimiter(ttl_seconds=10)
_notification_test_service = NotificationTestService(
    telegram_adapter=_telegram_adapter,
    email_adapter=_email_adapter,
    settings_service=_alert_settings_service,
    test_rate_limiter=_notification_test_rate_limiter,
)


def get_alert_service() -> AlertService:
    """FastAPI Depends factory — wires the module-level ConnectionManager
    singleton (RF-14) and the critical-failure Telegram notifier (RF-24)."""
    return AlertService(_ws_manager, critical_notifier=_critical_notifier)


def get_alert_settings_service() -> AlertSettingsService:
    """FastAPI Depends factory — RF-25, usado por `routers/settings.py`."""
    return _alert_settings_service


def get_notification_test_service() -> NotificationTestService:
    """FastAPI Depends factory — RF-25, botão "Testar Notificação"."""
    return _notification_test_service
