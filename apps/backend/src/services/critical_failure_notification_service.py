"""
CriticalFailureNotificationService — RF-24 / RNF-48.

Orquestra: threshold (> 0.85, estrito) -> rate limit por equipamento
(15 min, TelegramAlertRateLimiter) -> NotificationAdapter (Telegram).

    prediction
        v
    probability > 0.85 ?  (CRITICAL_FAILURE_THRESHOLD — constante própria,
        |                   NÃO reaproveita MAINTENANCE_SUGGESTION_THRESHOLD
        v                   do RF-22 nem ALERT_PROBABILITY_THRESHOLD do RF-14)
    try_acquire(equipment_id)  -- atômico, ver telegram_alert_rate_limiter.py
        v
    rate limited? -> loga e retorna (Telegram NUNCA chamado)
        v
    adapter.send_critical_failure(...)
        v
    falhou? -> release(equipment_id) (permite retry) + loga, NUNCA propaga
        v
    sucesso -> loga

Nunca levanta exceção — uma falha de notificação externa não pode derrubar
o pipeline de inferência que já calculou a predição (RF-24 §16). Chamado a
partir de `AlertService.process_prediction` (RF-14), não do pipeline de ML
diretamente — reaproveita o mesmo ponto de integração já usado pelo alerta
WebSocket, sem uma segunda inferência nem lógica duplicada.
"""

from __future__ import annotations

import structlog

from src.core.exceptions import TelegramNotificationError
from src.services.telegram_alert_rate_limiter import TelegramAlertRateLimiter
from src.services.telegram_notification_adapter import (
    CriticalFailureNotification,
    NotificationAdapter,
)

log = structlog.get_logger(__name__)

# RF-24 — regra de negócio fixa: estritamente > 0.85 (nunca >=). Constante
# própria — ver comparação com RF-14 (0.70) e RF-22 (0.70) no docstring do
# módulo: mesma faixa de valores por coincidência de domínio, três decisões
# de negócio independentes.
CRITICAL_FAILURE_THRESHOLD: float = 0.85


class CriticalFailureNotificationService:
    def __init__(
        self,
        adapter: NotificationAdapter,
        rate_limiter: TelegramAlertRateLimiter,
        dashboard_url: str,
    ) -> None:
        self._adapter = adapter
        self._rate_limiter = rate_limiter
        # Sem barra final — a rota é sempre concatenada com "/" explícito
        # (ver _build_dashboard_url), nunca duplicando ou omitindo a barra.
        self._dashboard_url = dashboard_url.rstrip("/")

    async def notify_if_critical(
        self,
        *,
        equipment_id: str,
        equipment_name: str,
        probability: float,
        timestamp: str,
    ) -> None:
        """Nunca lança — qualquer falha (rate limit, Telegram, ou erro
        inesperado) é logada e engolida aqui. Chamador nunca precisa de
        try/except."""
        if probability <= CRITICAL_FAILURE_THRESHOLD:
            return

        try:
            acquired = await self._rate_limiter.try_acquire(equipment_id)
        except Exception:
            # Falha no próprio rate limiter (ex.: DB indisponível) não pode
            # derrubar a inferência — loga e desiste desta notificação.
            log.exception(
                "critical_failure_rate_limiter_error", equipment_id=equipment_id
            )
            return

        if not acquired:
            log.info(
                "critical_failure_notification_rate_limited",
                equipment_id=equipment_id,
                probability=probability,
            )
            return

        notification = CriticalFailureNotification(
            equipment_id=equipment_id,
            equipment_name=equipment_name,
            probability=probability,
            timestamp=timestamp,
            dashboard_url=self._build_dashboard_url(equipment_id),
        )

        try:
            await self._adapter.send_critical_failure(notification)
        except TelegramNotificationError as exc:
            log.warning(
                "critical_failure_notification_failed",
                equipment_id=equipment_id,
                error=exc.detail,
            )
            # Falha não pode consumir a janela — permite retry na próxima
            # predição crítica (RF-24 §5), em vez de bloquear por 15 min
            # por causa de uma falha transitória do Telegram.
            await self._rate_limiter.release(equipment_id)
            return
        except Exception:
            log.exception(
                "critical_failure_notification_unexpected_error",
                equipment_id=equipment_id,
            )
            await self._rate_limiter.release(equipment_id)
            return

        log.info(
            "critical_failure_notification_sent",
            equipment_id=equipment_id,
            probability=probability,
        )

    def _build_dashboard_url(self, equipment_id: str) -> str:
        """
        URL absoluta construída SÓ a partir de `DASHBOARD_URL` (configuração
        do operador) + rota já existente no frontend (`/sensors/[id]`, ver
        `apps/frontend/app/sensors/[id]`) — nunca a partir de dado vindo da
        predição/LLM (RF-24 §4, evita open redirect). `equipment_id` só pode
        ser um dos valores internos que este serviço já recebeu do pipeline,
        nunca input de rede externo.
        """
        return f"{self._dashboard_url}/sensors/{equipment_id}"
