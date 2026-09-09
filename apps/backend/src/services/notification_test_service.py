"""
NotificationTestService — RF-25 §10-18 (botão "Testar Notificação").

Permite ao operador validar a configuração de notificação (Telegram e/ou
e-mail) sem depender de uma predição real. Reaproveita os adapters já
existentes (`TelegramNotificationAdapter` do RF-24,
`EmailNotificationAdapter` desta mesma task) — nenhuma implementação
paralela de envio.

Testa exatamente os canais habilitados na configuração global
(`AlertSettingsService`) — nunca aciona `CriticalFailureNotificationService`/
`AlertService` (não cria alerta no histórico, não consome uma predição
real, não afeta o estado de nenhum equipamento).

Rate limit do teste (RF-25 §13) — deliberadamente SEPARADO do rate limit de
falha crítica (RF-24, 15 min por `equipment_id`): um clique manual não
representa uma falha real de equipamento, então não pode consumir a janela
de 15 min que protegeria um alerta crítico genuíno logo em seguida.
Reaproveita a MESMA classe `TelegramAlertRateLimiter` (mesmo padrão atômico,
mesma tabela `telegram_alert_locks`) — só com uma instância própria (TTL
curto, 10s — só evita duplo-clique/spam acidental do botão) e uma chave
reservada (`TEST_LOCK_KEY`) que nunca colide com um `equipment_id` real. Não
introduz nenhuma infraestrutura nova (sem Redis, sem tabela nova).
"""

from __future__ import annotations

import structlog

from src.core.exceptions import (
    EmailNotificationError,
    NoNotificationChannelEnabledError,
    NotificationTestFailedError,
    NotificationTestRateLimitedError,
    TelegramNotificationError,
)
from src.services.alert_settings_service import AlertSettingsService
from src.services.email_notification_adapter import EmailNotificationAdapter
from src.services.telegram_alert_rate_limiter import TelegramAlertRateLimiter
from src.services.telegram_notification_adapter import NotificationAdapter

log = structlog.get_logger(__name__)

# Chave reservada no mesmo namespace de `equipment_id` — os dois underscores
# nunca aparecem em um ID de equipamento real do projeto.
TEST_LOCK_KEY = "__settings_notification_test__"

FAILURE_MESSAGE = (
    "Não foi possível enviar a notificação. Verifique a configuração do Telegram."
)


class NotificationTestService:
    def __init__(
        self,
        telegram_adapter: NotificationAdapter,
        email_adapter: EmailNotificationAdapter,
        settings_service: AlertSettingsService,
        test_rate_limiter: TelegramAlertRateLimiter,
    ) -> None:
        self._telegram_adapter = telegram_adapter
        self._email_adapter = email_adapter
        self._settings_service = settings_service
        self._rate_limiter = test_rate_limiter

    async def send_test_notification(self) -> str:
        """
        Testa somente os canais habilitados na configuração atual — nunca
        um canal desligado. Retorna uma mensagem combinada (ex.:
        `"Telegram: enviado · E-mail: falhou"`) quando ao menos um canal
        enviou com sucesso; levanta:
        - `NotificationTestRateLimitedError` (429) se chamado de novo antes
          do TTL curto anti-spam expirar;
        - `NoNotificationChannelEnabledError` (400) se nenhum canal está
          habilitado (RF-25 §10) — nada para testar;
        - `NotificationTestFailedError` (502) se TODOS os canais
          habilitados falharem.
        Nunca vaza detalhes internos (token, API key, corpo de resposta) em
        qualquer uma dessas mensagens.
        """
        acquired = await self._rate_limiter.try_acquire(TEST_LOCK_KEY)
        if not acquired:
            log.info("notification_test_rate_limited")
            raise NotificationTestRateLimitedError()

        config = await self._settings_service.get_settings()
        if not config.telegram_enabled and not config.email_enabled:
            await self._rate_limiter.release(TEST_LOCK_KEY)
            raise NoNotificationChannelEnabledError()

        results: list[str] = []
        any_success = False

        if config.telegram_enabled:
            try:
                await self._telegram_adapter.send_test_notification()
                results.append("Telegram: enviado")
                any_success = True
            except TelegramNotificationError:
                results.append("Telegram: falhou")

        if config.email_enabled:
            if not config.alert_email:
                results.append("E-mail: endereço não configurado")
            else:
                try:
                    await self._email_adapter.send_test_email(to=config.alert_email)
                    results.append("E-mail: enviado")
                    any_success = True
                except EmailNotificationError:
                    results.append("E-mail: falhou")

        message = " · ".join(results)

        if not any_success:
            # Nenhum canal habilitado conseguiu enviar — libera o lock
            # imediatamente (permite corrigir a configuração e tentar de
            # novo sem esperar o TTL anti-spam, RF-25 §19).
            await self._rate_limiter.release(TEST_LOCK_KEY)
            log.warning("notification_test_all_channels_failed", detail=message)
            raise NotificationTestFailedError(message)

        log.info("notification_test_sent", detail=message)
        return message
