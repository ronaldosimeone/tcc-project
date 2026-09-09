"""
TelegramNotificationAdapter — RF-24 / RNF-48.

Encapsula TODA a comunicação com a Telegram Bot API — nenhuma chamada HTTP
ao Telegram acontece fora deste módulo. Segue o mesmo padrão de client já
estabelecido no projeto (`OllamaClient`, `MCPSearchClient`, RF-22): HTTP
puro via `httpx` (já dependência), levanta uma exceção de domínio própria
em vez de devolver bool/None em silêncio.

`NotificationAdapter` (Protocol) existe só para deixar explícito o ponto de
extensão para futuros canais (email, WhatsApp, push) sem acoplar
`CriticalFailureNotificationService` a uma implementação concreta — nenhum
outro adapter é implementado nesta task (escopo estrito: Telegram).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import structlog

from src.core.exceptions import TelegramNotificationError

log = structlog.get_logger(__name__)


@dataclass
class CriticalFailureNotification:
    """Dados de uma notificação de falha crítica — nunca contém segredos."""

    equipment_id: str
    equipment_name: str
    probability: float
    timestamp: str
    dashboard_url: str


class NotificationAdapter(Protocol):
    """Interface mínima para um canal de notificação de falha crítica —
    ponto de extensão futuro (email, WhatsApp, push). Só Telegram é
    implementado nesta task."""

    async def send_critical_failure(
        self, notification: CriticalFailureNotification
    ) -> None: ...


def build_critical_failure_message(notification: CriticalFailureNotification) -> str:
    """
    Mensagem em texto simples — sem `parse_mode` (RF-24 §18): o conteúdo do
    equipamento/timestamp não é sanitizado para MarkdownV2/HTML do Telegram,
    então usar `parse_mode` exigiria escapar caracteres especiais para
    evitar erro de parsing (ou pior, injeção de formatação). Texto simples
    elimina essa superfície de risco por completo.
    """
    return (
        "🚨 FALHA CRÍTICA DETECTADA\n\n"
        f"Equipamento: {notification.equipment_name}\n"
        f"Probabilidade de falha: {notification.probability:.0%}\n"
        f"Timestamp: {notification.timestamp}\n\n"
        "Uma falha crítica foi detectada pelo modelo preditivo.\n\n"
        f"Acesse o Dashboard: {notification.dashboard_url}"
    )


class TelegramNotificationAdapter:
    """Adapter concreto — `POST {api_base_url}/bot{token}/sendMessage`."""

    def __init__(
        self,
        bot_token: str | None,
        chat_id: str | None,
        api_base_url: str = "https://api.telegram.org",
        timeout: float = 10.0,
    ) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._api_base_url = api_base_url.rstrip("/")
        self._timeout = timeout

    async def send_critical_failure(
        self, notification: CriticalFailureNotification
    ) -> None:
        """
        Levanta `TelegramNotificationError` (RF-24 §16) para qualquer falha
        — configuração ausente, timeout, HTTP não-200, JSON inválido, ou
        `ok: false` na resposta do Telegram. Nunca inclui o token/URL
        completa na mensagem da exceção nem em log (RF-24 §2/§15).
        """
        if not self._bot_token or not self._chat_id:
            log.warning("telegram_config_missing")
            raise TelegramNotificationError(
                "Configuração do Telegram ausente (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID)."
            )

        message = build_critical_failure_message(notification)
        # URL contém o token no path (convenção da Bot API) — NUNCA logada.
        url = f"{self._api_base_url}/bot{self._bot_token}/sendMessage"
        payload = {"chat_id": self._chat_id, "text": message}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            log.warning("telegram_timeout", equipment_id=notification.equipment_id)
            raise TelegramNotificationError(
                "Timeout ao enviar notificação Telegram."
            ) from exc
        except httpx.HTTPError as exc:
            log.warning(
                "telegram_connection_error", equipment_id=notification.equipment_id
            )
            raise TelegramNotificationError(
                "Não foi possível conectar à API do Telegram."
            ) from exc

        if response.status_code != 200:
            # Só o status_code é logado — nunca o corpo (pode ecoar o chat_id
            # ou detalhes de configuração de volta em alguns erros 4xx).
            log.warning(
                "telegram_http_error",
                equipment_id=notification.equipment_id,
                status_code=response.status_code,
            )
            raise TelegramNotificationError(
                f"Telegram retornou HTTP {response.status_code}."
            )

        try:
            data: dict[str, Any] = response.json()
        except ValueError as exc:
            log.warning("telegram_invalid_json", equipment_id=notification.equipment_id)
            raise TelegramNotificationError(
                "Resposta do Telegram não é JSON válido."
            ) from exc

        if not data.get("ok"):
            log.warning(
                "telegram_reported_failure", equipment_id=notification.equipment_id
            )
            raise TelegramNotificationError(
                "Telegram reportou falha no envio da mensagem."
            )
