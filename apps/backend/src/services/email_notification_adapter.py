"""
EmailNotificationAdapter — RF-25 / RNF-49.

Encapsula TODA a comunicação com a API do Resend — nenhuma chamada HTTP de
e-mail acontece fora deste módulo. Mesmo padrão de client já estabelecido no
projeto (`TelegramNotificationAdapter`, RF-24): HTTP puro via `httpx`, sem
SDK — a API REST do Resend é simples o bastante (`POST /emails` com
`Authorization: Bearer <API_KEY>`) que adicionar o pacote `resend` como
dependência só para isso não se justifica.

Auditoria explícita antes desta task: NENHUMA integração de e-mail existia
no projeto (nem Resend, nem SMTP, nem qualquer outro provedor) — este é o
primeiro canal de e-mail do PredictIQ, não uma reintegração.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from src.core.exceptions import EmailNotificationError

log = structlog.get_logger(__name__)


def build_critical_failure_email_text(
    *,
    equipment_name: str,
    probability: float,
    timestamp: str,
    dashboard_url: str,
) -> str:
    """Texto simples — mesmo raciocínio do Telegram (RF-24): sem HTML,
    elimina qualquer superfície de injeção de markup no corpo do e-mail."""
    return (
        "FALHA CRÍTICA DETECTADA\n\n"
        f"Equipamento: {equipment_name}\n"
        f"Probabilidade de falha: {probability:.0%}\n"
        f"Timestamp: {timestamp}\n\n"
        "Uma falha crítica foi detectada pelo modelo preditivo.\n\n"
        f"Acesse o Dashboard: {dashboard_url}"
    )


_TEST_EMAIL_SUBJECT = "PredictIQ — Teste de Notificação"
_TEST_EMAIL_TEXT = (
    "Este é um teste de notificação do PredictIQ.\n\n"
    "A configuração de alertas está funcionando corretamente.\n\n"
    "Este e-mail não representa uma falha real de equipamento."
)


class EmailNotificationAdapter:
    """Adapter concreto — `POST {api_base_url}/emails` (Resend)."""

    def __init__(
        self,
        api_key: str | None,
        from_email: str,
        api_base_url: str = "https://api.resend.com",
        timeout: float = 10.0,
    ) -> None:
        self._api_key = api_key
        self._from_email = from_email
        self._api_base_url = api_base_url.rstrip("/")
        self._timeout = timeout

    async def send_critical_failure_email(
        self,
        *,
        to: str,
        equipment_name: str,
        probability: float,
        timestamp: str,
        dashboard_url: str,
    ) -> None:
        text = build_critical_failure_email_text(
            equipment_name=equipment_name,
            probability=probability,
            timestamp=timestamp,
            dashboard_url=dashboard_url,
        )
        await self._send(
            to=to,
            subject="PredictIQ — Falha crítica detectada",
            text=text,
            log_context={"purpose": "critical_failure"},
        )

    async def send_test_email(self, *, to: str) -> None:
        """RF-25 §11 — mensagem claramente marcada como teste, nunca o
        texto de falha crítica real."""
        await self._send(
            to=to,
            subject=_TEST_EMAIL_SUBJECT,
            text=_TEST_EMAIL_TEXT,
            log_context={"purpose": "test"},
        )

    async def _send(
        self, *, to: str, subject: str, text: str, log_context: dict[str, Any]
    ) -> None:
        """
        Levanta `EmailNotificationError` para qualquer falha — configuração
        ausente, timeout, HTTP não-2xx, JSON inválido. Nunca inclui a API
        key nem o corpo de erro do Resend em log (pode ecoar dados da
        requisição de volta).
        """
        if not self._api_key or not to:
            log.warning("resend_config_missing", **log_context)
            raise EmailNotificationError(
                "Configuração de e-mail ausente (RESEND_API_KEY) ou destinatário não informado."
            )

        url = f"{self._api_base_url}/emails"
        # A API key NUNCA é logada — só usada no header desta única chamada.
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {
            "from": self._from_email,
            "to": [to],
            "subject": subject,
            "text": text,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            log.warning("resend_timeout", **log_context)
            raise EmailNotificationError(
                "Timeout ao enviar e-mail via Resend."
            ) from exc
        except httpx.HTTPError as exc:
            log.warning("resend_connection_error", **log_context)
            raise EmailNotificationError(
                "Não foi possível conectar à API do Resend."
            ) from exc

        if response.status_code not in (200, 201):
            # Só o status_code é logado — nunca o corpo (pode ecoar o
            # destinatário ou detalhes de configuração de volta).
            log.warning(
                "resend_http_error", status_code=response.status_code, **log_context
            )
            raise EmailNotificationError(
                f"Resend retornou HTTP {response.status_code}."
            )

        try:
            response.json()
        except ValueError as exc:
            log.warning("resend_invalid_json", **log_context)
            raise EmailNotificationError(
                "Resposta do Resend não é JSON válido."
            ) from exc
