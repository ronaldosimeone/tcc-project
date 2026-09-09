"""
Celery tasks — envio assíncrono de notificações críticas (RNF-50 / RNF-51).

Roda no processo do `celery-worker` (container próprio, MESMA imagem do
backend — ver docker-compose.yml). Os adapters (`TelegramNotificationAdapter`,
`EmailNotificationAdapter`) e o rate limiter NÃO atravessam o broker — não
são serializáveis (RNF-50 §6) — são reconstruídos aqui, no processo do
worker, a partir da MESMA configuração (env vars) que o processo web usa.

`dispatch_channels()` (a lógica real de fan-out Telegram/e-mail, falha
parcial e release do rate limit) é 100% reaproveitada de
`CriticalFailureNotificationService` — nenhuma regra de negócio duplicada
aqui, esta task é só o "onde" (processo do worker), não o "o quê".

Payload — só tipos JSON-serializáveis (RNF-50 §6/§9.2): strings, float,
bool, `None`. Nenhum objeto SQLAlchemy/FastAPI/adapter instanciado atravessa
o broker — ver `CriticalFailureNotificationService.notify_if_critical`, que
monta exatamente esse dict antes de enfileirar.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from src.core.celery_app import celery_app
from src.core.config import settings
from src.core.database import engine
from src.services.critical_failure_notification_service import (
    CriticalFailureNotificationService,
)
from src.services.email_notification_adapter import EmailNotificationAdapter
from src.services.telegram_alert_rate_limiter import TelegramAlertRateLimiter
from src.services.telegram_notification_adapter import TelegramNotificationAdapter

log = structlog.get_logger(__name__)

# Nome explícito e estável da task — usado por `.delay()`/`apply_async()` do
# lado do produtor e pelo `celery -A src.core.celery_app worker` do lado do
# consumidor. Não depende do caminho do módulo Python (robusto a refactors).
TASK_NAME = "notifications.send_critical_failure"


def _build_service_for_worker() -> CriticalFailureNotificationService:
    """Instâncias PRÓPRIAS do processo do worker — nunca as do processo web
    (não são serializáveis, não atravessam o broker). `enqueue_notification`
    fica `None` aqui de propósito: dentro do worker, o envio É a operação
    real — não há mais nada para enfileirar."""
    adapter = TelegramNotificationAdapter(
        bot_token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
        api_base_url=settings.telegram_api_base_url,
        timeout=settings.telegram_client_timeout_seconds,
    )
    email_adapter = EmailNotificationAdapter(
        api_key=settings.resend_api_key,
        from_email=settings.resend_from_email,
        api_base_url=settings.resend_api_base_url,
        timeout=settings.resend_client_timeout_seconds,
    )
    rate_limiter = TelegramAlertRateLimiter(
        ttl_seconds=settings.critical_failure_rate_limit_seconds
    )
    return CriticalFailureNotificationService(
        adapter=adapter,
        rate_limiter=rate_limiter,
        dashboard_url=settings.dashboard_url,
        email_adapter=email_adapter,
    )


async def _dispatch_and_cleanup(payload: dict[str, Any]) -> None:
    """
    Chama `dispatch_channels` e SEMPRE descarta o pool de conexões async do
    SQLAlchemy (`engine.dispose()`) ao final.

    Por quê: cada execução da task roda `asyncio.run(...)` (Celery é
    síncrono por padrão; nosso código de domínio é async) — cada chamada
    cria um NOVO event loop. Conexões asyncpg abertas num loop não podem ser
    reutilizadas por outro (`RuntimeError: Future attached to a different
    loop`). Como o `engine` do SQLAlchemy é um singleton de módulo
    (`core/database.py`, importado também pelo processo web), sem este
    dispose a SEGUNDA task executada por um worker de vida longa quebraria
    ao tentar reutilizar uma conexão do loop da PRIMEIRA. `dispose()` fecha
    as conexões existentes ainda dentro do loop atual (seguro) — a próxima
    task, em um loop novo, abre conexões novas sob demanda, normalmente.
    """
    try:
        service = _build_service_for_worker()
        await service.dispatch_channels(**payload)
    finally:
        await engine.dispose()


@celery_app.task(name=TASK_NAME, max_retries=0)
def send_critical_failure_notification_task(payload: dict[str, Any]) -> None:
    """
    Task Celery — executa o envio real (Telegram/e-mail) de uma falha
    crítica já decidida e rate-limited pelo processo web (ver
    `CriticalFailureNotificationService.notify_if_critical`).

    Sem retry automático (`max_retries=0`) — decisão deliberada (RNF-50
    §16): uma falha de rede aqui já libera o rate limit dentro de
    `dispatch_channels` (mesma regra do RF-24), permitindo que a PRÓXIMA
    predição crítica real tente de novo naturalmente — um retry cego da
    MESMA tentativa antiga adicionaria complexidade (contagem, backoff, at
    least once) sem benefício real hoje. Reavaliar se o volume de falhas
    transitórias justificar no futuro.
    """
    log.info(
        "notification_task_received",
        equipment_id=payload.get("equipment_id"),
        telegram_enabled=payload.get("telegram_enabled"),
        email_enabled=payload.get("email_enabled"),
    )
    try:
        asyncio.run(_dispatch_and_cleanup(payload))
    except Exception:
        # Nunca deixa uma exceção não tratada derrubar o worker — mesma
        # filosofia do RF-24 (falha de notificação nunca derruba nada).
        log.exception(
            "notification_task_unexpected_error",
            equipment_id=payload.get("equipment_id"),
        )
        return
    log.info("notification_task_completed", equipment_id=payload.get("equipment_id"))


def enqueue_critical_failure_notification(payload: dict[str, Any]) -> None:
    """
    Ponto único de enfileiramento — injetado em
    `CriticalFailureNotificationService` pela wiring real de produção
    (`services/alert_service.py`) como `enqueue_notification`. Isolado numa
    função própria (em vez de `alert_service.py` chamar `.delay()`
    diretamente) para o serviço de domínio nunca precisar importar Celery.
    """
    send_critical_failure_notification_task.delay(payload)
