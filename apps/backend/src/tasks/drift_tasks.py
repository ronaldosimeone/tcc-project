"""
Celery task — análise diária de data drift (RF-27 / RNF-55).

Roda no processo do `celery-worker` (MESMA aplicação Celery de RNF-50/51 —
`src/core/celery_app.py`; nenhuma segunda app criada), disparada pelo
`celery-beat` uma vez por dia (ver `beat_schedule` em `celery_app.py`).

Mesma estratégia de `notification_tasks.py`: `asyncio.run(...)` (Celery é
síncrono por padrão, `DriftMonitor` é async) + `engine.dispose()` no
`finally` (evita `RuntimeError: Future attached to a different loop` num
worker de vida longa que processa mais de uma task ao longo do tempo — ver
o comentário completo em `notification_tasks.py::_dispatch_and_cleanup`).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import structlog

from src.core.celery_app import celery_app
from src.core.database import AsyncSessionFactory, engine
from src.services.drift_monitor import DriftMonitor

log = structlog.get_logger(__name__)

# Nome explícito e estável — mesmo padrão de `notification_tasks.TASK_NAME`.
TASK_NAME = "monitoring.daily_drift_analysis"


async def _run_and_persist() -> None:
    """
    Executa a análise completa e persiste o resultado — SEMPRE persiste,
    mesmo em `status="insufficient_data"`/`"error"` (RF-27 §"SEM DADOS
    INVENTADOS": o motivo fica registrado no histórico, nunca escondido).
    """
    now = datetime.now(timezone.utc)
    monitor = DriftMonitor()

    try:
        async with AsyncSessionFactory() as db:
            result = await monitor.run_daily_analysis(db, now=now)
            await monitor.persist_report(db, result, analysis_date=now.date())
        log.info(
            "drift_task_completed",
            status=result.status,
            psi=result.psi,
            drift_detected=result.drift_detected,
        )
    finally:
        # Mesmo motivo de `notification_tasks.py`: fecha as conexões
        # asyncpg abertas NESTE event loop antes que o worker reutilize o
        # `engine` singleton numa próxima task, em um loop novo.
        await engine.dispose()


@celery_app.task(name=TASK_NAME, max_retries=0)
def daily_drift_analysis_task() -> None:
    """
    Task Celery Beat — RF-27 §Fase 7. Não recebe payload (a janela current
    é sempre "últimas 24h a partir de agora", calculada dentro de
    `DriftMonitor.load_current_data`).

    Sem retry automático (`max_retries=0`) — mesma filosofia de RF-24/
    RNF-50: uma falha aqui (ex.: Postgres momentaneamente indisponível) já
    fica registrada como `status="error"` no histórico; a PRÓXIMA execução
    diária do Beat naturalmente tenta de novo. Um retry cego da MESMA
    tentativa antiga não traria benefício real para uma task diária.
    """
    log.info("drift_task_received")
    try:
        asyncio.run(_run_and_persist())
    except Exception:
        # Nunca deixa uma exceção não tratada derrubar o worker — mesma
        # filosofia de `notification_tasks.py`.
        log.exception("drift_task_unexpected_error")
