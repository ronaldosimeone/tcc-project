"""
Celery application — RNF-50 / RNF-51.

Broker: Redis (`CELERY_BROKER_URL`, ver `core/config.py`) — usado para
desacoplar o ENVIO real das notificações críticas (RF-24/RF-25, chamadas de
rede ao Telegram/Resend que podem levar segundos) do caminho síncrono da
inferência (`AlertService.process_prediction` -> RF-14). Ver
`src/services/critical_failure_notification_service.py` (onde o
`.delay()`/`apply_async()` acontece) e `src/tasks/notification_tasks.py`
(a task em si, que roda no processo do `celery-worker` — container próprio,
ver docker-compose.yml).

Auditoria antes desta task: nenhum Celery/Redis existia no projeto. O rate
limiter do RF-24 continua em Postgres — Redis aqui é SÓ o broker do Celery,
nunca um substituto do banco.

Decisões de configuração:
- `task_serializer="json"` / `accept_content=["json"]` — NUNCA pickle
  (RNF-50 §15/§8: pickle permite execução arbitrária de código a partir de
  uma mensagem da fila; JSON é seguro e o payload das tasks já é só
  strings/números/booleanos — ver `notification_tasks.py`).
- `task_ignore_result=True`, sem result backend — nenhum código deste
  projeto precisa ler de volta o resultado de uma task (fire-and-forget por
  design: a inferência não espera nem consulta o envio). Adicionar um
  result backend (Redis ou outro) sem essa necessidade seria complexidade
  sem benefício real (RNF-50 §4).
- Import explícito do módulo de tasks no final deste arquivo — mesmo estilo
  do projeto em `main.py` (routers registrados um a um, sem autodiscovery
  "mágico" de pacotes) — garante que a task fica registrada tanto no
  processo do worker (`celery -A src.core.celery_app worker`) quanto no
  processo web (que só precisa do nome da task para enfileirar).

RF-27 / RNF-55 — Celery Beat (análise diária de data drift):
Auditoria antes desta task: nenhum Celery Beat existia no projeto (só
`celery-worker`, consumidor). Reaproveita esta MESMA aplicação Celery — o
`beat_schedule` abaixo só REGISTRA quando `monitoring.daily_drift_analysis`
(`src/tasks/drift_tasks.py`) deve ser enfileirada; quem efetivamente
EXECUTA a task continua sendo o `celery-worker` (container próprio,
`docker-compose.yml`), nunca o processo `celery beat` em si (que só publica
mensagens na fila, no horário configurado). `timezone="UTC"`/`enable_utc`
já definidos acima (RNF-50) — reaproveitados, não redefinidos.
"""

from __future__ import annotations

from celery import Celery
from celery.schedules import crontab

from src.core.config import settings

celery_app: Celery = Celery("predictiq", broker=settings.celery_broker_url)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_ignore_result=True,
    timezone="UTC",
    enable_utc=True,
    # Explícito (Celery 6.0 muda o default) — reconecta ao Redis se ele
    # ainda não estiver pronto no momento em que o worker/processo web sobe
    # (docker-compose não garante que o healthcheck do Redis termine antes
    # do primeiro import deste módulo dentro do container).
    broker_connection_retry_on_startup=True,
    # RF-27 §Fase 7 — uma execução por dia, 03:00 UTC (horário de baixo
    # tráfego, fora do expediente — escolha arbitrária apenas no SENTIDO de
    # "qualquer horário ocioso serve"; a REGRA "uma vez por dia" é a exigida
    # pelo requisito, não o minuto exato). `crontab(hour=3, minute=0)` roda
    # todo dia às 03:00 no `timezone` configurado acima (UTC).
    beat_schedule={
        "daily-drift-analysis": {
            "task": "monitoring.daily_drift_analysis",
            "schedule": crontab(hour=3, minute=0),
        },
    },
)

# Import explícito (não autodiscovery) — registra
# `notifications.send_critical_failure` e `monitoring.daily_drift_analysis`
# neste app Celery. Feito no final do módulo porque os módulos de task
# importam `celery_app` deste mesmo arquivo (decorator `@celery_app.task`).
from src.tasks import drift_tasks  # noqa: E402,F401
from src.tasks import notification_tasks  # noqa: E402,F401
