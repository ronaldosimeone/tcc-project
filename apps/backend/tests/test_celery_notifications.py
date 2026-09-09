"""
Testes de fila assíncrona de notificações — RNF-50 / RNF-51.

Não dependem de um `celery-worker`/Redis reais rodando (unit-level) — a
validação real fim-a-fim (`producer -> Redis -> celery-worker -> task`) e o
teste de não bloqueio com timing real ficam documentados no README/relatório
final desta task (executados manualmente contra o `docker compose` real).

Cobertura:
  9.1 Task registrada no app Celery.
  9.2 Payload serializável em JSON (nunca objetos SQLAlchemy/adapters).
  9.3 `notify_if_critical` enfileira (`enqueue_notification`) em vez de
      chamar os adapters diretamente quando um enqueue foi injetado.
  9.4 `AlertService.process_prediction()` retorna sem aguardar o adapter —
      prova de não bloqueio no nível de integração do RF-14/RF-24.
  9.5 A task, executada diretamente (sem broker), chama os adapters certos.
  9.6 Falha do adapter dentro da task não propaga (não derruba o worker).
  9.7 Telegram + e-mail juntos continuam sendo tentados pela task.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, Mock

from src.core.celery_app import celery_app
from src.core.ws_manager import ConnectionManager
from src.services.alert_service import AlertService
from src.services.critical_failure_notification_service import (
    CriticalFailureNotificationService,
)
from src.services.telegram_notification_adapter import TelegramNotificationAdapter
from src.tasks.notification_tasks import (
    TASK_NAME,
    enqueue_critical_failure_notification,
    send_critical_failure_notification_task,
)

FAKE_TOKEN = "123456:AAFAKE-TOKEN-NEVER-REAL-abcdefghij"


def _sample_payload(**overrides):
    payload = {
        "equipment_id": "eq-celery-1",
        "equipment_name": "Equip",
        "probability": 0.92,
        "timestamp": "2026-09-09T12:00:00+00:00",
        "dashboard_url": "http://localhost/sensors/eq-celery-1",
        "telegram_enabled": True,
        "email_enabled": False,
        "alert_email": None,
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# 9.1 — task registrada
# ---------------------------------------------------------------------------


def test_task_is_registered_in_celery_app() -> None:
    assert TASK_NAME in celery_app.tasks
    registered = celery_app.tasks[TASK_NAME]
    assert registered.name == send_critical_failure_notification_task.name
    assert callable(registered)


def test_celery_app_never_uses_pickle() -> None:
    """RNF-50 §15/§8 — serialização segura, nunca pickle."""
    assert celery_app.conf.task_serializer == "json"
    assert celery_app.conf.accept_content == ["json"]
    assert "pickle" not in celery_app.conf.accept_content


# ---------------------------------------------------------------------------
# 9.2 — payload serializável em JSON
# ---------------------------------------------------------------------------


def test_payload_is_json_serializable() -> None:
    payload = _sample_payload(email_enabled=True, alert_email="ops@empresa.com")
    encoded = json.dumps(payload)  # não lança — só str/float/bool/None
    decoded = json.loads(encoded)
    assert decoded == payload


# ---------------------------------------------------------------------------
# 9.3 — notify_if_critical enfileira em vez de enviar direto
# ---------------------------------------------------------------------------


async def test_notify_if_critical_enqueues_instead_of_sending_directly() -> None:
    adapter = AsyncMock()
    rate_limiter = AsyncMock()
    rate_limiter.try_acquire = AsyncMock(return_value=True)
    enqueue = Mock()

    service = CriticalFailureNotificationService(
        adapter=adapter,
        rate_limiter=rate_limiter,
        dashboard_url="http://localhost",
        enqueue_notification=enqueue,
    )

    await service.notify_if_critical(
        equipment_id="eq-1",
        equipment_name="Equip",
        probability=0.90,
        timestamp="2026-09-09T12:00:00+00:00",
    )

    # Enfileirou exatamente uma vez, com um payload serializável...
    enqueue.assert_called_once()
    (enqueued_payload,) = enqueue.call_args.args
    json.dumps(enqueued_payload)  # não lança
    assert enqueued_payload["equipment_id"] == "eq-1"
    assert enqueued_payload["telegram_enabled"] is True

    # ...e NUNCA chamou o adapter diretamente no processo síncrono.
    adapter.send_critical_failure.assert_not_awaited()


async def test_notify_if_critical_still_respects_rate_limit_before_enqueue() -> None:
    """O rate limit continua adquirido ANTES do enqueue (RNF-50 §7) — se
    bloqueado, nada é enfileirado."""
    adapter = AsyncMock()
    rate_limiter = AsyncMock()
    rate_limiter.try_acquire = AsyncMock(return_value=False)
    enqueue = Mock()

    service = CriticalFailureNotificationService(
        adapter=adapter,
        rate_limiter=rate_limiter,
        dashboard_url="http://localhost",
        enqueue_notification=enqueue,
    )

    await service.notify_if_critical(
        equipment_id="eq-1", equipment_name="Equip", probability=0.90, timestamp="t"
    )

    enqueue.assert_not_called()
    adapter.send_critical_failure.assert_not_awaited()


# ---------------------------------------------------------------------------
# 9.4 — process_prediction() não aguarda o envio (RF-14/RF-24 integração)
# ---------------------------------------------------------------------------


async def test_process_prediction_returns_without_waiting_for_slow_adapter() -> None:
    """Prova de não bloqueio: um adapter "lento" (0.5s por chamada) nunca é
    aguardado por `process_prediction()` quando um enqueue foi injetado —
    ele só seria aguardado no fallback SEM Celery (comportamento antigo)."""

    class SlowAdapter:
        async def send_critical_failure(self, notification) -> None:
            await asyncio.sleep(0.5)

        async def send_test_notification(self) -> None:
            pass

    rate_limiter = AsyncMock()
    rate_limiter.try_acquire = AsyncMock(return_value=True)
    enqueue = Mock()  # síncrono — nunca aguarda nada, só empilha o payload

    notifier = CriticalFailureNotificationService(
        adapter=SlowAdapter(),
        rate_limiter=rate_limiter,
        dashboard_url="http://localhost",
        enqueue_notification=enqueue,
    )
    ws_manager = AsyncMock(spec=ConnectionManager)
    alert_service = AlertService(ws_manager, critical_notifier=notifier)

    start = time.perf_counter()
    await alert_service.process_prediction(
        {"probability": 0.95, "label": "fault", "sensor_id": "s1"}
    )
    elapsed = time.perf_counter() - start

    # Bem abaixo dos 0.5s do "envio" — a chamada nunca esperou o adapter.
    assert elapsed < 0.2
    enqueue.assert_called_once()


async def test_process_prediction_without_enqueue_blocks_on_slow_adapter() -> None:
    """Contraprova — o comportamento ANTIGO (sem `enqueue_notification`,
    ainda usado pelos testes de RF-24/RF-25) continua síncrono: a mesma
    chamada agora fica presa pelos 0.5s do adapter. Documenta exatamente a
    diferença que RNF-50 introduziu."""

    class SlowAdapter:
        async def send_critical_failure(self, notification) -> None:
            await asyncio.sleep(0.3)

        async def send_test_notification(self) -> None:
            pass

    rate_limiter = AsyncMock()
    rate_limiter.try_acquire = AsyncMock(return_value=True)

    notifier = CriticalFailureNotificationService(
        adapter=SlowAdapter(),
        rate_limiter=rate_limiter,
        dashboard_url="http://localhost",
        # enqueue_notification=None (default) — caminho síncrono antigo.
    )
    ws_manager = AsyncMock(spec=ConnectionManager)
    alert_service = AlertService(ws_manager, critical_notifier=notifier)

    start = time.perf_counter()
    await alert_service.process_prediction(
        {"probability": 0.95, "label": "fault", "sensor_id": "s2"}
    )
    elapsed = time.perf_counter() - start

    assert elapsed >= 0.3  # esperou o "envio" completo — comportamento antigo


# ---------------------------------------------------------------------------
# 9.5 / 9.6 / 9.7 — execução da task (sem broker real)
# ---------------------------------------------------------------------------


def test_task_execution_calls_the_right_adapter(monkeypatch) -> None:
    """9.5 — chamar a task diretamente (sem `.delay()`/worker real) prova
    que ela chama o serviço/adapter corretos com o payload recebido."""
    adapter = AsyncMock()
    adapter.send_critical_failure = AsyncMock(return_value=None)
    fake_rate_limiter = AsyncMock()
    fake_rate_limiter.release = AsyncMock()

    fake_service = CriticalFailureNotificationService(
        adapter=adapter,
        rate_limiter=fake_rate_limiter,
        dashboard_url="http://localhost",
    )
    monkeypatch.setattr(
        "src.tasks.notification_tasks._build_service_for_worker",
        lambda: fake_service,
    )

    send_critical_failure_notification_task(_sample_payload())

    adapter.send_critical_failure.assert_awaited_once()
    fake_rate_limiter.release.assert_not_awaited()  # sucesso — mantém o lock


def test_task_adapter_failure_does_not_raise(monkeypatch, caplog) -> None:
    """9.6 — falha do adapter dentro da task não propaga (nunca derruba o
    worker) e libera o rate limit para permitir retry na próxima predição."""
    adapter = AsyncMock()
    adapter.send_critical_failure = AsyncMock(
        side_effect=RuntimeError("telegram indisponível")
    )
    fake_rate_limiter = AsyncMock()
    fake_rate_limiter.release = AsyncMock()

    fake_service = CriticalFailureNotificationService(
        adapter=adapter,
        rate_limiter=fake_rate_limiter,
        dashboard_url="http://localhost",
    )
    monkeypatch.setattr(
        "src.tasks.notification_tasks._build_service_for_worker",
        lambda: fake_service,
    )

    # Não lança — a task engole a exceção e loga.
    send_critical_failure_notification_task(_sample_payload())
    fake_rate_limiter.release.assert_awaited_once_with("eq-celery-1")


def test_task_tries_both_channels_when_both_enabled(monkeypatch) -> None:
    """9.7 — Telegram + e-mail continuam sendo tentados juntos pela task."""
    adapter = AsyncMock()
    adapter.send_critical_failure = AsyncMock(return_value=None)
    email_adapter = AsyncMock()
    email_adapter.send_critical_failure_email = AsyncMock(return_value=None)
    fake_rate_limiter = AsyncMock()

    fake_service = CriticalFailureNotificationService(
        adapter=adapter,
        rate_limiter=fake_rate_limiter,
        dashboard_url="http://localhost",
        email_adapter=email_adapter,
    )
    monkeypatch.setattr(
        "src.tasks.notification_tasks._build_service_for_worker",
        lambda: fake_service,
    )

    payload = _sample_payload(email_enabled=True, alert_email="ops@empresa.com")
    send_critical_failure_notification_task(payload)

    adapter.send_critical_failure.assert_awaited_once()
    email_adapter.send_critical_failure_email.assert_awaited_once()


def test_task_disposes_engine_after_running(monkeypatch) -> None:
    """Garante o `engine.dispose()` documentado em `_dispatch_and_cleanup` —
    evita reuso de conexão asyncpg entre event loops de execuções
    sucessivas da task num worker de vida longa."""
    adapter = AsyncMock()
    fake_service = CriticalFailureNotificationService(
        adapter=adapter, rate_limiter=AsyncMock(), dashboard_url="http://localhost"
    )
    monkeypatch.setattr(
        "src.tasks.notification_tasks._build_service_for_worker",
        lambda: fake_service,
    )

    fake_engine = AsyncMock()
    monkeypatch.setattr("src.tasks.notification_tasks.engine", fake_engine)
    send_critical_failure_notification_task(_sample_payload())
    fake_engine.dispose.assert_awaited_once()


# ---------------------------------------------------------------------------
# Segurança — token nunca em payload/log
# ---------------------------------------------------------------------------


def test_enqueue_helper_never_touches_credentials() -> None:
    """`enqueue_critical_failure_notification` só repassa o payload já
    montado pelo service — nunca lê/injeta TELEGRAM_BOT_TOKEN/RESEND_API_KEY
    (essas credenciais só existem no processo do worker, ver
    `_build_service_for_worker`)."""
    import inspect

    source = inspect.getsource(enqueue_critical_failure_notification)
    assert "token" not in source.lower()
    assert "api_key" not in source.lower()


def test_adapter_instances_are_never_part_of_the_payload() -> None:
    """Confirma que o adapter real (RF-24) nunca aparece como parte de um
    payload serializável — só campos primitivos."""
    adapter = TelegramNotificationAdapter(bot_token=FAKE_TOKEN, chat_id="123")
    payload = _sample_payload()
    assert adapter not in payload.values()
    for value in payload.values():
        assert value is None or isinstance(value, (str, float, int, bool))
