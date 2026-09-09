"""
Testes de notificação crítica via Telegram (RF-24 / RNF-48).

Três camadas testadas separadamente:
  1. `TelegramNotificationAdapter` — HTTP mockado (`httpx.AsyncClient`
     nunca fala com a internet real), cobre sucesso/timeout/HTTP 4xx-5xx/
     JSON inválido/config ausente, e que o token nunca aparece em erro/log.
  2. `TelegramAlertRateLimiter` — banco REAL (SQLite em memória, `StaticPool`
     para garantir que todas as conexões da suíte de concorrência
     compartilhem o mesmo banco), incluindo o teste de concorrência real
     via `asyncio.gather` e o teste de TTL com clock injetado (sem
     `sleep` de 15 minutos).
  3. `CriticalFailureNotificationService` — orquestra as duas camadas
     acima; aqui SIM tudo é mockado (rate limiter e adapter), pra isolar
     a lógica de threshold/orquestração.

Validação de infraestrutura real (Postgres, não SQLite) e o teste real do
Telegram (se credenciais existirem) ficam fora deste arquivo — ver README
"RF-24 — Validação real" e o relatório final desta task.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from src.core.database import Base
from src.core.exceptions import TelegramNotificationError

# TelegramAlertLock é registrado em Base.metadata como efeito colateral do
# import de telegram_alert_rate_limiter abaixo — não precisa de import
# próprio aqui.
from src.services.critical_failure_notification_service import (
    CRITICAL_FAILURE_THRESHOLD,
    CriticalFailureNotificationService,
)
from src.services.telegram_alert_rate_limiter import TelegramAlertRateLimiter
from src.services.telegram_notification_adapter import (
    CriticalFailureNotification,
    TelegramNotificationAdapter,
    build_critical_failure_message,
)

FAKE_TOKEN = "123456:AAFAKE-TOKEN-NEVER-REAL-abcdefghij"  # nunca um token real


def _notification(**overrides: Any) -> CriticalFailureNotification:
    kwargs: dict[str, Any] = dict(
        equipment_id="APU-Trem-042",
        equipment_name="Compressor de Ar Industrial",
        probability=0.92,
        timestamp="2026-09-08T12:00:00+00:00",
        dashboard_url="http://localhost/sensors/APU-Trem-042",
    )
    kwargs.update(overrides)
    return CriticalFailureNotification(**kwargs)


def _mock_httpx_response(
    status_code: int, json_body: dict | None = None
) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=json_body if json_body is not None else {},
        request=httpx.Request("POST", "https://api.telegram.org/fake"),
    )


# ---------------------------------------------------------------------------
# 1. TelegramNotificationAdapter — HTTP mockado
# ---------------------------------------------------------------------------


async def test_adapter_success_calls_send_message_endpoint(monkeypatch) -> None:
    mock_post = AsyncMock(return_value=_mock_httpx_response(200, {"ok": True}))
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    adapter = TelegramNotificationAdapter(bot_token=FAKE_TOKEN, chat_id="12345")
    await adapter.send_critical_failure(_notification())  # não deve lançar

    mock_post.assert_awaited_once()
    assert mock_post.await_args is not None
    url = mock_post.await_args.args[0]
    assert url == f"https://api.telegram.org/bot{FAKE_TOKEN}/sendMessage"


async def test_adapter_missing_token_raises_clear_error() -> None:
    adapter = TelegramNotificationAdapter(bot_token=None, chat_id="12345")
    with pytest.raises(TelegramNotificationError, match="ausente"):
        await adapter.send_critical_failure(_notification())


async def test_adapter_missing_chat_id_raises_clear_error() -> None:
    adapter = TelegramNotificationAdapter(bot_token=FAKE_TOKEN, chat_id=None)
    with pytest.raises(TelegramNotificationError, match="ausente"):
        await adapter.send_critical_failure(_notification())


async def test_adapter_timeout_raises_notification_error(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx.AsyncClient,
        "post",
        AsyncMock(side_effect=httpx.TimeoutException("timed out")),
    )
    adapter = TelegramNotificationAdapter(bot_token=FAKE_TOKEN, chat_id="12345")
    with pytest.raises(TelegramNotificationError, match="[Tt]imeout"):
        await adapter.send_critical_failure(_notification())


async def test_adapter_http_500_raises_notification_error(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx.AsyncClient, "post", AsyncMock(return_value=_mock_httpx_response(500))
    )
    adapter = TelegramNotificationAdapter(bot_token=FAKE_TOKEN, chat_id="12345")
    with pytest.raises(TelegramNotificationError, match="500"):
        await adapter.send_critical_failure(_notification())


async def test_adapter_http_4xx_raises_notification_error(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx.AsyncClient, "post", AsyncMock(return_value=_mock_httpx_response(403))
    )
    adapter = TelegramNotificationAdapter(bot_token=FAKE_TOKEN, chat_id="12345")
    with pytest.raises(TelegramNotificationError):
        await adapter.send_critical_failure(_notification())


async def test_adapter_telegram_unavailable_raises_notification_error(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        httpx.AsyncClient,
        "post",
        AsyncMock(side_effect=httpx.ConnectError("connection refused")),
    )
    adapter = TelegramNotificationAdapter(bot_token=FAKE_TOKEN, chat_id="12345")
    with pytest.raises(TelegramNotificationError):
        await adapter.send_critical_failure(_notification())


async def test_adapter_invalid_json_response_raises_notification_error(
    monkeypatch,
) -> None:
    bad_response = httpx.Response(
        status_code=200,
        content=b"not-json",
        request=httpx.Request("POST", "https://api.telegram.org/fake"),
    )
    monkeypatch.setattr(httpx.AsyncClient, "post", AsyncMock(return_value=bad_response))
    adapter = TelegramNotificationAdapter(bot_token=FAKE_TOKEN, chat_id="12345")
    with pytest.raises(TelegramNotificationError):
        await adapter.send_critical_failure(_notification())


async def test_adapter_ok_false_raises_notification_error(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx.AsyncClient,
        "post",
        AsyncMock(
            return_value=_mock_httpx_response(
                200, {"ok": False, "description": "blocked"}
            )
        ),
    )
    adapter = TelegramNotificationAdapter(bot_token=FAKE_TOKEN, chat_id="12345")
    with pytest.raises(TelegramNotificationError):
        await adapter.send_critical_failure(_notification())


# ---------------------------------------------------------------------------
# Q) Token nunca aparece em erro/log
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "failure_mode",
    ["missing_config", "timeout", "http_500", "invalid_json", "ok_false"],
)
async def test_token_never_appears_in_exception_message(
    monkeypatch, failure_mode: str
) -> None:
    if failure_mode == "missing_config":
        adapter = TelegramNotificationAdapter(bot_token=None, chat_id="12345")
    else:
        adapter = TelegramNotificationAdapter(bot_token=FAKE_TOKEN, chat_id="12345")
        if failure_mode == "timeout":
            monkeypatch.setattr(
                httpx.AsyncClient,
                "post",
                AsyncMock(side_effect=httpx.TimeoutException("x")),
            )
        elif failure_mode == "http_500":
            monkeypatch.setattr(
                httpx.AsyncClient,
                "post",
                AsyncMock(return_value=_mock_httpx_response(500)),
            )
        elif failure_mode == "invalid_json":
            bad = httpx.Response(
                200, content=b"nope", request=httpx.Request("POST", "https://x/fake")
            )
            monkeypatch.setattr(httpx.AsyncClient, "post", AsyncMock(return_value=bad))
        elif failure_mode == "ok_false":
            monkeypatch.setattr(
                httpx.AsyncClient,
                "post",
                AsyncMock(return_value=_mock_httpx_response(200, {"ok": False})),
            )

    with pytest.raises(TelegramNotificationError) as excinfo:
        await adapter.send_critical_failure(_notification())

    assert FAKE_TOKEN not in str(excinfo.value)
    assert FAKE_TOKEN not in excinfo.value.detail


def test_message_builder_never_includes_token() -> None:
    # build_critical_failure_message nem recebe o token — garantido pela
    # assinatura (CriticalFailureNotification não tem esse campo), mas o
    # teste documenta a garantia explicitamente.
    message = build_critical_failure_message(_notification())
    assert "token" not in message.lower()
    assert FAKE_TOKEN not in message


# ---------------------------------------------------------------------------
# P) Dashboard URL na mensagem
# ---------------------------------------------------------------------------


def test_message_contains_dashboard_url_and_key_fields() -> None:
    message = build_critical_failure_message(
        _notification(
            equipment_name="Bomba Centrífuga 01",
            probability=0.92,
            dashboard_url="http://localhost/sensors/bomba-01",
        )
    )
    assert "http://localhost/sensors/bomba-01" in message
    assert "Bomba Centrífuga 01" in message
    assert "92%" in message
    assert "FALHA CRÍTICA" in message


# ---------------------------------------------------------------------------
# 2. TelegramAlertRateLimiter — banco real (SQLite em memória, StaticPool)
# ---------------------------------------------------------------------------


@pytest.fixture()
async def rate_limiter_session_factory():
    """
    SQLite em memória com `StaticPool` — TODAS as conexões da engine
    compartilham a MESMA base em memória (sem isso, cada checkout do pool
    abriria um `:memory:` diferente, quebrando o teste de concorrência que
    depende de duas conexões enxergarem a mesma tabela).
    """
    engine: AsyncEngine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )
    yield session_factory

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


async def test_first_alert_for_equipment_is_acquired(
    rate_limiter_session_factory,
) -> None:
    """E) primeiro alerta do equipamento — enviado (aquisição bem-sucedida)."""
    rl = TelegramAlertRateLimiter(session_factory=rate_limiter_session_factory)
    assert await rl.try_acquire("eq-1") is True


async def test_second_alert_within_window_is_blocked(
    rate_limiter_session_factory,
) -> None:
    """F) segundo alerta do mesmo equipamento dentro de 15 min — bloqueado."""
    rl = TelegramAlertRateLimiter(session_factory=rate_limiter_session_factory)
    assert await rl.try_acquire("eq-1") is True
    assert await rl.try_acquire("eq-1") is False


async def test_different_equipment_is_allowed(rate_limiter_session_factory) -> None:
    """G) equipamento diferente — permitido mesmo com eq-1 já bloqueado."""
    rl = TelegramAlertRateLimiter(session_factory=rate_limiter_session_factory)
    assert await rl.try_acquire("eq-1") is True
    assert await rl.try_acquire("eq-2") is True


async def test_release_allows_immediate_retry(rate_limiter_session_factory) -> None:
    """J/K/L) liberar a chave (falha de envio) permite nova tentativa imediata."""
    rl = TelegramAlertRateLimiter(session_factory=rate_limiter_session_factory)
    assert await rl.try_acquire("eq-1") is True
    assert await rl.try_acquire("eq-1") is False
    await rl.release("eq-1")
    assert await rl.try_acquire("eq-1") is True


async def test_concurrent_acquire_same_equipment_only_one_wins(
    rate_limiter_session_factory,
) -> None:
    """
    M) duas tentativas concorrentes do mesmo equipamento — no máximo uma
    adquire. `asyncio.gather` dispara as duas chamadas "ao mesmo tempo"
    (mesmo eventloop, mas ambas fazem I/O real contra o banco antes de
    resolver — suficiente para exercitar a serialização a nível de linha
    do UPSERT, não uma simples corrida de instruções Python síncronas).
    """
    rl = TelegramAlertRateLimiter(session_factory=rate_limiter_session_factory)
    results = await asyncio.gather(
        rl.try_acquire("eq-concurrent"),
        rl.try_acquire("eq-concurrent"),
    )
    assert sorted(results) == [False, True]  # exatamente um True


# ---------------------------------------------------------------------------
# 12) Teste de TTL — clock injetado, sem esperar 15 minutos de verdade
# ---------------------------------------------------------------------------


async def test_ttl_boundaries_with_injected_clock(rate_limiter_session_factory) -> None:
    """t=0 permitido; t=5min e t=14m59s bloqueados; t=15m permitido de novo."""
    current_time = {"now": datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)}

    def clock() -> datetime:
        return current_time["now"]

    rl = TelegramAlertRateLimiter(
        ttl_seconds=900, session_factory=rate_limiter_session_factory, clock=clock
    )

    # t = 0
    assert await rl.try_acquire("eq-ttl") is True

    # t = 5 min
    current_time["now"] += timedelta(minutes=5)
    assert await rl.try_acquire("eq-ttl") is False

    # t = 14m59s
    current_time["now"] = datetime(2026, 1, 1, 12, 14, 59, tzinfo=timezone.utc)
    assert await rl.try_acquire("eq-ttl") is False

    # t = 15 min exatos — janela expirou, permitido de novo
    current_time["now"] = datetime(2026, 1, 1, 12, 15, 0, tzinfo=timezone.utc)
    assert await rl.try_acquire("eq-ttl") is True


# ---------------------------------------------------------------------------
# 3. CriticalFailureNotificationService — threshold + orquestração
# ---------------------------------------------------------------------------


def _make_service(
    rate_limit_acquired: bool = True, adapter_error: Exception | None = None
):
    adapter = AsyncMock()
    if adapter_error is not None:
        adapter.send_critical_failure = AsyncMock(side_effect=adapter_error)
    else:
        adapter.send_critical_failure = AsyncMock(return_value=None)

    rate_limiter = AsyncMock()
    rate_limiter.try_acquire = AsyncMock(return_value=rate_limit_acquired)
    rate_limiter.release = AsyncMock(return_value=None)

    service = CriticalFailureNotificationService(
        adapter=adapter, rate_limiter=rate_limiter, dashboard_url="http://localhost"
    )
    return service, adapter, rate_limiter


def test_critical_failure_threshold_is_zero_point_eight_five() -> None:
    assert CRITICAL_FAILURE_THRESHOLD == 0.85


@pytest.mark.parametrize("probability", [0.0, 0.5, 0.84, 0.85])
async def test_probability_at_or_below_threshold_never_calls_telegram(
    probability: float,
) -> None:
    """A) 0.84 e B) 0.85 — Telegram não chamado."""
    service, adapter, rate_limiter = _make_service()

    await service.notify_if_critical(
        equipment_id="eq-1",
        equipment_name="Eq 1",
        probability=probability,
        timestamp="t",
    )

    adapter.send_critical_failure.assert_not_called()
    rate_limiter.try_acquire.assert_not_called()


@pytest.mark.parametrize("probability", [0.8501, 0.90, 0.9999])
async def test_probability_above_threshold_calls_telegram(probability: float) -> None:
    """C) 0.8501 e D) 0.90 — Telegram chamado."""
    service, adapter, rate_limiter = _make_service()

    await service.notify_if_critical(
        equipment_id="eq-1",
        equipment_name="Eq 1",
        probability=probability,
        timestamp="t",
    )

    rate_limiter.try_acquire.assert_awaited_once_with("eq-1")
    adapter.send_critical_failure.assert_awaited_once()


async def test_rate_limited_skips_telegram_entirely() -> None:
    service, adapter, rate_limiter = _make_service(rate_limit_acquired=False)

    await service.notify_if_critical(
        equipment_id="eq-1", equipment_name="Eq 1", probability=0.9, timestamp="t"
    )

    adapter.send_critical_failure.assert_not_called()


async def test_successful_send_does_not_release_rate_limit() -> None:
    """I) Telegram sucesso — rate limit permanece registrado (não é liberado)."""
    service, adapter, rate_limiter = _make_service(rate_limit_acquired=True)

    await service.notify_if_critical(
        equipment_id="eq-1", equipment_name="Eq 1", probability=0.9, timestamp="t"
    )

    rate_limiter.release.assert_not_called()


async def test_telegram_failure_releases_rate_limit_and_does_not_raise() -> None:
    """J/K/L) Telegram falha (erro genérico/timeout/HTTP 500) — rate limit
    liberado, e a falha NUNCA propaga para o chamador (RF-24 §16)."""
    service, adapter, rate_limiter = _make_service(
        rate_limit_acquired=True,
        adapter_error=TelegramNotificationError("qualquer falha simulada"),
    )

    await service.notify_if_critical(  # não deve lançar
        equipment_id="eq-1", equipment_name="Eq 1", probability=0.9, timestamp="t"
    )

    rate_limiter.release.assert_awaited_once_with("eq-1")


async def test_unexpected_exception_from_adapter_does_not_propagate() -> None:
    """Rede de segurança adicional — mesmo uma exceção NÃO prevista do
    adapter não pode derrubar o pipeline de inferência."""
    service, adapter, rate_limiter = _make_service(
        rate_limit_acquired=True, adapter_error=RuntimeError("bug inesperado")
    )

    await service.notify_if_critical(  # não deve lançar
        equipment_id="eq-1", equipment_name="Eq 1", probability=0.9, timestamp="t"
    )

    rate_limiter.release.assert_awaited_once_with("eq-1")


async def test_dashboard_url_built_from_equipment_id_route() -> None:
    service, adapter, _ = _make_service()

    await service.notify_if_critical(
        equipment_id="bomba-01",
        equipment_name="Bomba 01",
        probability=0.9,
        timestamp="t",
    )

    sent_notification: CriticalFailureNotification = (
        adapter.send_critical_failure.await_args.args[0]
    )
    assert sent_notification.dashboard_url == "http://localhost/sensors/bomba-01"
