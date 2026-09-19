"""
Testes de configuração global de alertas — RF-25 / RNF-49.

Camadas testadas:
  1. `AlertSettingsService` — banco REAL (SQLite em memória, `StaticPool`,
     mesmo padrão de `test_notifications.py`): defaults sem registro,
     upsert cria, upsert de novo atualiza a MESMA linha (singleton real).
  2. Validação Pydantic via HTTP (`PUT /v1/settings/alerts`) — threshold
     (7 valores do critério de aceite) + e-mail (válido/inválido/
     obrigatório-quando-habilitado).
  3. Autorização — mesmo mecanismo do RF-11 (`X-Admin-Token`).
  4. Integração real com `CriticalFailureNotificationService` (RF-24) —
     prova que threshold/canais configurados realmente mudam o
     comportamento do alerta (Telegram e e-mail), não é só visual.
  5. `NotificationTestService` (botão "Testar Notificação") — só canais
     habilitados, sucesso/falha por canal, falha parcial, nenhum canal
     habilitado, rate limit próprio.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from src.core.config import settings
from src.core.database import Base
from src.core.exceptions import (
    EmailNotificationError,
    NoNotificationChannelEnabledError,
    NotificationTestFailedError,
    NotificationTestRateLimitedError,
    TelegramNotificationError,
)
from src.main import create_app
from src.models.alert_settings import AlertSettings
from src.routers.settings import (
    get_alert_settings_service,
    get_notification_test_service,
)
from src.services.alert_settings_service import (
    AlertSettingsService,
    AlertSettingsSnapshot,
)
from src.services.critical_failure_notification_service import (
    CriticalFailureNotificationService,
)
from src.services.notification_test_service import (
    FAILURE_MESSAGE,
    TEST_LOCK_KEY,
    NotificationTestService,
)
from src.services.telegram_alert_rate_limiter import TelegramAlertRateLimiter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _dev_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """RNF-62 — o bypass de auth do RF-11 agora exige `DEBUG=true` (ver
    `src/core/auth.py`). Este arquivo testa o comportamento em modo dev
    (todos os casos assertam `admin_api_token == "change-me-in-production"`),
    então liga o DEBUG por padrão. Casos específicos sobrescrevem
    (`test_requires_admin_token_when_configured` põe um token real;
    `test_placeholder_token_without_debug_fails_closed` desliga o DEBUG)."""
    monkeypatch.setattr(settings, "debug", True)


# SQLite em memória (StaticPool), mesmo padrão de test_notifications.py.
@pytest.fixture()
async def db_session_factory():
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


def _default_response(**overrides):
    body = {
        "alert_threshold": 0.85,
        "telegram_enabled": True,
        "email_enabled": False,
        "alert_email": None,
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# 1. AlertSettingsService — banco real, singleton
# ---------------------------------------------------------------------------


async def test_get_settings_returns_defaults_when_no_row(db_session_factory) -> None:
    service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    config = await service.get_settings()
    assert config.alert_threshold == 0.85
    assert config.telegram_enabled is True
    assert config.email_enabled is False
    assert config.alert_email is None


async def test_get_settings_never_creates_a_row(db_session_factory) -> None:
    service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    await service.get_settings()
    async with db_session_factory() as db:
        result = await db.execute(select(AlertSettings))
        assert result.first() is None


async def test_upsert_settings_then_get_returns_saved_value(
    db_session_factory,
) -> None:
    service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    saved = await service.upsert_settings(
        alert_threshold=0.75,
        telegram_enabled=True,
        email_enabled=True,
        alert_email="alertas@empresa.com.br",
    )
    assert saved.alert_threshold == 0.75
    assert saved.email_enabled is True
    assert saved.alert_email == "alertas@empresa.com.br"

    config = await service.get_settings()
    assert config == saved


async def test_upsert_settings_twice_updates_same_row(db_session_factory) -> None:
    """Salvar de novo deve ATUALIZAR a linha existente — nunca criar uma
    segunda (singleton real, RF-25 §9 "Global")."""
    service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    await service.upsert_settings(
        alert_threshold=0.60,
        telegram_enabled=True,
        email_enabled=False,
        alert_email=None,
    )
    await service.upsert_settings(
        alert_threshold=0.90,
        telegram_enabled=False,
        email_enabled=True,
        alert_email="a@b.com",
    )

    async with db_session_factory() as db:
        rows = (await db.execute(select(AlertSettings))).scalars().all()
        assert len(rows) == 1
        assert rows[0].alert_threshold == 0.90
        assert rows[0].telegram_enabled is False
        assert rows[0].alert_email == "a@b.com"


async def test_upsert_threshold_preserves_other_fields(db_session_factory) -> None:
    """Compatibilidade RF-24: alterar só o limiar não deve apagar canais já
    configurados."""
    service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    await service.upsert_settings(
        alert_threshold=0.70,
        telegram_enabled=False,
        email_enabled=True,
        alert_email="ops@empresa.com",
    )
    saved_threshold = await service.upsert_threshold(0.80)
    assert saved_threshold == 0.80

    config = await service.get_settings()
    assert config.alert_threshold == 0.80
    assert config.telegram_enabled is False
    assert config.email_enabled is True
    assert config.alert_email == "ops@empresa.com"


# ---------------------------------------------------------------------------
# 2. Validação — HTTP (Pydantic) + persistência real
# ---------------------------------------------------------------------------


def _app_with_settings_service(service: AlertSettingsService):
    application = create_app()
    application.dependency_overrides[get_alert_settings_service] = lambda: service
    return application


async def _client_for(service: AlertSettingsService) -> AsyncClient:
    transport = ASGITransport(app=_app_with_settings_service(service))
    return AsyncClient(transport=transport, base_url="http://testserver")


@pytest.mark.parametrize(
    "value,expected_status",
    [
        (0.49, 422),
        (0.50, 200),
        (0.60, 200),
        (0.70, 200),
        (0.75, 200),
        (0.85, 200),
        (0.95, 200),
        (0.951, 422),
        (1.00, 422),
    ],
)
async def test_put_validates_threshold_range(
    db_session_factory, value: float, expected_status: int
) -> None:
    service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    async with await _client_for(service) as client:
        response = await client.put(
            "/v1/settings/alerts", json={"alert_threshold": value}
        )
    assert response.status_code == expected_status
    if expected_status == 200:
        assert response.json()["alert_threshold"] == value


@pytest.mark.parametrize(
    "email,expected_status",
    [
        ("alertas@empresa.com.br", 200),
        ("teste", 422),
        ("teste@", 422),
        ("@empresa.com", 422),
    ],
)
async def test_put_validates_email_format(
    db_session_factory, email: str, expected_status: int
) -> None:
    service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    async with await _client_for(service) as client:
        response = await client.put(
            "/v1/settings/alerts",
            json={
                "alert_threshold": 0.85,
                "email_enabled": True,
                "alert_email": email,
            },
        )
    assert response.status_code == expected_status


async def test_put_rejects_email_enabled_without_address(db_session_factory) -> None:
    service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    async with await _client_for(service) as client:
        response = await client.put(
            "/v1/settings/alerts",
            json={"alert_threshold": 0.85, "email_enabled": True},
        )
    assert response.status_code == 422


async def test_put_allows_email_disabled_without_address(db_session_factory) -> None:
    service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    async with await _client_for(service) as client:
        response = await client.put(
            "/v1/settings/alerts",
            json={"alert_threshold": 0.85, "email_enabled": False},
        )
    assert response.status_code == 200
    assert response.json()["alert_email"] is None


async def test_get_returns_defaults_before_any_put(db_session_factory) -> None:
    service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    async with await _client_for(service) as client:
        response = await client.get("/v1/settings/alerts")
    assert response.status_code == 200
    assert response.json() == _default_response()


async def test_get_after_put_reflects_saved_value(db_session_factory) -> None:
    """RF-25 §24 — persistência real de ponta a ponta via HTTP: PUT, depois
    um GET NOVO (nova requisição) devolve a configuração salva."""
    service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    async with await _client_for(service) as client:
        put_response = await client.put(
            "/v1/settings/alerts",
            json={
                "alert_threshold": 0.75,
                "telegram_enabled": True,
                "email_enabled": True,
                "alert_email": "alertas@empresa.com.br",
            },
        )
        assert put_response.status_code == 200
        assert put_response.json() == _default_response(
            alert_threshold=0.75,
            email_enabled=True,
            alert_email="alertas@empresa.com.br",
        )

        get_response = await client.get("/v1/settings/alerts")
        assert get_response.json() == put_response.json()

    # Reconfigura — threshold sobe, e-mail desliga, Telegram continua ligado.
    async with await _client_for(service) as client:
        await client.put(
            "/v1/settings/alerts",
            json={"alert_threshold": 0.90, "telegram_enabled": True},
        )
        get_response = await client.get("/v1/settings/alerts")
        assert get_response.json() == _default_response(alert_threshold=0.90)


# ---------------------------------------------------------------------------
# 3. Autorização — mesmo mecanismo do RF-11 (X-Admin-Token), sem auth nova
# ---------------------------------------------------------------------------


async def test_requires_admin_token_when_configured(
    db_session_factory, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "admin_api_token", "real-secret-token")
    service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    async with await _client_for(service) as client:
        no_token = await client.get("/v1/settings/alerts")
        wrong_token = await client.get(
            "/v1/settings/alerts", headers={"X-Admin-Token": "wrong"}
        )
        right_token = await client.get(
            "/v1/settings/alerts", headers={"X-Admin-Token": "real-secret-token"}
        )
    assert no_token.status_code == 401
    assert wrong_token.status_code == 401
    assert right_token.status_code == 200


async def test_dev_placeholder_token_allows_access_without_header(
    db_session_factory, monkeypatch
) -> None:
    """Com o token default (`change-me-in-production`) **e** `DEBUG=true`
    (modo dev — RF-11 / RNF-62), nenhum header é exigido."""
    assert settings.admin_api_token == "change-me-in-production"
    monkeypatch.setattr(settings, "debug", True)
    service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    async with await _client_for(service) as client:
        response = await client.get("/v1/settings/alerts")
    assert response.status_code == 200


async def test_placeholder_token_without_debug_fails_closed(
    db_session_factory, monkeypatch
) -> None:
    """RNF-62 — token ainda no placeholder mas `DEBUG` desligado (produção):
    as rotas admin NÃO ficam abertas, retornam 401 (fail-closed)."""
    assert settings.admin_api_token == "change-me-in-production"
    monkeypatch.setattr(settings, "debug", False)
    service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    async with await _client_for(service) as client:
        response = await client.get("/v1/settings/alerts")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# 4. Integração real com CriticalFailureNotificationService (RF-24/RF-25) —
#    prova que threshold/canais configurados realmente mudam o alerta.
# ---------------------------------------------------------------------------


def _notifier(settings_service, adapter=None, email_adapter=None, rate_limiter=None):
    rl = rate_limiter or AsyncMock()
    if rate_limiter is None:
        rl.try_acquire = AsyncMock(return_value=True)
    return CriticalFailureNotificationService(
        adapter=adapter or AsyncMock(),
        rate_limiter=rl,
        dashboard_url="http://localhost",
        settings_service=settings_service,
        email_adapter=email_adapter,
    )


async def test_configured_threshold_actually_gates_the_alert(
    db_session_factory,
) -> None:
    settings_service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    await settings_service.upsert_threshold(0.75)

    adapter = AsyncMock()
    notifier = _notifier(settings_service, adapter=adapter)

    await notifier.notify_if_critical(
        equipment_id="eq-1", equipment_name="Equip", probability=0.74, timestamp="t"
    )
    adapter.send_critical_failure.assert_not_awaited()

    await notifier.notify_if_critical(
        equipment_id="eq-1", equipment_name="Equip", probability=0.76, timestamp="t"
    )
    adapter.send_critical_failure.assert_awaited_once()


async def test_reconfiguring_threshold_changes_behavior_again(
    db_session_factory,
) -> None:
    settings_service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    await settings_service.upsert_threshold(0.75)
    adapter = AsyncMock()
    notifier = _notifier(settings_service, adapter=adapter)

    await notifier.notify_if_critical(
        equipment_id="eq-1", equipment_name="Equip", probability=0.80, timestamp="t"
    )
    assert adapter.send_critical_failure.await_count == 1

    await settings_service.upsert_threshold(0.90)
    await notifier.notify_if_critical(
        equipment_id="eq-2", equipment_name="Equip", probability=0.80, timestamp="t"
    )
    assert adapter.send_critical_failure.await_count == 1  # não mudou (0.80 <= 0.90)

    await notifier.notify_if_critical(
        equipment_id="eq-3", equipment_name="Equip", probability=0.91, timestamp="t"
    )
    assert adapter.send_critical_failure.await_count == 2


async def test_telegram_disabled_skips_telegram_channel(db_session_factory) -> None:
    settings_service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    await settings_service.upsert_settings(
        alert_threshold=0.70,
        telegram_enabled=False,
        email_enabled=True,
        alert_email="ops@empresa.com",
    )
    adapter = AsyncMock()
    email_adapter = AsyncMock()
    email_adapter.send_critical_failure_email = AsyncMock(return_value=None)
    notifier = _notifier(settings_service, adapter=adapter, email_adapter=email_adapter)

    await notifier.notify_if_critical(
        equipment_id="eq-1", equipment_name="Equip", probability=0.90, timestamp="t"
    )
    adapter.send_critical_failure.assert_not_awaited()
    email_adapter.send_critical_failure_email.assert_awaited_once()


async def test_both_channels_enabled_both_attempted(db_session_factory) -> None:
    settings_service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    await settings_service.upsert_settings(
        alert_threshold=0.70,
        telegram_enabled=True,
        email_enabled=True,
        alert_email="ops@empresa.com",
    )
    adapter = AsyncMock()
    email_adapter = AsyncMock()
    email_adapter.send_critical_failure_email = AsyncMock(return_value=None)
    notifier = _notifier(settings_service, adapter=adapter, email_adapter=email_adapter)

    await notifier.notify_if_critical(
        equipment_id="eq-1", equipment_name="Equip", probability=0.90, timestamp="t"
    )
    adapter.send_critical_failure.assert_awaited_once()
    email_adapter.send_critical_failure_email.assert_awaited_once()


async def test_one_channel_failing_does_not_block_the_other(
    db_session_factory,
) -> None:
    """RF-25 §17 — Telegram falhou + e-mail OK (e vice-versa): um canal não
    impede o outro, e o lock é mantido (ao menos um teve sucesso)."""
    settings_service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    await settings_service.upsert_settings(
        alert_threshold=0.70,
        telegram_enabled=True,
        email_enabled=True,
        alert_email="ops@empresa.com",
    )
    adapter = AsyncMock()
    adapter.send_critical_failure = AsyncMock(
        side_effect=TelegramNotificationError("falhou")
    )
    email_adapter = AsyncMock()
    email_adapter.send_critical_failure_email = AsyncMock(return_value=None)
    rate_limiter = AsyncMock()
    rate_limiter.try_acquire = AsyncMock(return_value=True)
    rate_limiter.release = AsyncMock()
    notifier = _notifier(
        settings_service,
        adapter=adapter,
        email_adapter=email_adapter,
        rate_limiter=rate_limiter,
    )

    await notifier.notify_if_critical(
        equipment_id="eq-1", equipment_name="Equip", probability=0.90, timestamp="t"
    )
    adapter.send_critical_failure.assert_awaited_once()
    email_adapter.send_critical_failure_email.assert_awaited_once()
    rate_limiter.release.assert_not_awaited()  # e-mail teve sucesso — mantém o lock


async def test_all_channels_failing_releases_the_lock(db_session_factory) -> None:
    settings_service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    await settings_service.upsert_settings(
        alert_threshold=0.70,
        telegram_enabled=True,
        email_enabled=True,
        alert_email="ops@empresa.com",
    )
    adapter = AsyncMock()
    adapter.send_critical_failure = AsyncMock(
        side_effect=TelegramNotificationError("falhou")
    )
    email_adapter = AsyncMock()
    email_adapter.send_critical_failure_email = AsyncMock(
        side_effect=EmailNotificationError("falhou")
    )
    rate_limiter = AsyncMock()
    rate_limiter.try_acquire = AsyncMock(return_value=True)
    rate_limiter.release = AsyncMock()
    notifier = _notifier(
        settings_service,
        adapter=adapter,
        email_adapter=email_adapter,
        rate_limiter=rate_limiter,
    )

    await notifier.notify_if_critical(
        equipment_id="eq-1", equipment_name="Equip", probability=0.90, timestamp="t"
    )
    rate_limiter.release.assert_awaited_once_with("eq-1")


async def test_settings_service_none_preserves_rf24_fixed_behavior() -> None:
    """Não-regressão explícita: sem AlertSettingsService (RF-24 original),
    o limiar continua fixo em 0.85, Telegram sempre tentado, e-mail nunca."""
    adapter = AsyncMock()
    rate_limiter = AsyncMock()
    rate_limiter.try_acquire = AsyncMock(return_value=True)
    notifier = CriticalFailureNotificationService(
        adapter=adapter, rate_limiter=rate_limiter, dashboard_url="http://localhost"
    )
    await notifier.notify_if_critical(
        equipment_id="eq-1", equipment_name="Equip", probability=0.85, timestamp="t"
    )
    adapter.send_critical_failure.assert_not_awaited()

    await notifier.notify_if_critical(
        equipment_id="eq-1", equipment_name="Equip", probability=0.86, timestamp="t"
    )
    adapter.send_critical_failure.assert_awaited_once()


async def test_settings_read_failure_falls_back_to_rf24_defaults() -> None:
    broken_settings_service = AsyncMock(spec=AlertSettingsService)
    broken_settings_service.get_settings = AsyncMock(
        side_effect=RuntimeError("db down")
    )
    adapter = AsyncMock()
    rate_limiter = AsyncMock()
    rate_limiter.try_acquire = AsyncMock(return_value=True)
    notifier = CriticalFailureNotificationService(
        adapter=adapter,
        rate_limiter=rate_limiter,
        dashboard_url="http://localhost",
        settings_service=broken_settings_service,
    )
    await notifier.notify_if_critical(
        equipment_id="eq-1", equipment_name="Equip", probability=0.86, timestamp="t"
    )
    adapter.send_critical_failure.assert_awaited_once()  # 0.86 > 0.85 (fallback)


# ---------------------------------------------------------------------------
# 5. NotificationTestService — botão "Testar Notificação"
# ---------------------------------------------------------------------------


def _test_service(
    settings_service, telegram_adapter=None, email_adapter=None, rate_limiter=None
) -> NotificationTestService:
    rl = rate_limiter or TelegramAlertRateLimiter()
    if rate_limiter is None:
        rl.try_acquire = AsyncMock(return_value=True)  # type: ignore[method-assign]
        rl.release = AsyncMock()  # type: ignore[method-assign]
    return NotificationTestService(
        telegram_adapter=telegram_adapter or AsyncMock(),
        email_adapter=email_adapter or AsyncMock(),
        settings_service=settings_service,
        test_rate_limiter=rl,
    )


async def test_notification_test_telegram_only(db_session_factory) -> None:
    settings_service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    # Default: telegram_enabled=True, email_enabled=False — nada a salvar.
    telegram_adapter = AsyncMock()
    telegram_adapter.send_test_notification = AsyncMock(return_value=None)
    email_adapter = AsyncMock()
    service = _test_service(
        settings_service, telegram_adapter=telegram_adapter, email_adapter=email_adapter
    )

    message = await service.send_test_notification()
    assert "Telegram: enviado" in message
    telegram_adapter.send_test_notification.assert_awaited_once()
    email_adapter.send_test_email.assert_not_awaited()


async def test_notification_test_email_only(db_session_factory) -> None:
    settings_service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    await settings_service.upsert_settings(
        alert_threshold=0.85,
        telegram_enabled=False,
        email_enabled=True,
        alert_email="ops@empresa.com",
    )
    telegram_adapter = AsyncMock()
    email_adapter = AsyncMock()
    email_adapter.send_test_email = AsyncMock(return_value=None)
    service = _test_service(
        settings_service, telegram_adapter=telegram_adapter, email_adapter=email_adapter
    )

    message = await service.send_test_notification()
    assert "E-mail: enviado" in message
    telegram_adapter.send_test_notification.assert_not_awaited()
    email_adapter.send_test_email.assert_awaited_once_with(to="ops@empresa.com")


async def test_notification_test_both_channels(db_session_factory) -> None:
    settings_service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    await settings_service.upsert_settings(
        alert_threshold=0.85,
        telegram_enabled=True,
        email_enabled=True,
        alert_email="ops@empresa.com",
    )
    telegram_adapter = AsyncMock()
    telegram_adapter.send_test_notification = AsyncMock(return_value=None)
    email_adapter = AsyncMock()
    email_adapter.send_test_email = AsyncMock(return_value=None)
    service = _test_service(
        settings_service, telegram_adapter=telegram_adapter, email_adapter=email_adapter
    )

    message = await service.send_test_notification()
    assert "Telegram: enviado" in message
    assert "E-mail: enviado" in message


async def test_notification_test_partial_failure_still_succeeds(
    db_session_factory,
) -> None:
    """Telegram falha, e-mail funciona — retorna sucesso (200) com o
    detalhe por canal, não um erro total (RF-25 §17)."""
    settings_service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    await settings_service.upsert_settings(
        alert_threshold=0.85,
        telegram_enabled=True,
        email_enabled=True,
        alert_email="ops@empresa.com",
    )
    telegram_adapter = AsyncMock()
    telegram_adapter.send_test_notification = AsyncMock(
        side_effect=TelegramNotificationError("falhou")
    )
    email_adapter = AsyncMock()
    email_adapter.send_test_email = AsyncMock(return_value=None)
    service = _test_service(
        settings_service, telegram_adapter=telegram_adapter, email_adapter=email_adapter
    )

    message = await service.send_test_notification()
    assert "Telegram: falhou" in message
    assert "E-mail: enviado" in message


async def test_notification_test_no_channel_enabled_raises_400(
    db_session_factory,
) -> None:
    settings_service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    await settings_service.upsert_settings(
        alert_threshold=0.85,
        telegram_enabled=False,
        email_enabled=False,
        alert_email=None,
    )
    service = _test_service(settings_service)

    with pytest.raises(NoNotificationChannelEnabledError):
        await service.send_test_notification()


async def test_notification_test_all_channels_failing_raises_502(
    db_session_factory,
) -> None:
    settings_service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    await settings_service.upsert_settings(
        alert_threshold=0.85,
        telegram_enabled=True,
        email_enabled=True,
        alert_email="ops@empresa.com",
    )
    telegram_adapter = AsyncMock()
    telegram_adapter.send_test_notification = AsyncMock(
        side_effect=TelegramNotificationError("falhou")
    )
    email_adapter = AsyncMock()
    email_adapter.send_test_email = AsyncMock(
        side_effect=EmailNotificationError("falhou")
    )
    service = _test_service(
        settings_service, telegram_adapter=telegram_adapter, email_adapter=email_adapter
    )

    with pytest.raises(NotificationTestFailedError):
        await service.send_test_notification()


async def test_notification_test_rate_limited_does_not_call_adapters(
    db_session_factory,
) -> None:
    settings_service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    telegram_adapter = AsyncMock()
    rate_limiter = TelegramAlertRateLimiter()
    rate_limiter.try_acquire = AsyncMock(return_value=False)  # type: ignore[method-assign]
    service = _test_service(
        settings_service, telegram_adapter=telegram_adapter, rate_limiter=rate_limiter
    )

    with pytest.raises(NotificationTestRateLimitedError):
        await service.send_test_notification()
    telegram_adapter.send_test_notification.assert_not_awaited()


async def test_notification_test_failure_releases_lock_for_retry(
    db_session_factory,
) -> None:
    settings_service = AlertSettingsService(
        default_threshold=0.85, session_factory=db_session_factory
    )
    telegram_adapter = AsyncMock()
    telegram_adapter.send_test_notification = AsyncMock(
        side_effect=TelegramNotificationError("config ausente")
    )
    rate_limiter = TelegramAlertRateLimiter()
    rate_limiter.try_acquire = AsyncMock(return_value=True)  # type: ignore[method-assign]
    rate_limiter.release = AsyncMock()  # type: ignore[method-assign]
    service = _test_service(
        settings_service, telegram_adapter=telegram_adapter, rate_limiter=rate_limiter
    )

    with pytest.raises(NotificationTestFailedError):
        await service.send_test_notification()
    rate_limiter.release.assert_awaited_once_with(TEST_LOCK_KEY)


def test_test_lock_key_never_collides_with_a_real_equipment_id() -> None:
    assert TEST_LOCK_KEY != settings.default_equipment_id
    assert "__" in TEST_LOCK_KEY


def test_failure_message_never_mentions_internal_details() -> None:
    assert "token" not in FAILURE_MESSAGE.lower()
    assert "api_key" not in FAILURE_MESSAGE.lower()
    assert "traceback" not in FAILURE_MESSAGE.lower()


# ---------------------------------------------------------------------------
# 6. Endpoint HTTP — POST /v1/settings/alerts/test
# ---------------------------------------------------------------------------


def _app_with_test_service(service: NotificationTestService):
    application = create_app()
    application.dependency_overrides[get_notification_test_service] = lambda: service
    return application


async def _test_client_for(service: NotificationTestService) -> AsyncClient:
    transport = ASGITransport(app=_app_with_test_service(service))
    return AsyncClient(transport=transport, base_url="http://testserver")


async def test_endpoint_test_notification_success() -> None:
    fake_service = AsyncMock(spec=NotificationTestService)
    fake_service.send_test_notification = AsyncMock(return_value="Telegram: enviado")
    async with await _test_client_for(fake_service) as client:
        response = await client.post("/v1/settings/alerts/test")
    assert response.status_code == 200
    assert response.json() == {"message": "Telegram: enviado"}


async def test_endpoint_test_notification_no_channel_returns_400() -> None:
    fake_service = AsyncMock(spec=NotificationTestService)
    fake_service.send_test_notification = AsyncMock(
        side_effect=NoNotificationChannelEnabledError()
    )
    async with await _test_client_for(fake_service) as client:
        response = await client.post("/v1/settings/alerts/test")
    assert response.status_code == 400


async def test_endpoint_test_notification_rate_limited_returns_429() -> None:
    fake_service = AsyncMock(spec=NotificationTestService)
    fake_service.send_test_notification = AsyncMock(
        side_effect=NotificationTestRateLimitedError()
    )
    async with await _test_client_for(fake_service) as client:
        response = await client.post("/v1/settings/alerts/test")
    assert response.status_code == 429


async def test_endpoint_test_notification_all_channels_failed_returns_502() -> None:
    fake_service = AsyncMock(spec=NotificationTestService)
    fake_service.send_test_notification = AsyncMock(
        side_effect=NotificationTestFailedError("Telegram: falhou · E-mail: falhou")
    )
    async with await _test_client_for(fake_service) as client:
        response = await client.post("/v1/settings/alerts/test")
    assert response.status_code == 502
    body = response.json()
    assert "TELEGRAM_BOT_TOKEN" not in str(body)
    assert "RESEND_API_KEY" not in str(body)
    assert "token" not in body["detail"].lower()


def test_alert_settings_snapshot_is_immutable() -> None:
    """RNF-64: `@dataclass(frozen=True)` — uma tentativa de mutar um campo
    depois de criado deve levantar, não silenciosamente aceitar (o
    snapshot é lido/logado em vários pontos assumindo que não muda por
    baixo)."""
    snapshot = AlertSettingsSnapshot(
        alert_threshold=0.85,
        telegram_enabled=True,
        email_enabled=False,
        alert_email=None,
    )
    with pytest.raises(AttributeError):
        snapshot.alert_threshold = 0.5  # type: ignore[misc]
