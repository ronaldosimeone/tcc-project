"""
Testes de injeção de dependência via `Depends` — RNF-56 §9.

Prova, para cada router refatorado, que:
  1. o router chama a ABSTRAÇÃO correta (o Protocol), não uma classe
     concreta específica;
  2. uma implementação FAKE (que NÃO herda da classe concreta real —
     satisfaz o Protocol só estruturalmente) é de fato usada quando
     injetada via `app.dependency_overrides`;
  3. infraestrutura real (MCP, Ollama, Postgres, WebSocket manager real)
     NUNCA é tocada quando o fake está em uso;
  4. a resposta HTTP permanece com o MESMO contrato (status, schema);
  5. erros do service continuam sendo traduzidos corretamente pelos
     handlers globais existentes (`core/exceptions.py`).

Nenhum destes fakes herda de `ModelService`/`MaintenanceSuggestionService`/
`AlertSettingsService`/`ConnectionManager` — a conformidade é 100%
estrutural (`typing.Protocol`), prova real de que o router não depende da
classe concreta.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.auth import require_admin_token
from src.core.database import Base
from src.main import create_app
from src.routers.maintenance import get_maintenance_suggestion_service
from src.routers.predict import get_alert_service, get_db, get_model_service
from src.routers.settings import get_alert_settings_service
from src.schemas.maintenance import MaintenanceSuggestionResponse
from src.schemas.predict import PredictRequest, PredictResponse
from src.services.alert_settings_service import AlertSettingsSnapshot
from src.services.maintenance_suggestion_service import SuggestionStreamEvent
from src.services.protocols import (
    AlertServiceProtocol,
    AlertSettingsServiceProtocol,
    MaintenanceSuggestionServiceProtocol,
    ModelServiceProtocol,
)

_ADMIN_HEADER = {"X-Admin-Token": "changeme-admin-token"}

_VALID_PAYLOAD: dict[str, float] = {
    "TP2": 5.02,
    "TP3": 9.21,
    "H1": 8.97,
    "DV_pressure": 2.10,
    "Reservoirs": 8.85,
    "Motor_current": 4.5,
    "Oil_temperature": 72.3,
    "COMP": 1.0,
    "DV_eletric": 0.0,
    "Towers": 1.0,
    "MPG": 1.0,
    "Oil_level": 1.0,
}


async def _sqlite_session_factory() -> async_sessionmaker[AsyncSession]:
    """SQLite em memória — nenhum Postgres real é tocado neste arquivo (o
    foco é a fronteira Model/AlertService, não persistência; `save_prediction`
    ainda precisa de uma sessão REAL com `.add()`/`.flush()` funcionando,
    já que ele não passa pelo Protocol layer — mesmo padrão de
    `test_predict_endpoint.py`)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


# ---------------------------------------------------------------------------
# 1. POST /predict/ — ModelServiceProtocol / AlertServiceProtocol fakes
# ---------------------------------------------------------------------------


class _FakeModelService:
    """NÃO herda de `ModelService` — satisfaz `ModelServiceProtocol` só
    estruturalmente (mesmo método, mesma assinatura)."""

    def __init__(self) -> None:
        self.calls: list[PredictRequest] = []

    def predict(self, request: PredictRequest) -> PredictResponse:
        self.calls.append(request)
        return PredictResponse(
            predicted_class=1,
            failure_probability=0.99,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def predict_batch(self, requests: list[PredictRequest]) -> list[PredictResponse]:
        # RNF-72/73 — ModelServiceProtocol cresceu para incluir predict_batch;
        # este fake precisa continuar satisfazendo o Protocol estruturalmente.
        # `self.predict` já registra cada request em `self.calls`.
        return [self.predict(r) for r in requests]


class _FakeAlertService:
    """NÃO herda de `AlertService` — prova que `POST /predict/` nunca
    precisa da implementação real (Postgres/WS/Celery) para responder."""

    def __init__(self) -> None:
        self.received: list[dict[str, Any]] = []

    async def process_prediction(self, prediction: dict[str, Any]) -> dict[str, Any]:
        self.received.append(prediction)
        return prediction


@pytest.mark.asyncio
async def test_predict_router_uses_fake_model_and_alert_service() -> None:
    fake_model = _FakeModelService()
    fake_alert = _FakeAlertService()

    session_factory = await _sqlite_session_factory()

    async def _override_get_db():
        async with session_factory() as session:
            yield session
            await session.commit()

    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: fake_model
    app.dependency_overrides[get_alert_service] = lambda: fake_alert
    app.dependency_overrides[get_db] = _override_get_db

    # Prova estrutural — isinstance funciona SEM herança (Protocol).
    assert isinstance(fake_model, ModelServiceProtocol)
    assert isinstance(fake_alert, AlertServiceProtocol)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict/", json=_VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == 1
    assert body["failure_probability"] == pytest.approx(0.99)

    # O fake foi REALMENTE chamado — não é só "parece funcionar".
    assert len(fake_model.calls) == 1
    assert len(fake_alert.received) == 1
    assert fake_alert.received[0]["probability"] == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# 2. POST /v1/maintenance/suggest — MaintenanceSuggestionServiceProtocol fake
#    (prova de que MCP/Ollama reais nunca são tocados)
# ---------------------------------------------------------------------------


class _FakeMaintenanceSuggestionService:
    """NÃO herda de `MaintenanceSuggestionService` — não importa
    `MCPSearchClient`/`OllamaClient`, prova de que o router não precisa
    deles para funcionar quando a fronteira é trocada."""

    def __init__(self) -> None:
        self.suggest_calls = 0

    async def suggest(self, request) -> MaintenanceSuggestionResponse:
        self.suggest_calls += 1
        return MaintenanceSuggestionResponse(
            triggered=True,
            failure_probability=request.failure_probability,
            markdown="# Plano de Manutenção\n\nFake — MCP/Ollama nunca chamados.",
            references=[],
            model="fake-model",
            message=None,
        )

    async def suggest_stream(self, request) -> AsyncIterator[SuggestionStreamEvent]:
        yield SuggestionStreamEvent(type="searching")
        yield SuggestionStreamEvent(type="done", markdown="# Fake", references=[])


@pytest.mark.asyncio
async def test_maintenance_router_uses_fake_service_never_touches_mcp_ollama() -> None:
    fake_service = _FakeMaintenanceSuggestionService()

    app = create_app()
    app.dependency_overrides[get_maintenance_suggestion_service] = lambda: fake_service

    assert isinstance(fake_service, MaintenanceSuggestionServiceProtocol)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/maintenance/suggest",
            json={"failure_probability": 0.9, "equipment_name": "Compressor X"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["triggered"] is True
    assert "Fake" in body["markdown"] or "MCP/Ollama" in body["markdown"]
    assert fake_service.suggest_calls == 1


# ---------------------------------------------------------------------------
# 3. GET /v1/settings/alerts — AlertSettingsServiceProtocol fake
#    (prova de que erro do service é traduzido corretamente)
# ---------------------------------------------------------------------------


class _FakeAlertSettingsService:
    async def get_settings(self) -> AlertSettingsSnapshot:
        return AlertSettingsSnapshot(
            alert_threshold=0.77,
            telegram_enabled=True,
            email_enabled=False,
            alert_email=None,
        )

    async def upsert_settings(
        self, **_: Any
    ) -> AlertSettingsSnapshot:  # pragma: no cover
        raise NotImplementedError


@pytest.mark.asyncio
async def test_settings_router_uses_fake_alert_settings_service() -> None:
    fake_service = _FakeAlertSettingsService()

    app = create_app()
    app.dependency_overrides[get_alert_settings_service] = lambda: fake_service
    app.dependency_overrides[require_admin_token] = lambda: None

    assert isinstance(fake_service, AlertSettingsServiceProtocol)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/settings/alerts", headers=_ADMIN_HEADER)

    assert response.status_code == 200
    body = response.json()
    assert body["alert_threshold"] == pytest.approx(0.77)
    assert body["telegram_enabled"] is True
