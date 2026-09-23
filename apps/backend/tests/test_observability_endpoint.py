"""
Teste de integração — GET /observability/error-rate (RNF-77).

Mesmo padrão de fixtures de `tests/test_health.py` (SQLite em memória via
`get_db` override) — este endpoint não usa DB, mas `create_app()` monta a
app inteira, então a fixture cobre qualquer dependência transitiva.
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.database import get_db
from src.main import create_app

TEST_DATABASE_URL: str = "sqlite+aiosqlite:///:memory:"

_test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
_TestSessionFactory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=_test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def _override_get_db():  # type: ignore[return]
    async with _TestSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@pytest.fixture()
def app_with_mock_db():
    application = create_app()
    application.dependency_overrides[get_db] = _override_get_db
    return application


@pytest_asyncio.fixture()
async def async_client(app_with_mock_db):
    transport = ASGITransport(app=app_with_mock_db)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


async def test_error_rate_endpoint_returns_200_even_without_prometheus(
    async_client: AsyncClient,
) -> None:
    """Sem Prometheus configurado/alcançável neste teste — o endpoint
    NUNCA propaga o erro de rede como 500 (RNF-77: widget não pode
    derrubar o Dashboard)."""
    response = await async_client.get("/observability/error-rate")
    assert response.status_code == 200


async def test_error_rate_endpoint_response_schema(
    async_client: AsyncClient,
) -> None:
    response = await async_client.get("/observability/error-rate")
    body = response.json()
    assert body["status"] in ("NORMAL", "WARNING", "CRITICAL")
    assert isinstance(body["error_rate"], float)
    assert "window" in body
    assert "threshold_warning" in body
    assert "threshold_critical" in body
    assert "prometheus_reachable" in body


async def test_error_rate_endpoint_reflects_prometheus_response(
    async_client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Não basta o endpoint responder 200 — confirma que o corpo reflete
    uma consulta Prometheus real (mockada na camada HTTP, não a função).

    `httpx.AsyncClient.get` é a MESMA classe usada por `async_client`
    (fixture, fala com a app via ASGITransport) e pelo
    `observability_service` (fala com o Prometheus) — um mock
    incondicional interceptaria as DUAS chamadas. `fake_get` só desvia a
    requisição que tem o Prometheus como destino; qualquer outra
    (incluindo a do próprio `async_client` contra a rota FastAPI) segue
    pro `_real_get` original.
    """
    ok_response = httpx.Response(
        status_code=200,
        json={
            "status": "success",
            "data": {"resultType": "vector", "result": [{"value": [0, "0.5"]}]},
        },
        request=httpx.Request("GET", "http://prometheus:9090/api/v1/query"),
    )
    _real_get = httpx.AsyncClient.get

    async def fake_get(self: httpx.AsyncClient, url, *args, **kwargs):
        if "prometheus" in str(url):
            return ok_response
        return await _real_get(self, url, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    response = await async_client.get("/observability/error-rate")
    body = response.json()
    assert body["status"] == "CRITICAL"
    assert body["error_rate"] == 0.5
    assert body["prometheus_reachable"] is True
