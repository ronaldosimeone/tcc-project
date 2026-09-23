"""
Testes de integração — GET /metrics (RNF-76 Fase 11).

Cada app de teste usa `create_app()` fresh (mesmo padrão de test_health.py)
— um registry Prometheus NOVO por app (ver `src/core/metrics.py`
`setup_http_instrumentation`), então os valores aqui são absolutos e
confiáveis dentro do MESMO app_client, sem interferência de outros
arquivos de teste rodando no mesmo processo.

Não basta `GET /metrics` devolver 200 (regra 8 do RNF-76) — cada teste
aqui dispara uma requisição REAL e confirma que o contador certo mudou,
não só que o endpoint responde.
"""

from __future__ import annotations

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


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_200(async_client: AsyncClient) -> None:
    response = await async_client.get("/metrics")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_metrics_endpoint_exposes_known_metric_names(
    async_client: AsyncClient,
) -> None:
    """Nomes concretos das métricas implementadas — não uma substring
    genérica tipo "http" (regra da Fase 11: "não testar apenas substring
    genérica; validar nomes concretos das métricas implementadas")."""
    response = await async_client.get("/metrics")
    body = response.text

    for metric_name in (
        "http_requests_total",
        "http_request_duration_seconds",
        "predictiq_inference_total",
        "predictiq_inference_duration_seconds",
        "predictiq_predictions_total",
        "predictiq_batch_requests_total",
        "predictiq_batch_samples_total",
    ):
        assert f"# HELP {metric_name}" in body, f"{metric_name} ausente de /metrics"


@pytest.mark.asyncio
async def test_metrics_endpoint_content_type_is_prometheus_text_format(
    async_client: AsyncClient,
) -> None:
    response = await async_client.get("/metrics")
    assert "text/plain" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_real_request_increments_http_requests_total(
    async_client: AsyncClient,
) -> None:
    """GET /health real → confirma que o contador HTTP correspondente
    aumentou (Fase 11: "executar request real... depois verificar que o
    contador aumentou", não apenas que /metrics responde)."""
    before = await async_client.get("/metrics")
    before_health_lines = [
        line
        for line in before.text.splitlines()
        if line.startswith("http_requests_total") and 'handler="/health"' in line
    ]
    assert before_health_lines == []

    await async_client.get("/health")
    await async_client.get("/health")

    after = await async_client.get("/metrics")
    health_lines = [
        line
        for line in after.text.splitlines()
        if line.startswith("http_requests_total")
        and 'handler="/health"' in line
        and 'status="200"' in line
    ]
    assert len(health_lines) == 1
    # value é o último token da linha "metric{labels} value"
    value = float(health_lines[0].rsplit(" ", 1)[-1])
    assert value == 2.0


@pytest.mark.asyncio
async def test_metrics_route_itself_is_excluded_from_http_requests_total(
    async_client: AsyncClient,
) -> None:
    """`excluded_handlers=["/metrics"]` — o scrape do Prometheus não deve
    se autocontar como tráfego de aplicação."""
    await async_client.get("/metrics")
    after = await async_client.get("/metrics")
    metrics_lines = [
        line
        for line in after.text.splitlines()
        if line.startswith("http_requests_total") and 'handler="/metrics"' in line
    ]
    assert metrics_lines == []
