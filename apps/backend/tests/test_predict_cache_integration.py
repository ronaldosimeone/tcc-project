"""
Integração POST /predict + InferenceCache — RNF-70 / RNF-71.

Diferente de test_predict_endpoint.py (que não conhece cache — roda com
`app.state.inference_cache` ausente, InferenceCache degrada pra MISS/no-op
contra um Redis inalcançável, ver services/inference_cache.py), este arquivo
injeta um InferenceCache real, contra Redis real (mesma exigência de
RNF-71 — TTL observável não pode ser provado com mock), via
`dependency_overrides[get_inference_cache]`.

Cobertura (RNF-70 §9):
  - 1ª requisição: MISS -> chama ModelService.predict + persiste em
    `predictions` + seta cache.
  - 2ª requisição idêntica (mesmos sensores, mesmo modelo): HIT -> NÃO
    chama ModelService.predict de novo, NÃO insere uma 2ª linha em
    `predictions`.
  - Requisição com sensor DIFERENTE: MISS de novo (chave diferente).
  - Depois do TTL expirar (acelerado via PEXPIRE, mesma técnica de
    test_inference_cache.py — não altera a constante de produção): MISS de
    novo, mesmo payload.

Exige Redis real (TEST_REDIS_URL, default redis://localhost:6379/15) — sem
skip (RNF-70 §16), mesma exigência de test_inference_cache.py.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, AsyncGenerator
from unittest.mock import MagicMock

import numpy as np
import pytest
import pytest_asyncio
import redis.asyncio as redis_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.config import settings
from src.core.database import Base, get_db
from src.main import create_app
from src.models.prediction import Prediction  # noqa: F401 — registers with Base
from src.schemas.predict import PredictRequest
from src.services.inference_cache import (
    InferenceCache,
    build_cache_key,
    get_inference_cache,
)
from src.services.model_service import ModelService, get_model_service

_TEST_REDIS_URL: str = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15")

_TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
_test_engine: AsyncEngine = create_async_engine(_TEST_DB_URL, echo=False)
_TestSessionFactory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=_test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

_VALID_PAYLOAD: dict = {
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


async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
    async with _TestSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@pytest_asyncio.fixture(autouse=True, scope="module", loop_scope="module")
async def _setup_db_schema() -> AsyncGenerator[None, None]:
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def _predictions_row_count() -> int:
    async with _TestSessionFactory() as session:
        result = await session.execute(select(func.count()).select_from(Prediction))
        return int(result.scalar_one())


@pytest.fixture()
def mock_service() -> tuple[ModelService, MagicMock]:
    """
    Probabilidade fixa em 0.20 — abaixo dos limiares de alerta (RF-14 em
    0.70, RF-24 em 0.85) de propósito: este arquivo testa CACHE, não
    alerta; manter a predição "sem graça" evita que AlertService toque
    Postgres (telegram_alert_locks/alert_settings) e polua a asserção de
    "nenhuma linha nova em predictions" com uma tabela diferente.
    """
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.80, 0.20]])
    service = ModelService(model=mock_model)
    return service, mock_model


@pytest.fixture()
def app_with_cache(mock_service: tuple[ModelService, MagicMock]) -> Any:
    service, _ = mock_service
    application = create_app()
    application.dependency_overrides[get_model_service] = lambda: service
    application.dependency_overrides[get_db] = _override_get_db
    return application


@pytest_asyncio.fixture()
async def real_cache() -> AsyncGenerator[InferenceCache, None]:
    client = redis_asyncio.from_url(
        _TEST_REDIS_URL, decode_responses=True, socket_connect_timeout=5.0
    )
    await client.flushdb()
    cache = InferenceCache(client)
    yield cache
    await client.flushdb()
    await client.aclose()


@pytest_asyncio.fixture()
async def async_client(
    app_with_cache: Any, real_cache: InferenceCache
) -> AsyncGenerator[AsyncClient, None]:
    app_with_cache.dependency_overrides[get_inference_cache] = lambda: real_cache
    transport = ASGITransport(app=app_with_cache)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.mark.asyncio
async def test_first_request_is_a_miss_and_persists(
    async_client: AsyncClient,
    mock_service: tuple[ModelService, MagicMock],
) -> None:
    _, mock_model = mock_service
    before = await _predictions_row_count()

    response = await async_client.post("/predict/", json=_VALID_PAYLOAD)

    assert response.status_code == 200
    mock_model.predict_proba.assert_called_once()
    assert await _predictions_row_count() == before + 1


@pytest.mark.asyncio
async def test_second_identical_request_is_a_hit_no_recompute_no_duplicate_row(
    async_client: AsyncClient,
    mock_service: tuple[ModelService, MagicMock],
) -> None:
    _, mock_model = mock_service

    first = await async_client.post("/predict/", json=_VALID_PAYLOAD)
    assert mock_model.predict_proba.call_count == 1
    rows_after_first = await _predictions_row_count()

    second = await async_client.post("/predict/", json=_VALID_PAYLOAD)

    assert second.status_code == 200
    # RNF-70 §9 — HIT: inference NÃO roda de novo.
    assert mock_model.predict_proba.call_count == 1
    # HIT: nenhuma linha nova em `predictions`.
    assert await _predictions_row_count() == rows_after_first
    # A resposta é a MESMA (mesmo timestamp — prova de que é a cópia
    # cacheada, não uma recomputação que por coincidência deu o mesmo valor).
    assert second.json() == first.json()


@pytest.mark.asyncio
async def test_different_sensor_value_is_a_separate_miss(
    async_client: AsyncClient,
    mock_service: tuple[ModelService, MagicMock],
) -> None:
    _, mock_model = mock_service

    await async_client.post("/predict/", json=_VALID_PAYLOAD)
    assert mock_model.predict_proba.call_count == 1

    different_payload = {**_VALID_PAYLOAD, "TP2": _VALID_PAYLOAD["TP2"] + 5.0}
    await async_client.post("/predict/", json=different_payload)

    # Chave diferente (sensor diferente) -> MISS -> inferência roda de novo.
    assert mock_model.predict_proba.call_count == 2


@pytest.mark.asyncio
async def test_after_ttl_expiry_same_payload_is_a_fresh_miss(
    async_client: AsyncClient,
    mock_service: tuple[ModelService, MagicMock],
    real_cache: InferenceCache,
) -> None:
    """
    Mesma técnica de test_inference_cache.py: acelera a expiração da chave
    ESPECÍFICA via PEXPIRE no Redis real (não mexe na constante de
    produção de 60s) para provar, sem um sleep(60) real, que expiração
    dispara reexecução genuína — não só teoricamente, mas observando
    `predict_proba` ser chamado de novo.
    """
    _, mock_model = mock_service

    await async_client.post("/predict/", json=_VALID_PAYLOAD)
    assert mock_model.predict_proba.call_count == 1

    # app.state.model_registry não existe neste app de teste (lifespan não
    # roda com ASGITransport) — get_active_model_name cai no fallback
    # settings.active_model (ver inference_cache.py); reproduz a MESMA
    # chave aqui para achar a entrada certa no Redis.
    key = build_cache_key(
        PredictRequest(**_VALID_PAYLOAD), model_name=settings.active_model
    )
    client = real_cache._client  # type: ignore[attr-defined]
    ttl_before = await client.ttl(key)
    assert 55 <= ttl_before <= 60

    await client.pexpire(key, 300)
    await asyncio.sleep(0.6)

    await async_client.post("/predict/", json=_VALID_PAYLOAD)
    assert mock_model.predict_proba.call_count == 2
