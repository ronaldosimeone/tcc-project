"""
Testes unitários de `prediction_service` (RF-09 / RNF-15, RNF-64/RNF-65).

`test_predictions_endpoint.py` já cobre o fluxo via HTTP/router; este
arquivo testa `save_prediction`/`list_predictions` DIRETAMENTE contra um
SQLite real (sem FastAPI/HTTP no meio) — isola a lógica de paginação
(offset, total==0, ordenação) da camada de transporte, e cobre
especificamente os ramos que o coverage apontou como não exercitados
(`total == 0` early-return, cálculo de `offset`, `return record`).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.database import Base
from src.schemas.predict import PredictRequest, PredictResponse
from src.services.prediction_service import list_predictions, save_prediction

_NOW = datetime(2026, 9, 18, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
async def db_engine() -> AsyncGenerator[AsyncEngine, None]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture()
def session_factory(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=db_engine, class_=AsyncSession, expire_on_commit=False
    )


def _request(**overrides) -> PredictRequest:
    kwargs = dict(
        TP2=8.4,
        TP3=9.1,
        H1=8.5,
        DV_pressure=2.1,
        Reservoirs=8.7,
        Motor_current=4.2,
        Oil_temperature=68.5,
        COMP=1.0,
        DV_eletric=0.0,
        Towers=1.0,
        MPG=1.0,
        Oil_level=1.0,
    )
    kwargs.update(overrides)
    return PredictRequest(**kwargs)


def _response(
    *, timestamp: datetime, predicted_class: int = 0, prob: float = 0.1
) -> PredictResponse:
    return PredictResponse(
        predicted_class=predicted_class,
        failure_probability=prob,
        timestamp=timestamp.isoformat(),
    )


# ---------------------------------------------------------------------------
# save_prediction
# ---------------------------------------------------------------------------


async def test_save_prediction_persists_and_returns_the_record(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as db:
        record = await save_prediction(
            db,
            _request(TP2=99.9),
            _response(timestamp=_NOW, predicted_class=1, prob=0.95),
        )
        await db.commit()

        assert record.id is not None  # flush já atribuiu o id
        assert record.TP2 == 99.9
        assert record.predicted_class == 1
        assert record.failure_probability == 0.95
        assert record.timestamp == _NOW


async def test_save_prediction_timestamp_matches_response_exactly(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """O timestamp gravado vem de `response.timestamp` (parseado), não de
    `datetime.now()` no momento da persistência — garante que o valor
    devolvido ao cliente é EXATAMENTE o que fica no banco."""
    ts = _NOW - timedelta(hours=3)
    async with session_factory() as db:
        record = await save_prediction(db, _request(), _response(timestamp=ts))
        assert record.timestamp == ts


# ---------------------------------------------------------------------------
# list_predictions — total==0, paginação, ordenação
# ---------------------------------------------------------------------------


async def test_list_predictions_empty_database_returns_empty_page_without_querying_rows(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as db:
        page = await list_predictions(db, page=1, size=20)

    assert page.total == 0
    assert page.items == []
    assert page.pages == 0
    assert page.page == 1
    assert page.size == 20


async def _seed(factory: async_sessionmaker[AsyncSession], count: int) -> None:
    async with factory() as db:
        for i in range(count):
            await save_prediction(
                db,
                _request(),
                _response(timestamp=_NOW - timedelta(minutes=i)),
            )
        await db.commit()


async def test_list_predictions_computes_offset_from_page_and_size(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await _seed(session_factory, 10)
    async with session_factory() as db:
        page1 = await list_predictions(db, page=1, size=4)
        page2 = await list_predictions(db, page=2, size=4)
        page3 = await list_predictions(db, page=3, size=4)

    assert page1.total == 10
    assert len(page1.items) == 4
    assert len(page2.items) == 4
    assert len(page3.items) == 2  # offset=8, resta só 2 registros
    # Nenhum item repetido entre páginas (offset calculado corretamente).
    ids_p1 = {i.id for i in page1.items}
    ids_p2 = {i.id for i in page2.items}
    assert ids_p1.isdisjoint(ids_p2)


async def test_list_predictions_orders_newest_timestamp_first(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as db:
        await save_prediction(
            db, _request(), _response(timestamp=_NOW - timedelta(hours=2))
        )
        await save_prediction(db, _request(), _response(timestamp=_NOW))
        await save_prediction(
            db, _request(), _response(timestamp=_NOW - timedelta(hours=1))
        )
        await db.commit()

        page = await list_predictions(db, page=1, size=10)

    timestamps = [item.timestamp for item in page.items]
    assert timestamps == sorted(timestamps, reverse=True)


async def test_list_predictions_page_beyond_total_returns_empty_items_not_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await _seed(session_factory, 3)
    async with session_factory() as db:
        page = await list_predictions(db, page=99, size=10)

    assert page.total == 3
    assert page.items == []
