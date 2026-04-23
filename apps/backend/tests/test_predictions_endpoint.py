"""
Integration tests for RF-09 / RNF-15.

Tests
-----
POST /predict
    - Persists a record in the database after a successful inference.
    - DB record has correct sensor values and ML outputs.

GET /v1/predictions
    - Returns HTTP 200 with the correct Page envelope schema.
    - Returns an empty list when the database has no records.
    - Returns paginated results (total, page, size, pages).
    - Respects the `page` and `size` query parameters.
    - Orders results newest-first (descending timestamp).
    - Rejects page < 1 with HTTP 422.
    - Rejects size < 1 with HTTP 422.
    - Rejects size > 100 with HTTP 422.
    - Returns an empty page when page exceeds total pages.

Strategy
--------
Each test receives a fresh SQLite in-memory database (function scope) for
complete isolation.  The model dependency is overridden with a MagicMock-
backed ModelService, and get_db is overridden with a SQLite async session.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator
from unittest.mock import MagicMock

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.database import Base, get_db
from src.main import create_app
from src.models.prediction import Prediction  # noqa: F401 — registers with Base
from src.services.model_service import ModelService, get_model_service

# ---------------------------------------------------------------------------
# Shared test payload
# ---------------------------------------------------------------------------

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

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_service() -> ModelService:
    """ModelService backed by a MagicMock — returns class 1 at 0.85 probability."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.15, 0.85]])
    return ModelService(model=mock_model)


@pytest_asyncio.fixture()
async def db_engine() -> AsyncGenerator[AsyncEngine, None]:
    """Fresh in-memory SQLite engine with the full schema per test."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture()
def session_factory(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )


@pytest.fixture()
def test_app(
    mock_service: ModelService, session_factory: async_sessionmaker[AsyncSession]
):
    """FastAPI app with both model and DB dependencies overridden."""
    application = create_app()
    application.dependency_overrides[get_model_service] = lambda: mock_service

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    application.dependency_overrides[get_db] = _override_get_db
    return application


@pytest_asyncio.fixture()
async def client(test_app) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client bound to the test app."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


# ---------------------------------------------------------------------------
# Helper: seed N predictions directly into the test DB
# ---------------------------------------------------------------------------


async def _seed_predictions(
    factory: async_sessionmaker[AsyncSession],
    count: int,
    base_dt: datetime | None = None,
) -> list[Prediction]:
    """Insert `count` prediction rows spaced 1 second apart (oldest first)."""
    now = base_dt or datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    records: list[Prediction] = []

    async with factory() as session:
        for i in range(count):
            rec = Prediction(
                timestamp=now + timedelta(seconds=i),
                TP2=float(i),
                TP3=9.0,
                H1=8.0,
                DV_pressure=2.0,
                Reservoirs=8.0,
                Motor_current=4.0,
                Oil_temperature=70.0,
                COMP=1.0,
                DV_eletric=0.0,
                Towers=1.0,
                MPG=1.0,
                Oil_level=1.0,
                predicted_class=i % 2,
                failure_probability=round(0.1 * (i % 10 + 1), 1),
            )
            session.add(rec)
            records.append(rec)
        await session.commit()

    return records


# ---------------------------------------------------------------------------
# POST /predict — persistence (RF-09)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_predict_persists_record(
    client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """RF-09: POST /predict must save exactly one row in the predictions table."""
    response = await client.post("/predict/", json=_VALID_PAYLOAD)
    assert response.status_code == 200, response.text

    async with session_factory() as session:
        from sqlalchemy import select

        result = (await session.execute(select(Prediction))).scalars().all()

    assert len(result) == 1


@pytest.mark.asyncio
async def test_post_predict_persists_sensor_values(
    client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """RF-09: The persisted row must carry the exact sensor values from the request."""
    await client.post("/predict/", json=_VALID_PAYLOAD)

    async with session_factory() as session:
        from sqlalchemy import select

        row: Prediction = (await session.execute(select(Prediction))).scalars().first()

    assert row is not None
    assert row.TP2 == pytest.approx(_VALID_PAYLOAD["TP2"])
    assert row.TP3 == pytest.approx(_VALID_PAYLOAD["TP3"])
    assert row.Motor_current == pytest.approx(_VALID_PAYLOAD["Motor_current"])
    assert row.Oil_temperature == pytest.approx(_VALID_PAYLOAD["Oil_temperature"])
    assert row.COMP == pytest.approx(_VALID_PAYLOAD["COMP"])


@pytest.mark.asyncio
async def test_post_predict_persists_ml_outputs(
    client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """RF-09: The persisted row must carry predicted_class and failure_probability."""
    response = await client.post("/predict/", json=_VALID_PAYLOAD)
    body = response.json()

    async with session_factory() as session:
        from sqlalchemy import select

        row: Prediction = (await session.execute(select(Prediction))).scalars().first()

    assert row is not None
    assert row.predicted_class == body["predicted_class"]
    assert row.failure_probability == pytest.approx(
        body["failure_probability"], abs=1e-5
    )


@pytest.mark.asyncio
async def test_post_predict_persists_timestamp(
    client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """RF-09: The DB timestamp must match the timestamp returned to the client."""
    response = await client.post("/predict/", json=_VALID_PAYLOAD)
    client_ts = datetime.fromisoformat(response.json()["timestamp"])

    async with session_factory() as session:
        from sqlalchemy import select

        row: Prediction = (await session.execute(select(Prediction))).scalars().first()

    assert row is not None
    # Compare as UTC-aware — replace tzinfo if stored as naive by SQLite
    stored_ts = row.timestamp
    if stored_ts.tzinfo is None:
        stored_ts = stored_ts.replace(tzinfo=timezone.utc)
    if client_ts.tzinfo is None:
        client_ts = client_ts.replace(tzinfo=timezone.utc)

    assert abs((stored_ts - client_ts).total_seconds()) < 1.0


# ---------------------------------------------------------------------------
# GET /v1/predictions — empty database
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_predictions_empty_database(client: AsyncClient) -> None:
    """Empty DB must return items=[], total=0, pages=0 (not 404)."""
    response = await client.get("/v1/predictions")
    assert response.status_code == 200
    body = response.json()

    assert body["items"] == []
    assert body["total"] == 0
    assert body["pages"] == 0


# ---------------------------------------------------------------------------
# GET /v1/predictions — schema validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_predictions_envelope_schema(
    client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Response must contain all Page envelope fields with correct types."""
    await _seed_predictions(session_factory, count=3)
    response = await client.get("/v1/predictions")
    assert response.status_code == 200
    body = response.json()

    assert "items" in body
    assert "total" in body
    assert "page" in body
    assert "size" in body
    assert "pages" in body
    assert isinstance(body["items"], list)
    assert isinstance(body["total"], int)
    assert isinstance(body["pages"], int)


@pytest.mark.asyncio
async def test_get_predictions_item_schema(
    client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Each item in the response must contain id, timestamp, sensors and ML outputs."""
    await _seed_predictions(session_factory, count=1)
    response = await client.get("/v1/predictions")
    body = response.json()

    item = body["items"][0]
    required_keys = {
        "id",
        "timestamp",
        "TP2",
        "TP3",
        "H1",
        "DV_pressure",
        "Reservoirs",
        "Motor_current",
        "Oil_temperature",
        "COMP",
        "DV_eletric",
        "Towers",
        "MPG",
        "Oil_level",
        "predicted_class",
        "failure_probability",
    }
    assert required_keys.issubset(
        item.keys()
    ), f"Missing keys: {required_keys - item.keys()}"


# ---------------------------------------------------------------------------
# GET /v1/predictions — total and pagination metadata (RNF-15)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_predictions_total_matches_db(
    client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """RNF-15: `total` must equal the number of rows in the database."""
    await _seed_predictions(session_factory, count=7)
    body = (await client.get("/v1/predictions")).json()
    assert body["total"] == 7


@pytest.mark.asyncio
async def test_get_predictions_default_page_and_size(
    client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Default page=1, size=20 — must return all items when count <= 20."""
    await _seed_predictions(session_factory, count=5)
    body = (await client.get("/v1/predictions")).json()

    assert body["page"] == 1
    assert body["size"] == 20
    assert len(body["items"]) == 5


@pytest.mark.asyncio
async def test_get_predictions_custom_size(
    client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """RNF-15: `size` query param controls items per page."""
    await _seed_predictions(session_factory, count=10)
    body = (await client.get("/v1/predictions?size=3")).json()

    assert len(body["items"]) == 3
    assert body["size"] == 3


@pytest.mark.asyncio
async def test_get_predictions_pages_computed_correctly(
    client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """RNF-15: `pages` must equal ceil(total / size)."""
    await _seed_predictions(session_factory, count=7)
    body = (await client.get("/v1/predictions?size=3")).json()

    # ceil(7 / 3) = 3
    assert body["total"] == 7
    assert body["pages"] == 3


@pytest.mark.asyncio
async def test_get_predictions_second_page(
    client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """RNF-15: page=2 must return the next slice of records."""
    await _seed_predictions(session_factory, count=5)

    page1 = (await client.get("/v1/predictions?size=3&page=1")).json()
    page2 = (await client.get("/v1/predictions?size=3&page=2")).json()

    ids_page1 = {item["id"] for item in page1["items"]}
    ids_page2 = {item["id"] for item in page2["items"]}

    # No overlap between pages
    assert ids_page1.isdisjoint(ids_page2)
    # Combined they cover all 5 records
    assert len(ids_page1 | ids_page2) == 5


@pytest.mark.asyncio
async def test_get_predictions_page_beyond_total(
    client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """RNF-15: Requesting a page beyond total pages must return empty items (not 404)."""
    await _seed_predictions(session_factory, count=3)
    body = (await client.get("/v1/predictions?size=3&page=99")).json()

    assert body["items"] == []
    assert body["total"] == 3  # total is still reported correctly


# ---------------------------------------------------------------------------
# GET /v1/predictions — ordering (newest first)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_predictions_ordered_newest_first(
    client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Results must be ordered by timestamp descending (most recent first)."""
    base = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    await _seed_predictions(session_factory, count=5, base_dt=base)

    body = (await client.get("/v1/predictions?size=5")).json()
    timestamps = [item["timestamp"] for item in body["items"]]

    # Each successive timestamp must be <= the previous one
    for i in range(len(timestamps) - 1):
        assert (
            timestamps[i] >= timestamps[i + 1]
        ), f"Order violation at index {i}: {timestamps[i]} < {timestamps[i + 1]}"


# ---------------------------------------------------------------------------
# GET /v1/predictions — input validation (RNF-15)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_predictions_page_zero_returns_422(client: AsyncClient) -> None:
    """page=0 must be rejected with HTTP 422 (ge=1 constraint)."""
    response = await client.get("/v1/predictions?page=0")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_predictions_size_zero_returns_422(client: AsyncClient) -> None:
    """size=0 must be rejected with HTTP 422 (ge=1 constraint)."""
    response = await client.get("/v1/predictions?size=0")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_predictions_size_over_max_returns_422(client: AsyncClient) -> None:
    """size=101 must be rejected with HTTP 422 (le=100 constraint)."""
    response = await client.get("/v1/predictions?size=101")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_predictions_size_at_max_returns_200(
    client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """size=100 is the allowed maximum and must return HTTP 200."""
    await _seed_predictions(session_factory, count=2)
    response = await client.get("/v1/predictions?size=100")
    assert response.status_code == 200
    assert response.json()["size"] == 100
