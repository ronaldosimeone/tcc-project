"""
Integration tests for GET /health.

Strategy
--------
The real database engine is replaced with an in-memory SQLite engine so the
tests run without a live PostgreSQL instance.  We override the `get_db`
dependency to yield a session backed by that engine.

Fixtures:
    app_with_mock_db – FastAPI test app with the DB dependency overridden.
    async_client     – httpx.AsyncClient bound to the test app.
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

# ---------------------------------------------------------------------------
# In-memory SQLite engine (no PostgreSQL required)
# ---------------------------------------------------------------------------

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
    """Dependency override that yields a SQLite-backed async session."""
    async with _TestSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app_with_mock_db():
    """Create a fresh app instance with the DB dependency overridden."""
    application = create_app()
    application.dependency_overrides[get_db] = _override_get_db
    return application


@pytest_asyncio.fixture()
async def async_client(app_with_mock_db):
    """Async HTTP client bound to the test app via ASGI transport."""
    transport = ASGITransport(app=app_with_mock_db)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_returns_200(async_client: AsyncClient) -> None:
    """GET /health must return HTTP 200."""
    response = await async_client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_response_schema(async_client: AsyncClient) -> None:
    """Response body must contain the required top-level keys."""
    response = await async_client.get("/health")
    body: dict = response.json()

    assert "status" in body, "Missing 'status' key"
    assert "version" in body, "Missing 'version' key"
    assert "database" in body, "Missing 'database' key"


@pytest.mark.asyncio
async def test_health_status_is_ok_when_db_connected(async_client: AsyncClient) -> None:
    """When SQLite responds to SELECT 1, status must be 'ok'."""
    response = await async_client.get("/health")
    body: dict = response.json()

    assert body["status"] == "ok"
    assert body["database"]["connected"] is True


@pytest.mark.asyncio
async def test_health_database_latency_is_positive(async_client: AsyncClient) -> None:
    """Reported DB latency must be a non-negative float."""
    response = await async_client.get("/health")
    latency = response.json()["database"]["latency_ms"]

    assert latency is not None
    assert isinstance(latency, float)
    assert latency >= 0.0


@pytest.mark.asyncio
async def test_health_version_matches_settings(async_client: AsyncClient) -> None:
    """Reported version must match the value in Settings."""
    from src.core.config import settings

    response = await async_client.get("/health")
    assert response.json()["version"] == settings.version


@pytest.mark.asyncio
async def test_health_degraded_when_db_unreachable() -> None:
    """
    When the DB dependency raises an exception, status must be 'degraded'
    and database.connected must be False.
    """

    async def _broken_get_db():  # type: ignore[return]
        raise ConnectionRefusedError("Simulated DB failure")
        yield  # make it a generator

    broken_app = create_app()

    async def _broken_db_override():  # type: ignore[return]
        """Override that simulates a failed DB session."""
        from sqlalchemy.ext.asyncio import AsyncSession
        from unittest.mock import AsyncMock, MagicMock

        mock_session = MagicMock(spec=AsyncSession)
        mock_session.execute = AsyncMock(
            side_effect=ConnectionRefusedError("Simulated DB failure")
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        yield mock_session

    broken_app.dependency_overrides[get_db] = _broken_db_override

    transport = ASGITransport(app=broken_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body: dict = response.json()
    assert body["status"] == "degraded"
    assert body["database"]["connected"] is False
    assert body["database"]["error"] is not None
