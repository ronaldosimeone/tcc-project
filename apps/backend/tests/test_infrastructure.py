"""
Infrastructure tests — RNF-18 (global exception handler) and RNF-19 (rate limiting).

Test strategy
-------------
RNF-18 — Unhandled exception → safe HTTP 500
    A broken ModelService that raises RuntimeError is injected via
    dependency_overrides.  The error propagates through the route handler
    and must be caught by the generic Exception handler, returning HTTP 500
    with a JSON body that contains NO traceback or exception class names.

RNF-19 — Rate limiting → HTTP 429
    A dedicated FastAPI fixture app exposes a single endpoint protected by
    a 2/minute limit.  The limiter uses a fixed key ("test-ip") so every
    request in the test consumes the same bucket, making it trivial to
    exhaust the limit in 3 requests without slow real-time waits.
    Each test fixture creates a fresh Limiter instance to guarantee
    complete isolation between test functions.

No live PostgreSQL or trained model is required — SQLite in-memory is used
for the predict tests and the rate-limit tests bypass the entire model stack.
"""

from __future__ import annotations

from typing import AsyncGenerator

import pytest
import pytest_asyncio
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.database import Base, get_db
from src.core.rate_limit import rate_limit_exceeded_handler
from src.main import create_app
from src.models.prediction import Prediction  # noqa: F401 — registers with Base
from src.services.model_service import get_model_service

# ---------------------------------------------------------------------------
# Shared fixtures
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

_TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture()
async def db_engine() -> AsyncGenerator[AsyncEngine, None]:
    """Fresh SQLite engine per test — no PostgreSQL required."""
    engine = create_async_engine(_TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture()
def session_factory(
    db_engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )


# ---------------------------------------------------------------------------
# RNF-18 — Unhandled exception returns HTTP 500 with a safe JSON body
# ---------------------------------------------------------------------------


class _BrokenModelService:
    """Stub ModelService whose predict() always raises an unhandled error."""

    def predict(self, request: object) -> object:  # noqa: ARG002
        raise RuntimeError(
            "Simulated unexpected failure — e.g. corrupt model weights, OOM"
        )


@pytest.fixture()
def app_with_broken_service(
    session_factory: async_sessionmaker[AsyncSession],
) -> FastAPI:
    """
    App with the model dependency swapped for a broken stub.

    When the route handler calls service.predict(payload) the RuntimeError
    propagates to the global Exception handler (RNF-18).
    """
    application = create_app()
    application.dependency_overrides[get_model_service] = lambda: _BrokenModelService()

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
async def broken_client(
    app_with_broken_service: FastAPI,
) -> AsyncGenerator[AsyncClient, None]:
    # raise_app_exceptions=False is required here because Starlette's
    # ServerErrorMiddleware ALWAYS re-raises the exception after invoking
    # the registered handler (by design, so that Uvicorn can log it).
    # With the default raise_app_exceptions=True, httpx would propagate that
    # re-raise into the test, making it impossible to inspect the HTTP response.
    # Our unhandled_exception_handler still runs and returns HTTP 500 before
    # the re-raise — with raise_app_exceptions=False httpx returns that response.
    # Dentro de tests/test_infrastructure.py
    transport = ASGITransport(app=app_with_broken_service, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


# ── RNF-18 tests ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rnf18_unhandled_exception_returns_500(
    broken_client: AsyncClient,
) -> None:
    """Any unhandled runtime exception must produce HTTP 500 (RNF-18)."""
    response = await broken_client.post("/predict/", json=_VALID_PAYLOAD)
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_rnf18_500_body_has_error_key(
    broken_client: AsyncClient,
) -> None:
    """HTTP 500 body must contain the 'error' key with value 'InternalServerError'."""
    response = await broken_client.post("/predict/", json=_VALID_PAYLOAD)
    body: dict = response.json()

    assert "error" in body, "Missing 'error' key in 500 response body"
    assert body["error"] == "InternalServerError"


@pytest.mark.asyncio
async def test_rnf18_500_body_has_detail_key(
    broken_client: AsyncClient,
) -> None:
    """HTTP 500 body must contain a 'detail' key with a user-safe message."""
    response = await broken_client.post("/predict/", json=_VALID_PAYLOAD)
    body: dict = response.json()

    assert "detail" in body, "Missing 'detail' key in 500 response body"
    assert isinstance(body["detail"], str)
    assert len(body["detail"]) > 0


@pytest.mark.asyncio
async def test_rnf18_500_body_has_no_traceback(
    broken_client: AsyncClient,
) -> None:
    """
    RNF-18 security requirement: stack traces must NEVER appear in the response.

    An attacker who can read stack traces learns file paths, library versions
    and internal variable names.
    """
    response = await broken_client.post("/predict/", json=_VALID_PAYLOAD)
    body_text: str = response.text

    assert "Traceback" not in body_text, "Stack trace leaked into HTTP 500 response"
    assert "traceback" not in body_text
    assert "File " not in body_text  # 'File "src/services/..."' line in tracebacks


@pytest.mark.asyncio
async def test_rnf18_500_body_has_no_exception_class(
    broken_client: AsyncClient,
) -> None:
    """
    RNF-18: The exception type name must not appear in the response body.

    Exposing 'RuntimeError' or 'sqlalchemy.exc.OperationalError' reveals
    internals that aid vulnerability research.
    """
    response = await broken_client.post("/predict/", json=_VALID_PAYLOAD)
    body_text: str = response.text

    assert "RuntimeError" not in body_text
    assert "Exception" not in body_text  # generic Python exception names
    assert "Error" not in body_text.replace(
        "InternalServerError", ""
    ), "A raw exception class name leaked into the 500 response"


@pytest.mark.asyncio
async def test_rnf18_500_content_type_is_json(
    broken_client: AsyncClient,
) -> None:
    """HTTP 500 response must be application/json so clients can parse it."""
    response = await broken_client.post("/predict/", json=_VALID_PAYLOAD)
    assert "application/json" in response.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# RNF-19 — Rate limiting returns HTTP 429 with a JSON body
# ---------------------------------------------------------------------------


@pytest.fixture()
def rate_limited_app() -> FastAPI:
    """
    Minimal FastAPI app with a 2/minute limit on a test endpoint.

    Uses a fixed key function ("test-ip") so every request in the test suite
    hits the same bucket — making it trivial to exhaust the limit in 3 requests
    without any real-time delay.

    A fresh Limiter instance is created per fixture invocation (function scope)
    so that no limit state leaks between test functions.
    """
    # Fresh limiter — clean in-memory storage for this test.
    # The key_func parameter MUST be named "request" (not "_request").
    # slowapi uses inspect.signature() to detect this name; if absent it
    # calls key_func() with no arguments, which would raise TypeError.
    test_limiter = Limiter(
        key_func=lambda request: "test-ip",  # noqa: ARG005 — unused but name is required
        default_limits=[],
    )

    app = FastAPI()
    app.state.limiter = test_limiter
    app.add_middleware(SlowAPIMiddleware)
    app.add_exception_handler(  # type: ignore[arg-type]
        RateLimitExceeded,
        rate_limit_exceeded_handler,
    )

    @app.get("/limited")
    @test_limiter.limit("2/minute")
    async def _limited_endpoint(request: Request) -> dict:  # noqa: RUF029
        return {"status": "ok"}

    return app


@pytest_asyncio.fixture()
async def rate_client(
    rate_limited_app: FastAPI,
) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=rate_limited_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


# ── RNF-19 helpers ───────────────────────────────────────────────────────


async def _exhaust_limit(client: AsyncClient, path: str = "/limited") -> None:
    """Consume the 2/minute budget (2 requests)."""
    await client.get(path)
    await client.get(path)


# ── RNF-19 tests ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rnf19_requests_within_limit_return_200(
    rate_client: AsyncClient,
) -> None:
    """Requests within the rate limit budget must return HTTP 200."""
    r1 = await rate_client.get("/limited")
    r2 = await rate_client.get("/limited")

    assert r1.status_code == 200
    assert r2.status_code == 200


@pytest.mark.asyncio
async def test_rnf19_excess_request_returns_429(
    rate_client: AsyncClient,
) -> None:
    """The request that exceeds the budget must return HTTP 429 (RNF-19)."""
    await _exhaust_limit(rate_client)
    response = await rate_client.get("/limited")

    assert response.status_code == 429


@pytest.mark.asyncio
async def test_rnf19_429_body_has_error_key(
    rate_client: AsyncClient,
) -> None:
    """HTTP 429 body must contain 'error': 'RateLimitExceeded'."""
    await _exhaust_limit(rate_client)
    body: dict = (await rate_client.get("/limited")).json()

    assert "error" in body
    assert body["error"] == "RateLimitExceeded"


@pytest.mark.asyncio
async def test_rnf19_429_body_has_detail_key(
    rate_client: AsyncClient,
) -> None:
    """HTTP 429 body must contain a 'detail' key with a user-safe message."""
    await _exhaust_limit(rate_client)
    body: dict = (await rate_client.get("/limited")).json()

    assert "detail" in body
    assert isinstance(body["detail"], str)
    assert len(body["detail"]) > 0


@pytest.mark.asyncio
async def test_rnf19_429_has_retry_after_header(
    rate_client: AsyncClient,
) -> None:
    """HTTP 429 must include Retry-After so clients know when to back off."""
    await _exhaust_limit(rate_client)
    response = await rate_client.get("/limited")

    assert "retry-after" in {k.lower() for k in response.headers}


@pytest.mark.asyncio
async def test_rnf19_429_content_type_is_json(
    rate_client: AsyncClient,
) -> None:
    """HTTP 429 response must be application/json (consistent with all API errors)."""
    await _exhaust_limit(rate_client)
    response = await rate_client.get("/limited")

    assert "application/json" in response.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_rnf19_429_body_has_no_internal_details(
    rate_client: AsyncClient,
) -> None:
    """
    HTTP 429 body must not expose internal rate limit configuration
    (e.g. exact limit strings like '2 per 1 minute' or bucket keys).
    """
    await _exhaust_limit(rate_client)
    body_text: str = (await rate_client.get("/limited")).text

    assert "test-ip" not in body_text  # key function result must not leak
    assert "per 1 minute" not in body_text.lower()  # internal limit string
