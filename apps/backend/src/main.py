"""
FastAPI application entrypoint.

Responsibilities of this module:
- Configure structured logging (structlog) before any I/O.
- Create and configure the FastAPI instance (CORS, lifespan).
- Register all routers.
- Register global exception handlers (domain errors, rate limits, catch-all).
- Nothing else — business logic belongs in services/.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.core.config import settings
from src.core.database import engine
from src.core.exceptions import (
    AppError,
    app_error_handler,
    unhandled_exception_handler,
)
from src.core.logging import configure_logging
from src.core.rate_limit import limiter, rate_limit_exceeded_handler
from src.routers import health as health_router
from src.routers import predict as predict_router
from src.routers import predictions as predictions_router
from src.services.model_service import load_model

# ── Bootstrap structured logging immediately ──────────────────────────────
# Must happen before any log.* call so processors are fully configured.
configure_logging(debug=settings.debug)

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage startup and shutdown of shared resources.

    Startup:
        - Configure the rate limiter on app.state.
        - Load the Random Forest model into memory as a singleton (RNF-11).

    Shutdown:
        - Dispose the SQLAlchemy async connection pool gracefully.
    """
    # ── Startup ──────────────────────────────────────────────────────────
    app.state.limiter = limiter

    try:
        app.state.model_service = load_model(settings.model_path)
        log.info("model_loaded", path=str(settings.model_path))
    except FileNotFoundError:
        app.state.model_service = None
        log.warning(
            "model_artefact_missing",
            path=str(settings.model_path),
            consequence="POST /predict will return HTTP 503 until the model is available",
        )

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    await engine.dispose()
    log.info("db_pool_disposed")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Construct and fully configure the FastAPI application."""

    app: FastAPI = FastAPI(
        title=settings.project_name,
        version=settings.version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Rate limiter — must be on app.state BEFORE the first request ────────
    # SlowAPIMiddleware reads app.state.limiter on every request.  We set it
    # here (in create_app) rather than relying solely on the lifespan, because
    # httpx ASGITransport used in tests does not trigger the ASGI lifespan.
    # Setting it twice (here + lifespan) is idempotent and harmless.
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)

    # ── CORS ─────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception handlers — ordered from specific to generic ─────────────
    # FastAPI walks the exception MRO, so subclasses (AppError) always win
    # over the generic Exception catch-all, regardless of registration order.
    # We register explicitly for clarity and documentation purposes.
    app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)  # type: ignore[arg-type]
    # RNF-18: catch-all — any unhandled exception returns HTTP 500 with a
    # safe JSON body; the full traceback is written to structured logs only.
    app.add_exception_handler(Exception, unhandled_exception_handler)  # type: ignore[arg-type]

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(health_router.router)
    app.include_router(predict_router.router)
    app.include_router(predictions_router.router)

    return app


app: FastAPI = create_app()
