"""
FastAPI application entrypoint.

Responsibilities of this module:
- Create and configure the FastAPI instance (CORS, lifespan).
- Register all routers.
- Register global exception handlers.
- Nothing else — business logic belongs in services/.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.core.database import engine
from src.core.exceptions import AppError, app_error_handler
from src.routers import health as health_router
from src.routers import predict as predict_router
from src.routers import predictions as predictions_router
from src.services.model_service import load_model

logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage startup and shutdown of shared resources.

    Startup:
        - (Future) load ONNX models into memory as singletons.
        - (Future) warm up ChromaDB / Ollama connections.

    Shutdown:
        - Dispose the SQLAlchemy connection pool gracefully.
    """
    # ── Startup ──────────────────────────────────────────────────────────
    # [RNF-11] Load the Random Forest exactly once into app.state.
    # A missing artefact degrades /predict to HTTP 503 but keeps the API alive.
    try:
        app.state.model_service = load_model(settings.model_path)
    except FileNotFoundError:
        app.state.model_service = None
        logger.warning(
            "[RNF-11] Model artefact not found at '%s' — POST /predict will "
            "return HTTP 503 until the model is trained and placed at that path.",
            settings.model_path,
        )

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    await engine.dispose()


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

    # ── CORS ─────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global exception handlers ─────────────────────────────────────────
    app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(health_router.router)
    app.include_router(predict_router.router)
    app.include_router(predictions_router.router)

    return app


app: FastAPI = create_app()
