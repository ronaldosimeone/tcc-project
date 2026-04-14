"""
FastAPI application entrypoint.

Responsibilities of this module:
- Create and configure the FastAPI instance (CORS, lifespan).
- Register all routers.
- Register global exception handlers.
- Nothing else — business logic belongs in services/.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.core.database import engine
from src.core.exceptions import AppError, app_error_handler
from src.routers import health as health_router


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
    # ONNX singleton will be loaded here in a future sprint (CLAUDE.md §4).
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

    return app


app: FastAPI = create_app()
