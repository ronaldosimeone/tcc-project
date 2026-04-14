"""
Async SQLAlchemy engine, session factory, and declarative base.

Usage in routers:
    from src.core.database import get_db
    ...
    async def my_route(db: AsyncSession = Depends(get_db)) -> ...:
        ...
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from src.core.config import settings


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

engine: AsyncEngine = create_async_engine(
    str(settings.postgres_url),
    echo=settings.debug,
    pool_pre_ping=True,  # verify connections before use
    pool_size=10,
    max_overflow=20,
)

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

AsyncSessionFactory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # avoids implicit I/O after commit
    autoflush=False,
    autocommit=False,
)


# ---------------------------------------------------------------------------
# Declarative base  (all SQLAlchemy models inherit from this)
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------


async def get_db() -> AsyncGenerator[AsyncSession, Any]:
    """
    FastAPI dependency that yields an async database session.

    The session is automatically closed (and rolled back on error)
    after the request completes.
    """
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
