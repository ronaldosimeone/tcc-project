"""
Alembic environment for the PredictIQ async backend.

Imports Base + all ORM models so that autogenerate can detect schema changes.
The DATABASE_URL env var (or the fallback in alembic.ini) is used as the
SQLAlchemy engine URL.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Import ORM models — registers all tables with Base.metadata
# ---------------------------------------------------------------------------

from src.core.database import Base  # noqa: E402
import src.models  # noqa: F401, E402 — side-effect: populates Base.metadata

# ---------------------------------------------------------------------------
# Alembic config
# ---------------------------------------------------------------------------

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Override URL from environment variable when available
# ---------------------------------------------------------------------------


def _get_url() -> str:
    """Prefer DATABASE_URL env var over the alembic.ini fallback."""
    import os

    return os.environ.get("DATABASE_URL") or config.get_main_option(
        "sqlalchemy.url", ""
    )


# ---------------------------------------------------------------------------
# Offline migrations (generate SQL script without a live DB)
# ---------------------------------------------------------------------------


def run_migrations_offline() -> None:
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online migrations (apply against a live DB connection)
# ---------------------------------------------------------------------------


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Use the async engine; asyncpg requires run_sync for Alembic."""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _get_url()

    connectable = async_engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
