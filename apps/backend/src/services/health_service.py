"""
Health service.

Contains the business logic for the health-check: probing the database
with a lightweight `SELECT 1` query and measuring round-trip latency.
"""

from __future__ import annotations

import time

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.schemas.health import DatabaseStatus, HealthResponse


async def check_health(db: AsyncSession) -> HealthResponse:
    """
    Probe the database and return a fully populated HealthResponse.

    The DB probe executes `SELECT 1` and measures wall-clock latency.
    Any exception is caught and surfaced as a non-connected status
    (the API itself remains operational).
    """
    db_status: DatabaseStatus = await _probe_database(db)

    overall_status: str = "ok" if db_status.connected else "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.version,
        database=db_status,
    )


async def _probe_database(db: AsyncSession) -> DatabaseStatus:
    """Execute `SELECT 1` and return a DatabaseStatus with timing."""
    start: float = time.perf_counter()
    try:
        await db.execute(text("SELECT 1"))
        latency_ms: float = round((time.perf_counter() - start) * 1_000, 2)
        return DatabaseStatus(connected=True, latency_ms=latency_ms)
    except Exception as exc:  # noqa: BLE001
        latency_ms = round((time.perf_counter() - start) * 1_000, 2)
        return DatabaseStatus(
            connected=False,
            latency_ms=latency_ms,
            error=str(exc),
        )
