"""
Health-check router.

Responsibilities:
- Expose GET /health for liveness/readiness probes.
- Delegate DB connectivity check to the health service via Depends.

No business logic lives here (Clean Arch rule from CLAUDE.md §3).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.schemas.health import HealthResponse
from src.services.health_service import check_health

router: APIRouter = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness & readiness probe",
    description="Returns API status and a live database connectivity check.",
)
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    """Return a structured health response with DB probe results."""
    return await check_health(db)
