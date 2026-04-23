"""
Predictions history router — RF-09 / RNF-15.

Responsibilities (Clean Arch §3):
- I/O only: parse query params, inject DB session, return response.
- Zero business logic — delegate entirely to prediction_service.list_predictions().
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.schemas.prediction import Page, PredictionResponse
from src.services.prediction_service import list_predictions

router: APIRouter = APIRouter(prefix="/v1", tags=["Predictions History"])


@router.get(
    "/predictions",
    response_model=Page[PredictionResponse],
    summary="Paginated prediction history",
    description=(
        "[RF-09 / RNF-15] Returns all persisted predictions ordered by "
        "timestamp descending (most recent first).  "
        "Use `page` and `size` to paginate results."
    ),
)
async def get_predictions(
    page: int = Query(
        default=1,
        ge=1,
        description="Page number, 1-indexed.",
    ),
    size: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Number of items per page (max 100).",
    ),
    db: AsyncSession = Depends(get_db),
) -> Page[PredictionResponse]:
    """Retrieve paginated prediction history, newest first."""
    return await list_predictions(db, page, size)
