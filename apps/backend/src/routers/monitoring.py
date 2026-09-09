"""
Drift monitoring router — RF-27 / RNF-55.

Responsibilities (Clean Arch §3):
- I/O only: parse query params, inject DB session, return response.
- Zero business logic — delegate entirely to
  `drift_monitor.list_drift_reports()`.
- NUNCA dispara a análise (RF-27 §Fase 7/9): este router só LÊ o histórico
  já persistido pela task Celery Beat diária (`src/tasks/drift_tasks.py`).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.schemas.drift import DriftReportResponse
from src.schemas.prediction import Page
from src.services.drift_monitor import list_drift_reports

router: APIRouter = APIRouter(prefix="/monitoring", tags=["Drift Monitoring"])


@router.get(
    "/drift",
    response_model=Page[DriftReportResponse],
    summary="Histórico de relatórios de data drift (PSI)",
    description=(
        "[RF-27 / RNF-55] Retorna o histórico de análises diárias de data "
        "drift (Evidently/PSI sobre as features de entrada do modelo), "
        "mais recente primeiro. Este endpoint NUNCA dispara a análise — "
        "ela roda em background via Celery Beat, uma vez por dia."
    ),
)
async def get_drift_history(
    page: int = Query(default=1, ge=1, description="Page number, 1-indexed."),
    size: int = Query(
        default=20, ge=1, le=100, description="Number of items per page (max 100)."
    ),
    db: AsyncSession = Depends(get_db),
) -> Page[DriftReportResponse]:
    """Histórico paginado de relatórios de drift, mais recente primeiro."""
    return await list_drift_reports(db, page, size)
