"""
Pydantic v2 DTOs for the drift monitoring history API — RF-27 / RNF-55.

DriftReportResponse — serialised DriftReport row returned by
                       GET /monitoring/drift.
Page[DriftReportResponse] — reaproveita o mesmo envelope genérico de
                       `schemas/prediction.py` (RNF-15), nenhuma paginação
                       nova inventada.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict


class DriftReportResponse(BaseModel):
    """Representação serializada de um relatório de drift (RF-27)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    analysis_date: date
    analyzed_at: datetime

    reference_period: str
    current_period_start: datetime
    current_period_end: datetime

    psi: float | None
    drift_detected: bool | None
    features: dict[str, float] | None

    status: str
    error_message: str | None
