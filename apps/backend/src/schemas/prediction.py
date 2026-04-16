"""
Pydantic v2 DTOs for the prediction history API — RF-09 / RNF-15.

PredictionResponse  — serialised ORM row returned by GET /api/v1/predictions.
Page[T]             — generic paginated envelope (items, total, page, size, pages).
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class PredictionResponse(BaseModel):
    """
    Serialised representation of a single persisted prediction row — RF-09.

    The sensor field names intentionally mirror PredictRequest so that
    clients can cross-reference request payloads with stored history.
    """

    model_config = ConfigDict(
        from_attributes=True,  # enables ORM → Pydantic mapping
    )

    id: int
    timestamp: datetime

    # ── Raw sensor inputs (12 features) ──────────────────────────────────
    TP2: float
    TP3: float
    H1: float
    DV_pressure: float
    Reservoirs: float
    Motor_current: float
    Oil_temperature: float
    COMP: float
    DV_eletric: float
    Towers: float
    MPG: float
    Oil_level: float

    # ── ML outputs ────────────────────────────────────────────────────────
    predicted_class: int
    failure_probability: float = Field(ge=0.0, le=1.0)


class Page(BaseModel, Generic[T]):
    """
    Generic paginated response envelope — RNF-15.

    items : Records for the requested page.
    total : Total matching records in the database.
    page  : Current page (1-indexed).
    size  : Items per page (as requested).
    pages : Total number of pages — 0 when database is empty.
    """

    items: list[T]
    total: int = Field(ge=0)
    page: int = Field(ge=1)
    size: int = Field(ge=1)
    pages: int = Field(ge=0)


def make_page(
    items: list[T],
    total: int,
    page: int,
    size: int,
) -> Page[T]:
    """Build a Page envelope from query results and pagination parameters."""
    pages = math.ceil(total / size) if total > 0 else 0
    return Page(items=items, total=total, page=page, size=size, pages=pages)
