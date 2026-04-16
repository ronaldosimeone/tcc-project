"""
PredictionService — RF-09 / RNF-15.

All persistence and retrieval logic for the predictions table lives here.
No HTTP or FastAPI concerns — those belong in routers/.

Public API
----------
save_prediction(db, request, response)  → Prediction
    Persist the result of a single inference cycle.

list_predictions(db, page, size)        → Page[PredictionResponse]
    Return paginated history ordered newest-first.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.prediction import Prediction
from src.schemas.predict import PredictRequest, PredictResponse
from src.schemas.prediction import Page, PredictionResponse, make_page


async def save_prediction(
    db: AsyncSession,
    request: PredictRequest,
    response: PredictResponse,
) -> Prediction:
    """
    Persist the completed inference to the database — RF-09.

    The timestamp is parsed from PredictResponse to guarantee that the
    value stored in the DB matches the value returned to the client exactly.

    The caller (predict router via get_db) owns the transaction boundary;
    this function issues a flush to acquire the auto-generated id without
    committing — commit happens when the request scope closes.
    """
    record = Prediction(
        timestamp=datetime.fromisoformat(response.timestamp),
        TP2=request.TP2,
        TP3=request.TP3,
        H1=request.H1,
        DV_pressure=request.DV_pressure,
        Reservoirs=request.Reservoirs,
        Motor_current=request.Motor_current,
        Oil_temperature=request.Oil_temperature,
        COMP=request.COMP,
        DV_eletric=request.DV_eletric,
        Towers=request.Towers,
        MPG=request.MPG,
        Oil_level=request.Oil_level,
        predicted_class=response.predicted_class,
        failure_probability=response.failure_probability,
    )
    db.add(record)
    # flush → assigns id; commit handled by get_db dependency
    await db.flush()
    return record


async def list_predictions(
    db: AsyncSession,
    page: int,
    size: int,
) -> Page[PredictionResponse]:
    """
    Return a page of predictions ordered by timestamp descending — RNF-15.

    Parameters
    ----------
    page : 1-indexed page number (validated by the router query param ge=1).
    size : Max records per page (validated by the router: ge=1, le=100).
    """
    # ── Total count ───────────────────────────────────────────────────────
    count_stmt = select(func.count()).select_from(Prediction)
    total: int = (await db.execute(count_stmt)).scalar_one()

    if total == 0:
        return Page(items=[], total=0, page=page, size=size, pages=0)

    # ── Paginated data ────────────────────────────────────────────────────
    offset = (page - 1) * size
    rows_stmt = (
        select(Prediction)
        .order_by(Prediction.timestamp.desc(), Prediction.id.desc())
        .offset(offset)
        .limit(size)
    )
    rows = (await db.execute(rows_stmt)).scalars().all()
    items = [PredictionResponse.model_validate(row) for row in rows]

    return make_page(items=items, total=total, page=page, size=size)
