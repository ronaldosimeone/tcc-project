"""
Prediction router.

Responsibilities (Clean Arch §3):
- I/O only: receive request, inject dependencies, return response.
- Zero business logic — delegates inference to ModelService and
  persistence to PredictionService.

Flow: POST /predict/
  1. ModelService.predict(payload)        → PredictResponse (ML inference)
  2. save_prediction(db, payload, result) → Prediction ORM record (RF-09)
  3. Return PredictResponse to client

Note on concurrency
-------------------
Model inference is CPU-bound.  For high-throughput production deployments,
wrap the service call with `asyncio.to_thread(service.predict, payload)` and
consider deploying multiple Uvicorn workers.  For the current workload this
direct call is acceptable.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.schemas.predict import PredictRequest, PredictResponse
from src.services.model_service import ModelService, get_model_service
from src.services.prediction_service import save_prediction

router: APIRouter = APIRouter(prefix="/predict", tags=["Predictions"])


@router.post(
    "/",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Fault prediction from sensor snapshot",
    description=(
        "Accepts a single MetroPT-3 compressor sensor reading and returns "
        "a binary fault prediction together with the failure probability. "
        "\n\n**RF-05** — response always contains `predicted_class` (int), "
        "`failure_probability` (float) and `timestamp` (ISO 8601)."
        "\n\n**RF-09** — every successful prediction is persisted to the database."
    ),
    responses={
        503: {"description": "Model not loaded — check startup logs."},
    },
)
async def predict(
    payload: PredictRequest,
    service: ModelService = Depends(get_model_service),
    db: AsyncSession = Depends(get_db),
) -> PredictResponse:
    """Run fault detection and persist the result (RF-09)."""
    result: PredictResponse = service.predict(payload)
    await save_prediction(db, payload, result)
    return result
