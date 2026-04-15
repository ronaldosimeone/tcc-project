"""
Prediction router.

Responsibilities (Clean Arch §3):
- I/O only: receive request, inject ModelService, return response.
- Zero business logic — delegate entirely to ModelService.predict().

Note on concurrency
-------------------
Model inference is CPU-bound.  For high-throughput production deployments,
wrap the service call with `asyncio.to_thread(service.predict, payload)` and
consider deploying multiple Uvicorn workers.  For the current workload this
direct call is acceptable.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from src.schemas.predict import PredictRequest, PredictResponse
from src.services.model_service import ModelService, get_model_service

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
    ),
    responses={
        503: {"description": "Model not loaded — check startup logs."},
    },
)
async def predict(
    payload: PredictRequest,
    service: ModelService = Depends(get_model_service),
) -> PredictResponse:
    """Run fault detection on a raw sensor snapshot."""
    return service.predict(payload)
