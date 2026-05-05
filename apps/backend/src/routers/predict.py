"""
Prediction router.

Responsibilities (Clean Arch §3):
- I/O only: receive request, inject dependencies, return response.
- Zero business logic — delegates inference to ModelService and
  persistence to PredictionService.

Flow: POST /predict/
  1. slowapi checks the per-IP rate budget (RNF-19) before the handler runs.
  2. ModelService.predict(payload)        → PredictResponse (ML inference)
  3. save_prediction(db, payload, result) → Prediction ORM record (RF-09)
  4. Return PredictResponse to client

Concurrency model
-----------------
Inference is CPU-bound (RandomForest / XGBoost / ONNX MLP) and would block
the asyncio event loop if executed inline.  We dispatch it to the default
threadpool via ``asyncio.to_thread`` so other coroutines (SSE broadcast,
WebSocket alerts, health checks) keep running while a single prediction is
in flight.  Combined with multiple Uvicorn workers in production this gives
near-linear horizontal scaling under concurrent load.
"""

import asyncio

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.core.rate_limit import PREDICT_RATE_LIMIT, limiter
from src.schemas.predict import PredictRequest, PredictResponse
from src.services.alert_service import AlertService, get_alert_service
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
        "\n\n**RNF-19** — limited to 100 requests per minute per IP address."
    ),
    responses={
        429: {"description": "Rate limit exceeded — slow down and retry."},
        503: {"description": "Model not loaded — check startup logs."},
    },
)
@limiter.limit(PREDICT_RATE_LIMIT)
async def predict(
    request: Request,
    payload: PredictRequest,
    service: ModelService = Depends(get_model_service),
    db: AsyncSession = Depends(get_db),
    alert_service: AlertService = Depends(get_alert_service),
) -> PredictResponse:
    """Run fault detection, persist the result (RF-09) and push WS alert (RF-14)."""
    # CPU-bound inference is dispatched to the default threadpool so the
    # event loop remains responsive for other I/O-bound coroutines.
    result: PredictResponse = await asyncio.to_thread(service.predict, payload)
    await save_prediction(db, payload, result)
    await alert_service.process_prediction(
        {
            "probability": result.failure_probability,
            "predicted_class": result.predicted_class,
            "timestamp": result.timestamp,
        }
    )
    return result
