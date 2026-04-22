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

Note on concurrency
-------------------
Model inference is CPU-bound.  For high-throughput production deployments,
wrap the service call with `asyncio.to_thread(service.predict, payload)` and
consider deploying multiple Uvicorn workers.  For the current workload this
direct call is acceptable.

Note on `request: Request` + `Body(...)`
-----------------------------------------
slowapi's @limiter.limit() requires `request: Request` as the first argument.
FastAPI follows __wrapped__ via inspect.signature() to recover the original
parameter list, but when `request: Request` precedes a Pydantic model,
FastAPI's source-inference heuristic can misclassify the model as a Query
parameter instead of a Body — producing HTTP 422 on valid JSON POSTs.

Using `Body(...)` as the default value is the canonical FastAPI fix: it makes
the body source explicit and unambiguous regardless of what any decorator
does to the function wrapper.  PredictRequest and PredictResponse must be
imported directly (never under `if TYPE_CHECKING`) so Pydantic v2's
TypeAdapter can resolve them at class-definition time.
"""

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
    result: PredictResponse = service.predict(payload)
    await save_prediction(db, payload, result)
    await alert_service.process_prediction(
        {
            "probability": result.failure_probability,
            "predicted_class": result.predicted_class,
            "timestamp": result.timestamp,
        }
    )
    return result
