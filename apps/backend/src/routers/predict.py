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

RNF-56: dependencies typed against `ModelServiceProtocol`/`AlertServiceProtocol`
(`src/services/protocols.py`) instead of the concrete classes — the
factories (`get_model_service`/`get_alert_service`) are unchanged, already
override-friendly (see `tests/test_predict_endpoint.py`).
"""

import asyncio

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.core.rate_limit import PREDICT_BATCH_RATE_LIMIT, PREDICT_RATE_LIMIT, limiter
from src.schemas.predict import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)
from src.services.alert_service import get_alert_service
from src.services.inference_cache import (
    build_cache_key,
    get_active_model_name,
    get_inference_cache,
)
from src.services.model_service import get_model_service
from src.services.prediction_service import save_prediction
from src.services.protocols import (
    AlertServiceProtocol,
    InferenceCacheProtocol,
    ModelServiceProtocol,
)

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
    service: ModelServiceProtocol = Depends(get_model_service),
    db: AsyncSession = Depends(get_db),
    alert_service: AlertServiceProtocol = Depends(get_alert_service),
    cache: InferenceCacheProtocol = Depends(get_inference_cache),
    active_model_name: str = Depends(get_active_model_name),
) -> PredictResponse:
    """
    Run fault detection, persist the result (RF-09) and push WS alert (RF-14).

    RNF-70/71 — cache-aside: mesmos 12 sensores + mesmo modelo ativo, dentro
    de 60s, devolve a MESMA PredictResponse já computada, sem reinferir, sem
    reinserir em `predictions` e sem reprocessar alerta (ver
    services/inference_cache.py para a justificativa completa da chave e do
    porquê HIT pula o restante do pipeline).
    """
    cache_key = build_cache_key(payload, model_name=active_model_name)
    cached: PredictResponse | None = await cache.get(cache_key)
    if cached is not None:
        return cached

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
    await cache.set(cache_key, result)
    return result


@router.post(
    "/batch",
    response_model=BatchPredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch fault prediction from multiple sensor snapshots",
    description=(
        "**RNF-73** — aceita de 1 a 100 snapshots numa única requisição e "
        "executa inferência vetorizada (uma única chamada ao modelo para "
        "todas as amostras, não um loop de N predições). `predictions[i]` "
        "corresponde a `samples[i]` — mesma ordem, mesma contagem."
        "\n\nFora do escopo desta rota (decisão deliberada, ver "
        "RELATORIO-RNF-72-RNF-73.md): não persiste em `predictions` nem "
        "aciona alertas/cache — RF-09/RF-14/RNF-70 continuam escopados ao "
        "`POST /predict/` de amostra única."
    ),
    responses={
        422: {"description": "Batch vazio, >100 amostras, ou amostra inválida."},
        429: {"description": "Rate limit exceeded — slow down and retry."},
        503: {"description": "Model not loaded — check startup logs."},
    },
)
@limiter.limit(PREDICT_BATCH_RATE_LIMIT)
async def predict_batch(
    request: Request,
    payload: BatchPredictRequest,
    service: ModelServiceProtocol = Depends(get_model_service),
) -> BatchPredictResponse:
    """RNF-73 — inferência vetorizada real (ver ModelService.predict_batch)."""
    results: list[PredictResponse] = await asyncio.to_thread(
        service.predict_batch, payload.samples
    )
    return BatchPredictResponse(predictions=results, count=len(results))
