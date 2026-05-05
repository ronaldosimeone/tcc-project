"""
Admin router — model management endpoints (RF-11).

All endpoints in this router require the `X-Admin-Token` header.

GET  /models         — list all registered models and their status.
PUT  /models/active  — atomically swap the active inference model.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from src.core.auth import require_admin_token
from src.core.config import settings
from src.schemas.models import (
    ModelSummary,
    ModelsListResponse,
    SwapModelRequest,
    SwapModelResponse,
)
from src.services.model_registry import KNOWN_MODELS, ModelRegistry, get_model_registry

router: APIRouter = APIRouter(
    prefix="/models",
    tags=["Model Management"],
    dependencies=[Depends(require_admin_token)],
)

# Maps each known model name to the Path that signals it is ready on disk.
# mlp requires *both* the ONNX file and the scaler; we check the ONNX file
# as the canonical signal (the scaler is always produced alongside it).
# V2 entries point at the .onnx files served by OnnxTreeAdapter.
_ARTEFACT_PATHS: dict[str, Path] = {
    "random_forest": settings.model_path,
    "xgboost": settings.xgboost_model_path,
    "mlp": settings.mlp_onnx_path,
    "random_forest_v2": settings.rf_v2_onnx_path,
    "xgboost_v2": settings.xgboost_v2_onnx_path,
}


@router.get(
    "",
    response_model=ModelsListResponse,
    summary="List all registered models",
    description=(
        "Returns the active model name and the availability status of every "
        "registered model artefact.  Requires `X-Admin-Token` header."
    ),
)
async def list_models(
    registry: ModelRegistry = Depends(get_model_registry),
) -> ModelsListResponse:
    active = registry.active_name
    models = [
        ModelSummary(
            name=name,
            active=(name == active),
            artefact_ready=_ARTEFACT_PATHS[name].exists(),
        )
        for name in sorted(KNOWN_MODELS)
    ]
    return ModelsListResponse(active_model=active, models=models)


@router.put(
    "/active",
    response_model=SwapModelResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Hot-swap the active model",
    description=(
        "Validates the artefact path and enqueues the model load as a "
        "background task (RNF-25).  Returns 202 immediately — the swap "
        "completes asynchronously so the HTTP connection is never held open "
        "during the potentially multi-second joblib.load().  "
        "Requires `X-Admin-Token` header."
    ),
    responses={
        404: {"description": "Model artefact not found on disk."},
        422: {
            "description": "Invalid model name — must be one of the registered models."
        },
    },
)
async def swap_active_model(
    payload: SwapModelRequest,
    background_tasks: BackgroundTasks,
    registry: ModelRegistry = Depends(get_model_registry),
) -> SwapModelResponse:
    # Validate artefact presence before accepting — avoids a silent failure
    # in the background task where the client would never learn about the error.
    artefact_path = _ARTEFACT_PATHS.get(payload.model_name)
    if artefact_path is not None and not artefact_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model artefact not found for '{payload.model_name}': {artefact_path}",
        )

    previous = registry.active_name
    background_tasks.add_task(registry.swap, payload.model_name)

    return SwapModelResponse(
        previous_model=previous,
        active_model=payload.model_name,
        message=(
            f"Model swap from '{previous}' to '{payload.model_name}' accepted. "
            "Loading in background — use GET /models to track the active model."
        ),
    )
