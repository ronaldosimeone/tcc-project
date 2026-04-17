"""
Admin router — model management endpoints (RF-11).

All endpoints in this router require the `X-Admin-Token` header.

GET  /models         — list all registered models and their status.
PUT  /models/active  — atomically swap the active inference model.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

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
_ARTEFACT_PATHS: dict[str, Path] = {
    "random_forest": settings.model_path,
    "xgboost": settings.xgboost_model_path,
    "mlp": settings.mlp_onnx_path,
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
    status_code=status.HTTP_200_OK,
    summary="Hot-swap the active model",
    description=(
        "Loads the specified model artefact in a background thread and "
        "atomically replaces the active `ModelService` (RNF-25).  "
        "In-flight prediction requests are never interrupted.  "
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
    registry: ModelRegistry = Depends(get_model_registry),
) -> SwapModelResponse:
    try:
        previous = await registry.swap(payload.model_name)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model artefact not found: {exc}",
        ) from exc

    return SwapModelResponse(
        previous_model=previous,
        active_model=payload.model_name,
        message=(
            f"Active model swapped from '{previous}' to '{payload.model_name}' "
            "successfully."
        ),
    )
