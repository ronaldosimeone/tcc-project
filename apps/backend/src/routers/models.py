"""
Admin router — model management endpoints (RF-11).

All endpoints in this router require the `X-Admin-Token` header.

GET  /models         — list all registered models and their status.
PUT  /models/active  — atomically swap the active inference model.

RNF-56: a checagem de "artefato pronto no disco" (antes um dict +
`.exists()` direto no router) foi movida para `ModelRegistry.list_summaries()`/
`ensure_artefact_ready()` (`src/services/model_registry.py`) — lógica de
domínio fora da rota. `ensure_artefact_ready` levanta `NotFoundError`
(`src/core/exceptions.py`), traduzida para HTTP 404 pelo handler global já
registrado em `main.py` — nenhuma tradução manual de exceção aqui.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, status

from src.core.auth import require_admin_token
from src.schemas.models import ModelsListResponse, SwapModelRequest, SwapModelResponse
from src.services.model_registry import get_model_registry
from src.services.protocols import ModelRegistryProtocol

router: APIRouter = APIRouter(
    prefix="/models",
    tags=["Model Management"],
    dependencies=[Depends(require_admin_token)],
)


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
    registry: ModelRegistryProtocol = Depends(get_model_registry),
) -> ModelsListResponse:
    return ModelsListResponse(
        active_model=registry.active_name, models=registry.list_summaries()
    )


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
    registry: ModelRegistryProtocol = Depends(get_model_registry),
) -> SwapModelResponse:
    # Validate artefact presence before accepting — avoids a silent failure
    # in the background task where the client would never learn about the error.
    registry.ensure_artefact_ready(payload.model_name)

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
