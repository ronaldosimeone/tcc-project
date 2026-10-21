"""
Pydantic schemas for the /models admin router (RF-11).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# Exhaustive list of selectable models.  Must stay in sync with
# services.model_registry.KNOWN_MODELS and services.model_service.load_model_by_name.
# The served generation (v3, leak-free nested protocol) — same names as
# model_service.SERVED_MODELS.
ModelName = Literal[
    "random_forest_v3",
    "xgboost_v3",
    "mlp_v3",
    "tcn_v3",
    "bilstm_v3",
    "patchtst_v3",
    "autoencoder_v3",
]


class ModelSummary(BaseModel):
    """Status of a single registered model."""

    name: str
    active: bool
    artefact_ready: bool


class ModelsListResponse(BaseModel):
    """Response for GET /models."""

    active_model: str
    # Limiar de decisão calibrado do modelo ativo (card): a UI classifica o
    # risco relativo a ele. None quando nenhum modelo está carregado.
    active_threshold: float | None = None
    models: list[ModelSummary]


class SwapModelRequest(BaseModel):
    """Request body for PUT /models/active."""

    model_name: ModelName


class SwapModelResponse(BaseModel):
    """Response for PUT /models/active."""

    previous_model: str
    active_model: str
    message: str
