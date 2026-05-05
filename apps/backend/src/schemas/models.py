"""
Pydantic schemas for the /models admin router (RF-11).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# Exhaustive list of selectable models.  Must stay in sync with
# services.model_registry.KNOWN_MODELS and services.model_service.load_model_by_name.
ModelName = Literal[
    "random_forest",
    "xgboost",
    "mlp",
    "random_forest_v2",
    "xgboost_v2",
]


class ModelSummary(BaseModel):
    """Status of a single registered model."""

    name: str
    active: bool
    artefact_ready: bool


class ModelsListResponse(BaseModel):
    """Response for GET /models."""

    active_model: str
    models: list[ModelSummary]


class SwapModelRequest(BaseModel):
    """Request body for PUT /models/active."""

    model_name: ModelName


class SwapModelResponse(BaseModel):
    """Response for PUT /models/active."""

    previous_model: str
    active_model: str
    message: str
