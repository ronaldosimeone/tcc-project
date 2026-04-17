"""
Pydantic schemas for the /models admin router (RF-11).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# Exhaustive list of selectable models.  Adding a new model here is sufficient
# to make it a valid value in SwapModelRequest — no other code change needed.
ModelName = Literal["random_forest", "xgboost", "mlp"]


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
