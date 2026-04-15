"""
ModelService — [RNF-11] Random Forest inference singleton.

Responsibilities
----------------
- Load the .joblib artefact exactly once (via `load_model`, called from the
  FastAPI lifespan in main.py).
- Build the 36-column feature vector that matches the training distribution
  produced by MetroPTPreprocessor for a stateless single-row request.
- Expose `predict(request)` as a pure synchronous method for clean testability.
- Provide `get_model_service(request)` as the FastAPI Depends factory.

Feature engineering notes
--------------------------
The Random Forest was trained on the output of MetroPTPreprocessor, which adds
rolling-window features (std-5, ma-5, ma-15) and a pressure delta.  For a
single, stateless reading:
  - pressure_delta  = 0.0  (no prior sample available)
  - rolling std     = 0.0  (variance of a single point is zero)
  - rolling MA      = raw sensor value  (window collapses to the sample itself)

This matches exactly what MetroPTPreprocessor produces with min_periods=1 on a
one-row DataFrame — the feature vector is identical to the training pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import Request

from src.core.exceptions import ModelNotAvailableError
from src.schemas.predict import PredictRequest, PredictResponse

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature column order — must match the training script exactly.
# ---------------------------------------------------------------------------

# 7 analogue sensors that receive rolling features in MetroPTPreprocessor
_SENSOR_COLS: list[str] = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
]

# 7 digital / switch columns (no rolling features applied)
_BINARY_COLS: list[str] = [
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
]

# Ordered list of all 36 features (14 raw + 1 delta + 7 std + 7 ma5 + 7 ma15)
_FEATURE_COLS: list[str] = (
    _SENSOR_COLS
    + _BINARY_COLS
    + ["TP2_delta"]
    + [f"{c}_std_5" for c in _SENSOR_COLS]
    + [f"{c}_ma_5" for c in _SENSOR_COLS]
    + [f"{c}_ma_15" for c in _SENSOR_COLS]
)


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_model(path: Path) -> "ModelService":
    """
    Deserialise the Random Forest from disk and wrap it in a `ModelService`.

    Called once during application startup (FastAPI lifespan).
    Raises FileNotFoundError when the artefact is absent.
    """
    model: Any = joblib.load(path)
    logger.info("[RNF-11] Model loaded from %s", path)
    return ModelService(model)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ModelService:
    """
    Stateless inference wrapper around the trained RandomForestClassifier.

    Attributes
    ----------
    _model : sklearn estimator
        The deserialised Random Forest instance.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def predict(self, request: PredictRequest) -> PredictResponse:
        """
        Build the feature vector, run inference, return a PredictResponse.

        Parameters
        ----------
        request : PredictRequest
            Raw sensor readings from a single acquisition cycle.

        Returns
        -------
        PredictResponse
            predicted_class, failure_probability, and UTC ISO timestamp.
        """
        X: pd.DataFrame = self._build_feature_row(request)

        # Garante a ordem exata de colunas que o modelo viu no treinamento
        X = X[self._model.feature_names_in_]

        predicted_class: int = int(self._model.predict(X)[0])
        failure_probability: float = float(self._model.predict_proba(X)[0][1])
        timestamp: str = datetime.now(timezone.utc).isoformat()

        return PredictResponse(
            predicted_class=predicted_class,
            failure_probability=round(failure_probability, 6),
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_feature_row(self, req: PredictRequest) -> pd.DataFrame:
        """
        Construct the 36-column feature DataFrame for a single reading.

        Matches MetroPTPreprocessor output for a 1-row stateless input:
        delta = 0, rolling std = 0, rolling MA = raw value.
        """
        raw: dict[str, float] = {
            "TP2": req.TP2,
            "TP3": req.TP3,
            "H1": req.H1,
            "DV_pressure": req.DV_pressure,
            "Reservoirs": req.Reservoirs,
            "Oil_temperature": req.Oil_temperature,
            "Motor_current": req.Motor_current,
            "COMP": req.COMP,
            "DV_eletric": req.DV_eletric,
            "Towers": req.Towers,
            "MPG": req.MPG,
            "Pressure_switch": req.Pressure_switch,
            "Oil_level": req.Oil_level,
            "Caudal_impulses": req.Caudal_impulses,
        }

        row: dict[str, float] = dict(raw)
        row["TP2_delta"] = 0.0

        for col in _SENSOR_COLS:
            row[f"{col}_std_5"] = 0.0
            row[f"{col}_ma_5"] = raw[col]
            row[f"{col}_ma_15"] = raw[col]

        # Preserve training-time column order to avoid sklearn feature-name warnings
        return pd.DataFrame([row], columns=_FEATURE_COLS).astype(np.float32)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


def get_model_service(request: Request) -> ModelService:
    """
    FastAPI Depends factory — retrieves the model singleton from app.state.

    Raises
    ------
    ModelNotAvailableError
        When the model failed to load during startup (HTTP 503).
    """
    service: ModelService | None = getattr(request.app.state, "model_service", None)
    if service is None:
        raise ModelNotAvailableError()
    return service
