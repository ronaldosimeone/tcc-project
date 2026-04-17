"""
ModelService — [RNF-11] inference singleton, [RF-10] multi-model factory.

The same ModelService class wraps any sklearn-compatible estimator that
exposes `.predict()`, `.predict_proba()` and `.feature_names_in_` —
RandomForestClassifier, XGBClassifier and OnnxMlpAdapter all satisfy this
contract.

Model selection at startup (RF-10)
-----------------------------------
The active model is determined by the `ACTIVE_MODEL` environment variable
(via `settings.active_model`).  No code change or redeploy is required:

    ACTIVE_MODEL=random_forest   →  loads random_forest_final.joblib  (default)
    ACTIVE_MODEL=xgboost         →  loads xgboost_v1.joblib
    ACTIVE_MODEL=mlp             →  loads mlp_v1.onnx + mlp_scaler.joblib

Use `load_active_model()` in the FastAPI lifespan instead of `load_model()`.
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

from src.core.config import settings
from src.core.exceptions import ModelNotAvailableError
from src.schemas.predict import PredictRequest, PredictResponse

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuração de Colunas (Paridade com o Treinamento)
# ---------------------------------------------------------------------------

_SENSOR_COLS: list[str] = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Motor_current",
    "Oil_temperature",
]

_BINARY_COLS: list[str] = ["COMP", "DV_eletric", "Towers", "MPG", "Oil_level"]

# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_model(path: Path) -> "ModelService":
    """Carrega o modelo do disco. Se não achar, tenta o nome alternativo."""
    if not path.exists():
        # Tentativa amigável caso o config ainda aponte para o nome antigo
        alt_path = path.parent / "random_forest_final.joblib"
        if alt_path.exists():
            logger.info("[RNF-11] Usando caminho alternativo: %s", alt_path)
            path = alt_path
        else:
            logger.error("[RNF-11] Modelo não encontrado em %s", path)
            raise FileNotFoundError(f"Model artefact missing at {path}")

    model: Any = joblib.load(path)
    logger.info("[RNF-11] Modelo carregado com sucesso: %s", path)
    return ModelService(model)


# RF-10 registry — maps ACTIVE_MODEL values to their artefact paths.
_MODEL_REGISTRY: dict[str, Path] = {
    "random_forest": settings.model_path,
    "xgboost": settings.xgboost_model_path,
}


def load_model_by_name(model_name: str) -> "ModelService":
    """
    Load a ModelService by explicit name (RF-10, RNF-25).

    Called by `load_active_model()` at startup and by `ModelRegistry.swap()`
    at runtime.  Keeping the loading logic here (rather than in the registry)
    preserves the single-responsibility boundary: model_service owns *how* to
    load; model_registry owns *when* and *atomically*.

    Parameters
    ----------
    model_name : str
        One of ``"random_forest"``, ``"xgboost"``, or ``"mlp"``.
        Falls back to the default random-forest path for unrecognised keys.
    """
    model_name = model_name.lower().strip()

    if model_name == "mlp":
        from src.services.mlp_adapter import OnnxMlpAdapter

        logger.info(
            "[RF-10] Loading model 'mlp' | onnx=%s | scaler=%s",
            settings.mlp_onnx_path,
            settings.mlp_scaler_path,
        )
        adapter = OnnxMlpAdapter(
            onnx_path=settings.mlp_onnx_path,
            scaler_path=settings.mlp_scaler_path,
        )
        return ModelService(adapter)

    path = _MODEL_REGISTRY.get(model_name, settings.model_path)
    logger.info("[RF-10] Loading model '%s' | path = %s", model_name, path)
    return load_model(path)


def load_active_model() -> "ModelService":
    """
    Load whichever model is selected by `settings.active_model` (RF-10).

    Thin wrapper around `load_model_by_name` for startup use.  The runtime
    hot-swap path calls `load_model_by_name` directly via `ModelRegistry`.
    """
    return load_model_by_name(settings.active_model)


# ---------------------------------------------------------------------------
# Classe Service
# ---------------------------------------------------------------------------


class ModelService:
    def __init__(self, model: Any) -> None:
        self._model = model
        # Mapeia as colunas que o modelo realmente espera para evitar KeyError
        self._expected_features = list(self._model.feature_names_in_)

    def predict(self, request: PredictRequest) -> PredictResponse:
        """Executa a inferência com tratamento de erro 500."""
        try:
            # 1. Constrói o DataFrame base (12 sensores + rolling features)
            X: pd.DataFrame = self._build_feature_row(request)

            # 2. Alinhamento Dinâmico: Garante que TODAS as colunas esperadas existam
            for col in self._expected_features:
                if col not in X.columns:
                    # Se o modelo pedir LPS ou Caudal, injetamos valor neutro (1.0)
                    X[col] = 1.0

            # 3. Reordena as colunas exatamente como o modelo foi treinado
            X = X[self._expected_features]

            # 4. Predição
            predicted_class: int = int(self._model.predict(X)[0])
            failure_probability: float = float(self._model.predict_proba(X)[0][1])

            return PredictResponse(
                predicted_class=predicted_class,
                failure_probability=round(failure_probability, 6),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        except Exception as e:
            logger.error("❌ Erro na predição: %s", str(e))
            raise e

    def _build_feature_row(self, req: PredictRequest) -> pd.DataFrame:
        """Constrói o vetor de 34 colunas exigido pelo preprocessor."""
        # 12 sensores brutos
        raw = {
            "TP2": req.TP2,
            "TP3": req.TP3,
            "H1": req.H1,
            "DV_pressure": req.DV_pressure,
            "Reservoirs": req.Reservoirs,
            "Motor_current": req.Motor_current,
            "Oil_temperature": req.Oil_temperature,
            "COMP": req.COMP,
            "DV_eletric": req.DV_eletric,
            "Towers": req.Towers,
            "MPG": req.MPG,
            "Oil_level": req.Oil_level,
        }

        features = dict(raw)
        features["TP2_delta"] = 0.0

        # Rolling features "stateless" (médias = valor atual, desvio = 0)
        for col in _SENSOR_COLS:
            features[f"{col}_std_5"] = 0.0
            features[f"{col}_ma_5"] = raw[col]
            features[f"{col}_ma_15"] = raw[col]

        return pd.DataFrame([features]).astype(np.float32)


# ---------------------------------------------------------------------------
# Dependência FastAPI
# ---------------------------------------------------------------------------


async def get_model_service(request: Request) -> ModelService:
    """
    FastAPI dependency — resolves the active ModelService from the registry.

    Reads from `app.state.model_registry` (set by the lifespan via ModelRegistry).
    Falls back to the legacy `app.state.model_service` so that existing tests
    that override this dependency via `dependency_overrides` continue to work
    without modification.
    """
    registry = getattr(request.app.state, "model_registry", None)
    if registry is not None:
        return await registry.get()

    # Legacy fallback (used by tests that override this dependency directly)
    service: ModelService | None = getattr(request.app.state, "model_service", None)
    if service is not None:
        return service

    raise ModelNotAvailableError()
