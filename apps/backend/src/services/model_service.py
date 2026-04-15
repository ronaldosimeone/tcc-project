"""
ModelService — [RNF-11] Random Forest inference singleton.
Refatorado para suportar o modelo random_forest_final.joblib com 12 sensores base.
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


def get_model_service(request: Request) -> ModelService:
    service: ModelService | None = getattr(request.app.state, "model_service", None)
    if service is None:
        raise ModelNotAvailableError()
    return service
