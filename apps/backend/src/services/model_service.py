"""
ModelService — [RNF-11] inference singleton, [RF-10] multi-model factory.

The same ModelService class wraps any sklearn-compatible estimator that
exposes ``.predict()``, ``.predict_proba()`` and ``.feature_names_in_`` —
RandomForestClassifier, XGBClassifier and OnnxMlpAdapter all satisfy this
contract.

Model selection at startup (RF-10)
-----------------------------------
The active model is determined by the ``ACTIVE_MODEL`` environment variable
(via ``settings.active_model``).  No code change or redeploy is required:

    ACTIVE_MODEL=random_forest   →  loads random_forest_final.joblib  (default)
    ACTIVE_MODEL=xgboost         →  loads xgboost_v1.joblib
    ACTIVE_MODEL=mlp             →  loads mlp_v1.onnx + mlp_scaler.joblib

V2 — Decision threshold tuning
------------------------------
The default classification threshold of 0.5 is suboptimal for industrial
imbalanced fault detection (cost of a missed fault ≫ cost of a false alarm).
The training scripts now compute an F2-favouring threshold from the
Precision-Recall curve and persist it under ``decision_threshold`` in the
model card.  ``ModelService`` reads that value at load time and applies it
during inference; legacy cards without the field fall back to 0.5 so older
artefacts continue to work unchanged.
"""

from __future__ import annotations

import json
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

# V2 — pares de sensores correlacionados usados nos cross-features
_DEFAULT_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Helpers — model card / threshold resolution
# ---------------------------------------------------------------------------


_MODEL_CARDS: dict[str, str] = {
    "random_forest": "model_card.json",
    "xgboost": "xgboost_v1_card.json",
    "mlp": "mlp_v1_card.json",
    # V2 ONNX exports share the same training scripts (and therefore the same
    # cards) as their joblib siblings — train_random_forest.py writes
    # model_card.json and train_xgboost.py writes xgboost_v1_card.json.
    "random_forest_v2": "model_card.json",
    "xgboost_v2": "xgboost_v1_card.json",
    # Sequential DL models — RNF-24 extension
    "tcn": "tcn_v1_card.json",
    "bilstm": "bilstm_v1_card.json",
    "patchtst": "patchtst_v1_card.json",
    # Unsupervised Conv1D Autoencoder
    "autoencoder": "autoencoder_v1_card.json",
}


def _read_model_card(model_name: str) -> dict[str, Any] | None:
    """Return the parsed model card for *model_name*, or None if unavailable."""
    card_name = _MODEL_CARDS.get(model_name)
    if card_name is None:
        return None

    card_path = settings.model_path.parent / card_name
    if not card_path.exists():
        return None

    try:
        return json.loads(card_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to parse model card %s: %s", card_path, exc)
        return None


def _resolve_threshold(model_name: str) -> float:
    """
    Read ``decision_threshold`` from the matching model card, with safe fallback.

    Returns ``_DEFAULT_THRESHOLD`` (0.5) when the card is missing, malformed,
    or does not contain the field — guarantees that V1 artefacts keep working.
    """
    card = _read_model_card(model_name)
    if card is None:
        return _DEFAULT_THRESHOLD

    raw = card.get("decision_threshold", _DEFAULT_THRESHOLD)
    try:
        threshold = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_THRESHOLD

    # Sanity clamp — refuse pathological values silently rather than crashing
    if not 0.0 < threshold < 1.0:
        logger.warning(
            "decision_threshold=%s out of (0,1) range — using 0.5",
            threshold,
        )
        return _DEFAULT_THRESHOLD

    return threshold


def _resolve_feature_names(model_name: str) -> list[str]:
    """
    Read ``feature_names`` from the matching model card.

    Required by ``OnnxTreeAdapter`` because tree-based ONNX graphs do not
    embed column names — the adapter has to be told the training column
    order.  RF V2 has 80 features (cross-features + lags + min/max/range);
    XGB V2 has 34 (no lags).  Reading the card keeps the adapter generic.
    """
    card = _read_model_card(model_name)
    if card is None:
        raise FileNotFoundError(
            f"Model card for '{model_name}' is missing — cannot resolve "
            "feature_names for the ONNX adapter."
        )
    names = card.get("feature_names")
    if not isinstance(names, list) or not names:
        raise ValueError(
            f"Model card for '{model_name}' has no usable 'feature_names' field."
        )
    return [str(n) for n in names]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_model(
    path: Path, decision_threshold: float = _DEFAULT_THRESHOLD
) -> "ModelService":
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
    return ModelService(model, decision_threshold=decision_threshold)


# RF-10 registry — maps ACTIVE_MODEL values to their artefact paths.
_MODEL_REGISTRY: dict[str, Path] = {
    "random_forest": settings.model_path,
    "xgboost": settings.xgboost_model_path,
}


def load_model_by_name(model_name: str) -> "ModelService":
    """
    Load a ModelService by explicit name (RF-10, RNF-25).

    Reads the optimal decision threshold from the matching model card so that
    V2 artefacts apply their tuned cut-off automatically.
    """
    model_name = model_name.lower().strip()
    threshold = _resolve_threshold(model_name)

    if threshold != _DEFAULT_THRESHOLD:
        logger.info(
            "[V2] %s usará threshold=%.4f (PR-curve / F2-score)",
            model_name,
            threshold,
        )

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
        return ModelService(adapter, decision_threshold=threshold)

    if model_name in {"random_forest_v2", "xgboost_v2"}:
        from src.services.onnx_tree_adapter import OnnxTreeAdapter

        onnx_path = (
            settings.rf_v2_onnx_path
            if model_name == "random_forest_v2"
            else settings.xgboost_v2_onnx_path
        )
        feature_names = _resolve_feature_names(model_name)
        logger.info(
            "[RF-10] Loading model '%s' | onnx=%s | features=%d",
            model_name,
            onnx_path,
            len(feature_names),
        )
        return ModelService(
            OnnxTreeAdapter(onnx_path=onnx_path, feature_names=feature_names),
            decision_threshold=threshold,
        )

    if model_name in {"tcn", "bilstm", "patchtst"}:
        return _load_sequential_model(model_name, threshold)

    if model_name == "autoencoder":
        return _load_autoencoder_model(threshold)

    path = _MODEL_REGISTRY.get(model_name, settings.model_path)
    logger.info("[RF-10] Loading model '%s' | path = %s", model_name, path)
    return load_model(path, decision_threshold=threshold)


def _load_sequential_model(model_name: str, threshold: float) -> "ModelService":
    """
    Load a sequential ONNX model (TCN, BiLSTM or PatchTST) via OnnxSequenceAdapter.

    Reads ``window_size`` and ``feature_names`` from the model card so the
    adapter is self-configured without hard-coded constants here.
    """
    from src.services.onnx_sequence_adapter import OnnxSequenceAdapter

    onnx_path_map: dict[str, Path] = {
        "tcn": settings.tcn_onnx_path,
        "bilstm": settings.bilstm_onnx_path,
        "patchtst": settings.patchtst_onnx_path,
    }
    scaler_path_map: dict[str, Path] = {
        "tcn": settings.tcn_scaler_path,
        "bilstm": settings.bilstm_scaler_path,
        "patchtst": settings.patchtst_scaler_path,
    }

    onnx_path: Path = onnx_path_map[model_name]
    scaler_path: Path = scaler_path_map[model_name]

    # Read window_size and channel_names from the card — avoids hard-coding.
    card: dict[str, Any] | None = _read_model_card(model_name)
    window_size: int = 60  # safe default matching train_sequential.py
    channel_names: list[str] | None = None
    if card is not None:
        # ``card.get("inference") or {}`` handles both the missing-key case
        # (returns {}) and the explicit-null case (returns {}); the bare
        # ``card.get("inference", {})`` would let an explicit JSON null
        # propagate and crash on the next ``.get(...)``.
        inference_cfg: dict[str, Any] = card.get("inference") or {}
        window_size = int(inference_cfg.get("window_size", window_size))
        raw_names = card.get("feature_names")
        if isinstance(raw_names, list) and raw_names:
            channel_names = [str(n) for n in raw_names]

    logger.info(
        "[RF-10] Loading model '%s' | onnx=%s | T=%d | C=%s",
        model_name,
        onnx_path,
        window_size,
        len(channel_names) if channel_names else "default",
    )

    adapter: OnnxSequenceAdapter = OnnxSequenceAdapter(
        onnx_path=onnx_path,
        scaler_path=scaler_path,
        window_size=window_size,
        channel_names=channel_names,
    )
    return ModelService(adapter, decision_threshold=threshold)


def _load_autoencoder_model(threshold: float) -> "ModelService":
    """
    Load the Conv1D Autoencoder via OnnxAutoencoderAdapter.

    Reads ``mse_threshold`` and ``feature_names`` from the model card so the
    adapter is fully self-configured without hard-coded constants here.
    """
    from src.services.onnx_autoencoder_adapter import OnnxAutoencoderAdapter

    card: dict[str, Any] | None = _read_model_card("autoencoder")
    if card is None:
        raise FileNotFoundError(
            "autoencoder_v1_card.json not found — run `python src/train_autoencoder.py` first."
        )

    mse_threshold = card.get("mse_threshold")
    if mse_threshold is None:
        raise ValueError(
            "autoencoder_v1_card.json is missing 'mse_threshold'. "
            "Re-run `python src/train_autoencoder.py` to regenerate the card."
        )

    window_size: int = 60
    channel_names: list[str] | None = None
    inference_cfg: dict[str, Any] = card.get("inference") or {}
    window_size = int(inference_cfg.get("window_size", window_size))
    raw_names = card.get("feature_names")
    if isinstance(raw_names, list) and raw_names:
        channel_names = [str(n) for n in raw_names]

    logger.info(
        "[RF-10] Loading model 'autoencoder' | onnx=%s | T=%d | mse_threshold=%.6f",
        settings.autoencoder_onnx_path,
        window_size,
        float(mse_threshold),
    )

    adapter = OnnxAutoencoderAdapter(
        onnx_path=settings.autoencoder_onnx_path,
        scaler_path=settings.autoencoder_scaler_path,
        mse_threshold=float(mse_threshold),
        window_size=window_size,
        channel_names=channel_names,
    )
    return ModelService(adapter, decision_threshold=threshold)


def load_active_model() -> "ModelService":
    """
    Load whichever model is selected by ``settings.active_model`` (RF-10).

    Thin wrapper around ``load_model_by_name`` for startup use.  The runtime
    hot-swap path calls ``load_model_by_name`` directly via ``ModelRegistry``.
    """
    return load_model_by_name(settings.active_model)


# ---------------------------------------------------------------------------
# Classe Service
# ---------------------------------------------------------------------------


class ModelService:
    """
    Wrapper that produces a :class:`PredictResponse` from a single sensor
    snapshot.

    Parameters
    ----------
    model:
        Any sklearn-compatible estimator exposing ``predict_proba`` and
        ``feature_names_in_``.
    decision_threshold:
        Probability cut-off for the positive (fault) class.  ``0.5`` is the
        sklearn default; values below 0.5 favour recall (industrial setting),
        values above 0.5 favour precision.  Loaded from the model card by
        :func:`load_model_by_name` so the same code path serves V1 and V2
        artefacts.
    """

    def __init__(
        self,
        model: Any,
        decision_threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self._model = model
        self._threshold: float = decision_threshold
        # Mapeia as colunas que o modelo realmente espera para evitar KeyError
        self._expected_features = list(self._model.feature_names_in_)

    @property
    def decision_threshold(self) -> float:
        """Active classification cut-off."""
        return self._threshold

    def predict(self, request: PredictRequest) -> PredictResponse:
        """
        Stateless inference path — used by ``POST /predict`` and as the
        cold-start fallback in ``InferencePipelineService`` before the sensor
        buffer is warm.  The rolling/lag features come out neutralised
        (std=0, ma=current value, lag=0) which is why production code prefers
        :meth:`predict_from_features` once enough history is buffered.
        """
        return self.predict_from_features(self._build_feature_row(request))

    def predict_from_features(self, X: pd.DataFrame) -> PredictResponse:
        """
        Run inference on a feature row that was already engineered upstream.

        Parameters
        ----------
        X : pd.DataFrame
            Single-row DataFrame containing at least the columns the model
            was trained on.  Extra columns are dropped, missing columns are
            backfilled with neutral defaults — same alignment policy as the
            stateless :meth:`predict` path so V1 and V2 cards stay swappable.

        Returns
        -------
        PredictResponse
            Same shape as :meth:`predict`.  This is the path used by
            ``InferencePipelineService`` once ``SensorBuffer.is_warm()`` —
            it skips :meth:`_build_feature_row` (which zeroes rolling stats)
            and feeds the model the real time-series features.
        """
        try:
            X = X.copy()  # never mutate caller's DataFrame

            # Alinhamento Dinâmico: garante que TODAS as colunas esperadas existam.
            for col in self._expected_features:
                if col not in X.columns:
                    # LPS / Caudal / Pressure_switch existem no schema antigo
                    # com valor lógico "1 = OK"; o resto cai pra zero neutro.
                    X[col] = (
                        1.0
                        if col in {"LPS", "Pressure_switch", "Caudal_impulses"}
                        else 0.0
                    )

            # Reordena exatamente como o modelo foi treinado (drop extras).
            X = X[self._expected_features]

            failure_probability: float = float(self._model.predict_proba(X)[0][1])
            predicted_class: int = int(failure_probability >= self._threshold)

            return PredictResponse(
                predicted_class=predicted_class,
                failure_probability=round(failure_probability, 6),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        except Exception:
            # Bare `raise` preserves the original traceback (PEP 8 / B904).
            logger.exception("Erro na predição")
            raise

    def _build_feature_row(self, req: PredictRequest) -> pd.DataFrame:
        """
        Constrói o vetor de features para um único snapshot.

        Inclui:
          • 12 sensores brutos
          • TP2_delta (0 — não há histórico em produção)
          • Rolling stats stateless (std=0, ma=valor atual)
          • V2 — Cross-sensor features (TP2_TP3_diff/ratio, work_per_pressure,
            reservoir_drop) — totalmente stateless, paridade exata com o
            preprocessor de treino.

        Modelos V1 (treinados sem cross-features) ignoram silenciosamente as
        novas colunas via ``X[self._expected_features]``.  Modelos V2 as
        consomem normalmente.
        """
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

        # ── V2 cross-sensor features (stateless, paridade total com preprocessing.py) ──
        eps = 1e-6
        features["TP2_TP3_diff"] = raw["TP2"] - raw["TP3"]
        features["TP2_TP3_ratio"] = raw["TP2"] / (raw["TP3"] + eps)
        features["work_per_pressure"] = raw["Motor_current"] / (raw["TP2"] + eps)
        features["reservoir_drop"] = raw["Reservoirs"] - raw["TP3"]

        return pd.DataFrame([features]).astype(np.float32)


# ---------------------------------------------------------------------------
# Dependência FastAPI
# ---------------------------------------------------------------------------


async def get_model_service(request: Request) -> ModelService:
    """
    FastAPI dependency — resolves the active ModelService from the registry.

    Reads from ``app.state.model_registry`` (set by the lifespan via ModelRegistry).
    Falls back to the legacy ``app.state.model_service`` so that existing tests
    that override this dependency via ``dependency_overrides`` continue to work
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
