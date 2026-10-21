"""
ModelService — [RNF-11] inference singleton, [RF-10] multi-model factory.

The same ModelService class wraps the ONNX adapters (tree, MLP, sequence,
autoencoder): each exposes ``.predict()`` / ``.predict_proba()`` and the
feature or channel order it expects.

Served models (RF-10)
---------------------
Only ONE generation is served: ``*_v3`` (``SERVED_MODELS``), produced by
``apps/ml/src/train_production.py`` with the leak-free nested protocol —
hyper-parameters by leave-one-failure-out over the training failures, Platt
calibration on out-of-fold scores, RN-02 threshold on the calibrated scale.
Each card's ``serving`` block says how to load it, and its
``decision_threshold`` is the calibrated decision the alerts follow.
``ACTIVE_MODEL`` picks one at startup; ``PUT /models/active`` swaps at run time.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Request

from src.core.config import settings
from src.core.exceptions import ModelNotAvailableError
from src.core.metrics import record_batch, record_inference
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


# The ONE generation served by the platform — leak-free nested protocol
# (apps/ml/src/train_production.py). Earlier generations (v1 random split +
# SMOTE, v2 trees) leaked and were removed from serving.
SERVED_MODELS: tuple[str, ...] = (
    "random_forest_v3",
    "xgboost_v3",
    "mlp_v3",
    "tcn_v3",
    "bilstm_v3",
    "patchtst_v3",
    "autoencoder_v3",
)

_MODEL_CARDS: dict[str, str] = {name: f"{name}_card.json" for name in SERVED_MODELS}


def _read_model_card(model_name: str) -> dict[str, Any] | None:
    """Return the parsed model card for *model_name*, or None if unavailable."""
    card_name = _MODEL_CARDS.get(model_name)
    if card_name is None:
        return None

    card_path = settings.models_dir / card_name
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


def resolve_threshold(model_name: str) -> float:
    """Public read of a model's calibrated decision threshold (model card)."""
    return _resolve_threshold(model_name.lower().strip())


def load_model_by_name(model_name: str) -> "ModelService":
    """
    Load a served model by name (RF-10, RNF-25).

    Every served artefact is described by its card's ``serving`` block
    (adapter, ONNX file, scaler, window); the card also carries the
    calibrated ``decision_threshold`` chosen without leakage.
    """
    model_name = model_name.lower().strip()
    if model_name not in SERVED_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Served: {', '.join(SERVED_MODELS)}."
        )
    card = _read_model_card(model_name)
    if card is None or not isinstance(card.get("serving"), dict):
        raise FileNotFoundError(
            f"Model card for '{model_name}' is missing or has no 'serving' block "
            f"in {settings.models_dir} — run `python -m src.train_production`."
        )
    return _load_from_serving_card(model_name, card, _resolve_threshold(model_name))


def _load_from_serving_card(
    model_name: str, card: dict[str, Any], threshold: float
) -> "ModelService":
    """Card-driven loading for the leakage-free retrains.

    ``card["serving"]`` = {"adapter": tree|mlp|sequence|autoencoder,
    "onnx": file, "scaler": file?, "window_size": int?, "mse_threshold": float?}
    with files relative to the models directory. Feature/channel order comes
    from ``card["feature_names"]``.
    """
    serving: dict[str, Any] = card["serving"]
    models_dir = settings.models_dir
    onnx_path = models_dir / str(serving["onnx"])
    names = [str(n) for n in card.get("feature_names") or []]
    kind = str(serving["adapter"])
    logger.info(
        "[RF-10] Loading model '%s' from card | adapter=%s | onnx=%s | inputs=%d",
        model_name,
        kind,
        onnx_path.name,
        len(names),
    )
    if kind == "tree":
        from src.services.onnx_tree_adapter import OnnxTreeAdapter

        adapter: Any = OnnxTreeAdapter(onnx_path=onnx_path, feature_names=names)
    elif kind == "mlp":
        from src.services.mlp_adapter import OnnxMlpAdapter

        adapter = OnnxMlpAdapter(
            onnx_path=onnx_path,
            scaler_path=models_dir / str(serving["scaler"]),
            feature_names=names,
        )
    elif kind == "sequence":
        from src.services.onnx_sequence_adapter import OnnxSequenceAdapter

        adapter = OnnxSequenceAdapter(
            onnx_path=onnx_path,
            scaler_path=models_dir / str(serving["scaler"]),
            window_size=int(serving.get("window_size", 60)),
            channel_names=names,
        )
    elif kind == "autoencoder":
        from src.services.onnx_autoencoder_adapter import OnnxAutoencoderAdapter

        adapter = OnnxAutoencoderAdapter(
            onnx_path=onnx_path,
            scaler_path=models_dir / str(serving["scaler"]),
            mse_threshold=float(serving["mse_threshold"]),
            window_size=int(serving.get("window_size", 60)),
            channel_names=names,
        )
    else:
        raise ValueError(f"Unknown serving adapter {kind!r} in {model_name} card.")
    return ModelService(adapter, decision_threshold=threshold, model_name=model_name)


def resolve_active_model(name: str) -> str:
    """``ACTIVE_MODEL`` as served: a legacy or unknown name (e.g. an old
    ``.env`` with ``random_forest_v2``) falls back to the champion with a
    warning instead of failing the startup."""
    normalized = name.lower().strip()
    if normalized in SERVED_MODELS:
        return normalized
    logger.warning(
        "ACTIVE_MODEL=%r is not served (served: %s) — using %s",
        name,
        ", ".join(SERVED_MODELS),
        SERVED_MODELS[0],
    )
    return SERVED_MODELS[0]


def load_active_model() -> "ModelService":
    """
    Load whichever model is selected by ``settings.active_model`` (RF-10).

    Thin wrapper around ``load_model_by_name`` for startup use.  The runtime
    hot-swap path calls ``load_model_by_name`` directly via ``ModelRegistry``.
    """
    return load_model_by_name(resolve_active_model(settings.active_model))


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
    model_name:
        Nome do modelo em `ModelRegistry.KNOWN_MODELS` (ex. "random_forest").
        Usado exclusivamente como label de baixa cardinalidade nas métricas
        Prometheus (RNF-76) — nunca afeta o comportamento de inferência.
        Default "unknown" cobre apenas chamadas diretas de teste que não
        passam por `load_model_by_name`.
    """

    def __init__(
        self,
        model: Any,
        decision_threshold: float = _DEFAULT_THRESHOLD,
        model_name: str = "unknown",
    ) -> None:
        self._model = model
        self._threshold: float = decision_threshold
        self._model_name: str = model_name
        # Mapeia as colunas que o modelo realmente espera para evitar KeyError
        self._expected_features = list(self._model.feature_names_in_)

    @property
    def decision_threshold(self) -> float:
        """Active classification cut-off."""
        return self._threshold

    @property
    def history_rows(self) -> int:
        """Rows of history the model consumes per prediction.

        Sequence and autoencoder adapters read a window (``_window_size``);
        tabular models read only the most recent engineered row (0)."""
        return int(getattr(self._model, "_window_size", 0) or 0)

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
        start = time.perf_counter()
        try:
            X = self._align_columns(X)
            failure_probability: float = float(self._model.predict_proba(X)[0][1])
            predicted_class: int = int(failure_probability >= self._threshold)

            response = PredictResponse(
                predicted_class=predicted_class,
                failure_probability=round(failure_probability, 6),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        except Exception:
            # Bare `raise` preserves the original traceback (PEP 8 / B904).
            logger.exception("Erro na predição")
            record_inference(self._model_name, "error", time.perf_counter() - start)
            raise

        record_inference(
            self._model_name,
            "success",
            time.perf_counter() - start,
            predicted_class=predicted_class,
        )
        return response

    def predict_batch(self, requests: list[PredictRequest]) -> list[PredictResponse]:
        """
        Predição em lote — RNF-73.

        Constrói UMA matriz de features [N, features] (concatena a linha
        stateless de cada request — mesma :meth:`_build_feature_row` usada
        por :meth:`predict`, preprocessing por amostra é aceitável, ver
        RNF-73 §"Se alguma etapa do preprocessing exigir processamento
        individual") e chama ``predict_proba`` **uma única vez** para as N
        amostras — a MESMA chamada vetorizada que
        ``OnnxTreeAdapter.predict_proba``/``RandomForestClassifier.predict_proba``
        já fazem para qualquer número de linhas (nenhuma mudança nos
        adapters foi necessária). NUNCA um laço chamando :meth:`predict` N
        vezes — isso reduziria a N chamadas de inferência, exatamente o que
        RNF-73 pede para evitar.

        Levanta o mesmo tipo de exceção que :meth:`predict` em caso de erro
        (nenhum tratamento silencioso) — o chamador (router) decide como
        traduzir para HTTP.

        Lista vazia devolve lista vazia — sem isso, ``pd.concat([])``
        levanta ``ValueError`` genérico. A rota HTTP já rejeita batch vazio
        via ``BatchPredictRequest(min_length=1)`` (422 antes de chegar
        aqui); este guard é para o método continuar correto/testável
        isoladamente, sem depender do chamador validar primeiro.
        """
        if not requests:
            return []

        rows = [self._build_feature_row(req) for req in requests]
        X = pd.concat(rows, ignore_index=True)
        X = self._align_columns(X)

        try:
            probabilities: np.ndarray = np.asarray(self._model.predict_proba(X))
        except Exception:
            logger.exception("Erro na predição em lote")
            record_batch(self._model_name, "error", len(requests))
            raise

        timestamp = datetime.now(timezone.utc).isoformat()
        responses: list[PredictResponse] = []
        for row_probs in probabilities:
            failure_probability = float(row_probs[1])
            predicted_class = int(failure_probability >= self._threshold)
            responses.append(
                PredictResponse(
                    predicted_class=predicted_class,
                    failure_probability=round(failure_probability, 6),
                    timestamp=timestamp,
                )
            )
        record_batch(self._model_name, "success", len(requests))
        return responses

    def _align_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Garante que TODAS as colunas esperadas pelo modelo existam, na
        ordem exata do treino (drop extras) — compartilhado por
        :meth:`predict_from_features` e :meth:`predict_batch` (extraído
        nesta task, RNF-72/73, para não duplicar a lógica de alinhamento
        entre o caminho single e o batch).
        """
        X = X.copy()  # never mutate caller's DataFrame
        for col in self._expected_features:
            if col not in X.columns:
                # LPS / Caudal / Pressure_switch existem no schema antigo
                # com valor lógico "1 = OK"; o resto cai pra zero neutro.
                X[col] = (
                    1.0 if col in {"LPS", "Pressure_switch", "Caudal_impulses"} else 0.0
                )
        return X[self._expected_features]

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
