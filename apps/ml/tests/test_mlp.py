"""
Smoke tests for the MLP ONNX artefact — RNF-24 interface contract.

Strategy
--------
These tests load `mlp_v1.onnx` + `mlp_scaler.joblib` directly via ONNX Runtime
and verify that the combined inference pipeline respects the *same output contract*
as the Random Forest and XGBoost models, so that the backend `ModelService`
(via `OnnxMlpAdapter`) can swap in the MLP without code changes.

All tests are skipped when artefacts do not yet exist (pre-training).
Run `python src/train_mlp.py --max-epochs 10` to generate them quickly.

Interface contract verified:
  • ONNX session accepts a float32 tensor of shape (n, 34).
  • Raw output (logits) has shape (n, 2).
  • After softmax: each row sums to 1.0 ± 1e-5, all values in [0, 1].
  • predict_proba equivalent returns shape (1, 2) for a single row.
  • Failure probability = proba[0][1] is in [0.0, 1.0].
  • Single-row inference completes within 500 ms (p95).
  • StandardScaler transforms successfully before inference.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

# ── paths ─────────────────────────────────────────────────────────────────────
_ML_ROOT: Path = Path(__file__).resolve().parents[1]
_ONNX_PATH: Path = _ML_ROOT / "models" / "mlp_v1.onnx"
_SCALER_PATH: Path = _ML_ROOT / "models" / "mlp_scaler.joblib"

_FEATURE_NAMES: list[str] = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Motor_current",
    "Oil_temperature",
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "Oil_level",
    "TP2_delta",
    "TP2_std_5",
    "TP3_std_5",
    "H1_std_5",
    "DV_pressure_std_5",
    "Reservoirs_std_5",
    "Oil_temperature_std_5",
    "Motor_current_std_5",
    "TP2_ma_5",
    "TP2_ma_15",
    "TP3_ma_5",
    "TP3_ma_15",
    "H1_ma_5",
    "H1_ma_15",
    "DV_pressure_ma_5",
    "DV_pressure_ma_15",
    "Reservoirs_ma_5",
    "Reservoirs_ma_15",
    "Oil_temperature_ma_5",
    "Oil_temperature_ma_15",
    "Motor_current_ma_5",
    "Motor_current_ma_15",
]

_N_FEATURES: int = len(_FEATURE_NAMES)  # 34

pytestmark = pytest.mark.skipif(
    not (_ONNX_PATH.exists() and _SCALER_PATH.exists()),
    reason=(
        f"Artefacts not found: {_ONNX_PATH.name}, {_SCALER_PATH.name}. "
        "Run `python src/train_mlp.py --max-epochs 10` first."
    ),
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def session() -> ort.InferenceSession:
    return ort.InferenceSession(str(_ONNX_PATH), providers=["CPUExecutionProvider"])


@pytest.fixture(scope="module")
def scaler() -> StandardScaler:
    return joblib.load(_SCALER_PATH)


@pytest.fixture(scope="module")
def input_name(session: ort.InferenceSession) -> str:
    return session.get_inputs()[0].name


@pytest.fixture(scope="module")
def raw_single_row() -> pd.DataFrame:
    """Unscaled single-row input: all values 1.0 (neutral sensor reading)."""
    return pd.DataFrame(
        np.ones((1, _N_FEATURES), dtype=np.float32),
        columns=_FEATURE_NAMES,
    )


@pytest.fixture(scope="module")
def scaled_single_row(
    scaler: StandardScaler, raw_single_row: pd.DataFrame
) -> np.ndarray:
    return np.ascontiguousarray(scaler.transform(raw_single_row), dtype=np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Artefact loading
# ---------------------------------------------------------------------------


def test_onnx_model_loads() -> None:
    """ONNX model must load without errors."""
    sess = ort.InferenceSession(str(_ONNX_PATH), providers=["CPUExecutionProvider"])
    assert sess is not None


def test_scaler_loads() -> None:
    """StandardScaler must load and have 34 features."""
    sc = joblib.load(_SCALER_PATH)
    assert hasattr(sc, "mean_"), "Scaler has no mean_ — not fitted"
    assert sc.mean_.shape == (
        _N_FEATURES,
    ), f"Expected scaler with {_N_FEATURES} features, got {sc.mean_.shape[0]}"


# ---------------------------------------------------------------------------
# ONNX session metadata
# ---------------------------------------------------------------------------


def test_onnx_input_has_correct_feature_count(session: ort.InferenceSession) -> None:
    """ONNX input must expect 34 features."""
    input_shape = session.get_inputs()[0].shape
    # Shape is [batch_size, n_features]; batch_size is dynamic (None or str)
    assert (
        input_shape[1] == _N_FEATURES
    ), f"Expected {_N_FEATURES} input features, got {input_shape[1]}"


def test_onnx_output_has_two_classes(session: ort.InferenceSession) -> None:
    """ONNX output (logits) must have 2 class dimensions."""
    output_shape = session.get_outputs()[0].shape
    assert output_shape[1] == 2, f"Expected 2 output logits, got {output_shape[1]}"


# ---------------------------------------------------------------------------
# Scaler + ONNX pipeline (single row)
# ---------------------------------------------------------------------------


def test_scaler_transforms_without_error(
    scaler: StandardScaler, raw_single_row: pd.DataFrame
) -> None:
    """Scaler must transform a (1, 34) DataFrame without raising."""
    result = scaler.transform(raw_single_row)
    assert result.shape == (1, _N_FEATURES)


def test_predict_proba_single_row_shape(
    session: ort.InferenceSession, scaled_single_row: np.ndarray, input_name: str
) -> None:
    """Logits must have shape (1, 2) for a single-row input."""
    logits = session.run(None, {input_name: scaled_single_row})[0]
    proba = _softmax(logits)
    assert proba.shape == (1, 2), f"Expected (1, 2), got {proba.shape}"


def test_predict_proba_sums_to_one(
    session: ort.InferenceSession, scaled_single_row: np.ndarray, input_name: str
) -> None:
    """Class probabilities must sum to 1.0 (± floating-point tolerance)."""
    logits = session.run(None, {input_name: scaled_single_row})[0]
    proba = _softmax(logits)
    total = float(proba[0].sum())
    assert abs(total - 1.0) < 1e-5, f"Probabilities sum to {total}, expected 1.0"


def test_predict_proba_values_in_unit_range(
    session: ort.InferenceSession, scaled_single_row: np.ndarray, input_name: str
) -> None:
    """Each probability must be in [0.0, 1.0]."""
    logits = session.run(None, {input_name: scaled_single_row})[0]
    proba = _softmax(logits)
    assert (proba >= 0.0).all() and (
        proba <= 1.0
    ).all(), (
        f"Probabilities out of [0, 1]: min={proba.min():.4f}, max={proba.max():.4f}"
    )


def test_failure_probability_index(
    session: ort.InferenceSession, scaled_single_row: np.ndarray, input_name: str
) -> None:
    """ModelService reads predict_proba(X)[0][1] as the failure probability."""
    logits = session.run(None, {input_name: scaled_single_row})[0]
    proba = _softmax(logits)
    failure_prob = float(proba[0][1])
    assert (
        0.0 <= failure_prob <= 1.0
    ), f"Failure probability {failure_prob} out of range"


def test_predict_returns_binary_label(
    session: ort.InferenceSession, scaled_single_row: np.ndarray, input_name: str
) -> None:
    """Argmax of logits (i.e., predict) must return 0 or 1."""
    logits = session.run(None, {input_name: scaled_single_row})[0]
    predicted_class = int(np.argmax(logits[0]))
    assert predicted_class in {0, 1}, f"Expected 0 or 1, got {predicted_class}"


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------


def test_batch_inference_shape(
    session: ort.InferenceSession, scaler: StandardScaler, input_name: str
) -> None:
    """Batch of 32 rows must produce logits of shape (32, 2)."""
    batch = pd.DataFrame(
        np.ones((32, _N_FEATURES), dtype=np.float32),
        columns=_FEATURE_NAMES,
    )
    batch_scaled = np.ascontiguousarray(scaler.transform(batch), dtype=np.float32)
    logits = session.run(None, {input_name: batch_scaled})[0]
    assert logits.shape == (32, 2), f"Expected (32, 2), got {logits.shape}"


# ---------------------------------------------------------------------------
# Latency guard (p95 < 500 ms for a single row)
# ---------------------------------------------------------------------------


def test_single_row_latency_p95_under_500ms(
    session: ort.InferenceSession, scaled_single_row: np.ndarray, input_name: str
) -> None:
    """
    Single-row inference must complete in < 500 ms (p95).

    This matches the contract enforced for RF and XGBoost models, ensuring
    the MLP ONNX pipeline is fast enough for interactive dashboards.
    """
    times: list[float] = []
    for _ in range(50):
        t0 = time.perf_counter()
        session.run(None, {input_name: scaled_single_row})
        times.append((time.perf_counter() - t0) * 1_000)

    p95 = float(np.percentile(times, 95))
    assert p95 < 500.0, f"Inference p95 = {p95:.1f} ms exceeds 500 ms threshold"
