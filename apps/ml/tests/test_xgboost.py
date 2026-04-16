"""
Smoke tests for the XGBoost model artefact — RF-10 interface contract.

Strategy
--------
These tests verify that `xgboost_v1.joblib` exposes the *exact same prediction
interface* as the Random Forest, so that `ModelService` (backend) can load
either model without code changes.

All tests are skipped when the artefact does not yet exist (pre-training).
Run `python src/train_xgboost.py --n-trials 5` to generate the artefact.

Interface contract (must match `model_service.py` assumptions):
  • model.feature_names_in_  — 34 feature names in the correct order.
  • model.predict(X)         — returns ndarray of int {0, 1}.
  • model.predict_proba(X)   — returns ndarray of shape (n, 2), values in [0, 1].
  • Prediction on a single row completes within 500 ms (p95 guard).
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

# ── paths ─────────────────────────────────────────────────────────────────────
_ML_ROOT: Path = Path(__file__).resolve().parents[1]
_MODEL_PATH: Path = _ML_ROOT / "models" / "xgboost_v1.joblib"

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

_SAMPLE_ROW: dict = {f: 1.0 for f in _FEATURE_NAMES}

pytestmark = pytest.mark.skipif(
    not _MODEL_PATH.exists(),
    reason=f"Artefact not found: {_MODEL_PATH}. Run `python src/train_xgboost.py --n-trials 5` first.",
)


@pytest.fixture(scope="module")
def model():
    return joblib.load(_MODEL_PATH)


@pytest.fixture(scope="module")
def single_row(model) -> pd.DataFrame:
    return pd.DataFrame([_SAMPLE_ROW])[model.feature_names_in_].astype(np.float32)


# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------


def test_model_has_feature_names_in(model) -> None:
    """ModelService reads model.feature_names_in_ — must exist and be non-empty."""
    assert hasattr(model, "feature_names_in_"), "Missing feature_names_in_ attribute"
    assert len(model.feature_names_in_) > 0


def test_model_has_exactly_34_features(model) -> None:
    """Backend ModelService._build_feature_row always produces 34 columns."""
    assert (
        len(model.feature_names_in_) == 34
    ), f"Expected 34 features, got {len(model.feature_names_in_)}"


def test_feature_names_match_contract(model) -> None:
    """Feature order must be identical to the RF contract (model_service.py)."""
    actual = list(model.feature_names_in_)
    assert (
        actual == _FEATURE_NAMES
    ), f"Feature mismatch.\nExpected: {_FEATURE_NAMES}\nGot:      {actual}"


# ---------------------------------------------------------------------------
# predict() interface
# ---------------------------------------------------------------------------


def test_predict_returns_ndarray(model, single_row) -> None:
    """predict() must return a numpy array (not a list or scalar)."""
    result = model.predict(single_row)
    assert isinstance(result, np.ndarray)


def test_predict_single_row_shape(model, single_row) -> None:
    """predict() on a single row must return shape (1,)."""
    result = model.predict(single_row)
    assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"


def test_predict_output_is_binary(model, single_row) -> None:
    """predict() must return 0 or 1 (binary fault classification)."""
    result = model.predict(single_row)
    assert result[0] in {0, 1}, f"Expected 0 or 1, got {result[0]}"


# ---------------------------------------------------------------------------
# predict_proba() interface
# ---------------------------------------------------------------------------


def test_predict_proba_returns_ndarray(model, single_row) -> None:
    """predict_proba() must return a numpy array."""
    result = model.predict_proba(single_row)
    assert isinstance(result, np.ndarray)


def test_predict_proba_single_row_shape(model, single_row) -> None:
    """predict_proba() on a single row must return shape (1, 2)."""
    result = model.predict_proba(single_row)
    assert result.shape == (1, 2), f"Expected shape (1, 2), got {result.shape}"


def test_predict_proba_sums_to_one(model, single_row) -> None:
    """Class probabilities must sum to 1.0 (± floating-point tolerance)."""
    result = model.predict_proba(single_row)
    total = float(result[0].sum())
    assert abs(total - 1.0) < 1e-5, f"Probabilities sum to {total}, expected 1.0"


def test_predict_proba_values_in_unit_range(model, single_row) -> None:
    """Each probability must be in [0.0, 1.0]."""
    result = model.predict_proba(single_row)
    assert (result >= 0.0).all() and (result <= 1.0).all()


def test_failure_probability_index(model, single_row) -> None:
    """ModelService uses predict_proba(X)[0][1] as the failure probability."""
    result = model.predict_proba(single_row)
    failure_prob = float(result[0][1])
    assert 0.0 <= failure_prob <= 1.0


# ---------------------------------------------------------------------------
# Latency guard (p95 < 500 ms for a single row)
# ---------------------------------------------------------------------------


def test_single_row_latency_p95_under_500ms(model, single_row) -> None:
    """
    Single-row inference must complete in < 500 ms (p95).

    This is a soft guard — not a hard SLA — to catch regressions where a
    misconfigured model (e.g., 10 k estimators) would make the endpoint
    too slow for interactive dashboards.
    """
    times: list[float] = []
    for _ in range(50):
        t0 = time.perf_counter()
        model.predict_proba(single_row)
        times.append((time.perf_counter() - t0) * 1_000)

    p95 = float(np.percentile(times, 95))
    assert p95 < 500.0, f"Inference p95 = {p95:.1f} ms exceeds 500 ms threshold"
