"""
Smoke tests for Random Forest training artefacts — RF-04 / RNF-10.

What is verified
----------------
1. Both artefact files exist (rf_model.joblib, model_card.json).
2. model_card.json has the required structure and correct field types.
3. [RF-04] F1-Score on class 1 (fault) recorded in the card is >= 0.75.
4. The loaded model exposes predict / predict_proba and produces valid output.

These tests assume the artefacts were produced by a successful training run.
All fixtures skip gracefully when the artefacts are absent, so the suite can
be executed in CI before the first training run without hard failures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
MODEL_PATH: Path = _ML_ROOT / "models" / "random_forest_final.joblib"
MODEL_CARD_PATH: Path = _ML_ROOT / "models" / "model_card.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model() -> Any:
    """Load the serialised Random Forest model, skip if not present."""
    if not MODEL_PATH.exists():
        pytest.skip(f"Model artefact not found: {MODEL_PATH}")
    import joblib  # local import keeps the module importable without joblib

    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def model_card() -> dict[str, Any]:
    """Load model_card.json, skip if not present."""
    if not MODEL_CARD_PATH.exists():
        pytest.skip(f"Model card not found: {MODEL_CARD_PATH}")
    with MODEL_CARD_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Artefact existence
# ---------------------------------------------------------------------------


class TestArtefactExistence:
    def test_model_file_exists(self) -> None:
        """rf_model.joblib must be present after a successful training run."""
        assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"

    def test_model_card_file_exists(self) -> None:
        """model_card.json must be present after a successful training run."""
        assert MODEL_CARD_PATH.exists(), f"Model card not found: {MODEL_CARD_PATH}"

    def test_model_file_is_not_empty(self) -> None:
        if not MODEL_PATH.exists():
            pytest.skip("Model file absent")
        assert MODEL_PATH.stat().st_size > 0, "Model file is empty."

    def test_model_card_is_valid_json(self) -> None:
        if not MODEL_CARD_PATH.exists():
            pytest.skip("Model card absent")
        with MODEL_CARD_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert isinstance(data, dict), "model_card.json must be a JSON object."


# ---------------------------------------------------------------------------
# Model card structure
# ---------------------------------------------------------------------------


class TestModelCard:
    REQUIRED_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
        {
            "schema_version",
            "trained_at",
            "model_type",
            "dataset",
            "target_column",
            "train_size",
            "test_size",
            "best_hyperparameters",
            "metrics",
            "feature_names",
            "feature_count",
        }
    )
    REQUIRED_METRICS_KEYS: frozenset[str] = frozenset(
        {"f1_class1_test", "roc_auc_test", "classification_report"}
    )

    def test_required_top_level_keys_present(self, model_card: dict[str, Any]) -> None:
        missing = self.REQUIRED_TOP_LEVEL_KEYS - model_card.keys()
        assert not missing, f"Missing keys in model_card.json: {missing}"

    def test_required_metrics_keys_present(self, model_card: dict[str, Any]) -> None:
        missing = self.REQUIRED_METRICS_KEYS - model_card["metrics"].keys()
        assert not missing, f"Missing metrics keys: {missing}"

    def test_schema_version_is_string(self, model_card: dict[str, Any]) -> None:
        assert isinstance(model_card["schema_version"], str)

    def test_model_type_is_random_forest(self, model_card: dict[str, Any]) -> None:
        assert model_card["model_type"] == "RandomForestClassifier"

    def test_dataset_field_is_metropt3(self, model_card: dict[str, Any]) -> None:
        assert model_card["dataset"] == "MetroPT-3"

    def test_target_column_is_anomaly(self, model_card: dict[str, Any]) -> None:
        assert model_card["target_column"] == "anomaly"

    def test_train_size_is_positive_int(self, model_card: dict[str, Any]) -> None:
        assert isinstance(model_card["train_size"], int)
        assert model_card["train_size"] > 0

    def test_test_size_is_positive_int(self, model_card: dict[str, Any]) -> None:
        assert isinstance(model_card["test_size"], int)
        assert model_card["test_size"] > 0

    def test_feature_count_matches_feature_names_length(
        self, model_card: dict[str, Any]
    ) -> None:
        assert model_card["feature_count"] == len(model_card["feature_names"])

    def test_feature_names_is_non_empty_list(self, model_card: dict[str, Any]) -> None:
        assert isinstance(model_card["feature_names"], list)
        assert len(model_card["feature_names"]) > 0

    def test_best_hyperparameters_contains_n_estimators(
        self, model_card: dict[str, Any]
    ) -> None:
        # best_hyperparameters stores GridSearchCV best_params_ (prefixed with 'rf__')
        params: dict[str, Any] = model_card["best_hyperparameters"]
        assert any(
            k in params for k in ("rf__n_estimators", "n_estimators")
        ), f"Neither 'rf__n_estimators' nor 'n_estimators' found in best_hyperparameters: {params}"

    def test_rf04_f1_class1_meets_threshold(self, model_card: dict[str, Any]) -> None:
        """[RF-04] F1-Score on class 1 (fault) must be >= 0.75."""
        f1: float = model_card["metrics"]["f1_class1_test"]
        assert (
            f1 >= 0.75
        ), f"[RF-04] F1-Score class 1 = {f1:.4f} is below the 0.75 requirement."

    def test_roc_auc_above_random_baseline(self, model_card: dict[str, Any]) -> None:
        roc_auc: float = model_card["metrics"]["roc_auc_test"]
        assert roc_auc > 0.5, f"ROC-AUC = {roc_auc:.4f} is at or below random baseline."

    def test_f1_class1_is_float_in_range(self, model_card: dict[str, Any]) -> None:
        f1: float = model_card["metrics"]["f1_class1_test"]
        assert isinstance(f1, float)
        assert 0.0 <= f1 <= 1.0


# ---------------------------------------------------------------------------
# Model behaviour
# ---------------------------------------------------------------------------


class TestModelBehaviour:
    def test_model_has_predict_method(self, model: Any) -> None:
        """Loaded model must expose a callable predict method."""
        assert callable(getattr(model, "predict", None))

    def test_model_has_predict_proba_method(self, model: Any) -> None:
        """Model must expose predict_proba for downstream probability calibration."""
        assert callable(getattr(model, "predict_proba", None))

    def test_model_has_feature_importances(self, model: Any) -> None:
        """RandomForestClassifier must expose feature_importances_ after fitting."""
        assert hasattr(model, "feature_importances_")
        assert model.feature_importances_ is not None

    def test_predict_returns_binary_labels(
        self, model: Any, model_card: dict[str, Any]
    ) -> None:
        """predict must return only 0s and 1s for any numeric input."""
        rng = np.random.default_rng(0)
        X_dummy = pd.DataFrame(
            rng.uniform(size=(20, model_card["feature_count"])).astype("float32"),
            columns=model.feature_names_in_,
        )
        y_pred: np.ndarray = model.predict(X_dummy)
        assert set(y_pred.tolist()).issubset(
            {0, 1}
        ), f"predict returned unexpected labels: {set(y_pred.tolist())}"

    def test_predict_output_length_matches_input(
        self, model: Any, model_card: dict[str, Any]
    ) -> None:
        rng = np.random.default_rng(1)
        X_dummy = pd.DataFrame(
            rng.uniform(size=(15, model_card["feature_count"])).astype("float32"),
            columns=model.feature_names_in_,
        )
        y_pred: np.ndarray = model.predict(X_dummy)
        assert len(y_pred) == 15

    def test_predict_proba_shape_is_n_samples_by_2(
        self, model: Any, model_card: dict[str, Any]
    ) -> None:
        rng = np.random.default_rng(2)
        X_dummy = pd.DataFrame(
            rng.uniform(size=(8, model_card["feature_count"])).astype("float32"),
            columns=model.feature_names_in_,
        )
        proba: np.ndarray = model.predict_proba(X_dummy)
        assert proba.shape == (8, 2), f"Expected (8, 2), got {proba.shape}"

    def test_predict_proba_rows_sum_to_one(
        self, model: Any, model_card: dict[str, Any]
    ) -> None:
        rng = np.random.default_rng(3)
        X_dummy = pd.DataFrame(
            rng.uniform(size=(10, model_card["feature_count"])).astype("float32"),
            columns=model.feature_names_in_,
        )
        proba: np.ndarray = model.predict_proba(X_dummy)
        np.testing.assert_allclose(
            proba.sum(axis=1),
            np.ones(10),
            atol=1e-6,
            err_msg="predict_proba rows must sum to 1.0",
        )

    def test_feature_importances_sum_to_one(self, model: Any) -> None:
        total: float = float(np.sum(model.feature_importances_))
        assert (
            abs(total - 1.0) < 1e-5
        ), f"feature_importances_ sum = {total:.6f}, expected ~1.0"
