"""
OnnxMlpAdapter — sklearn-compatible wrapper for the MLP ONNX model (RNF-24, RF-10).

This adapter bridges the gap between the ONNX Runtime inference session and the
interface contract expected by `ModelService`:

    • `.feature_names_in_`  — ndarray of 34 feature names (same order as training).
    • `.predict(X)`         — returns ndarray of shape (n,) with values in {0, 1}.
    • `.predict_proba(X)`   — returns ndarray of shape (n, 2), values in [0.0, 1.0].

Scaling strategy
----------------
Neural networks require normalised inputs.  The StandardScaler artefact is kept
**separate** from the ONNX graph so it can be updated independently (e.g., after
retraining on new data without recompiling the graph).

The scaling is applied inside the adapter, transparently to `ModelService.predict()`,
which hands over a DataFrame already aligned to the 34-feature contract.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger: logging.Logger = logging.getLogger(__name__)

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


class OnnxMlpAdapter:
    """
    Production adapter that wraps `mlp_v1.onnx` + `mlp_scaler.joblib`
    behind the sklearn predict / predict_proba interface.

    Parameters
    ----------
    onnx_path : Path
        Absolute path to the exported `mlp_v1.onnx` artefact.
    scaler_path : Path
        Absolute path to the `mlp_scaler.joblib` StandardScaler artefact.
    """

    # Exposes the 34 feature names that ModelService reads at init to
    # build `_expected_features`.
    feature_names_in_: np.ndarray = np.array(_FEATURE_NAMES, dtype=object)

    def __init__(self, onnx_path: Path, scaler_path: Path) -> None:
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"MLP ONNX artefact not found at {onnx_path}. "
                "Run `python src/train_mlp.py` first."
            )
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"MLP scaler artefact not found at {scaler_path}. "
                "Run `python src/train_mlp.py` first."
            )

        # V2 — ORT graph optimisation (constant folding, layer fusion).
        # Single-row inference: 1 intra-op thread is enough; sequential mode
        # avoids thread-pool churn and gives the lowest p95 latency.
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1

        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name: str = self._session.get_inputs()[0].name
        self._scaler: StandardScaler = joblib.load(scaler_path)

        logger.info(
            "[RNF-24] OnnxMlpAdapter loaded | onnx=%s | scaler=%s",
            onnx_path.name,
            scaler_path.name,
        )

    # ------------------------------------------------------------------
    # Public sklearn-compatible interface
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute class probabilities.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with the 34 contracted columns (already aligned by
            ModelService before this call).

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            Columns are [P(class=0), P(class=1)].  Values are in [0.0, 1.0]
            and each row sums to 1.0.
        """
        scaled: np.ndarray = self._scale(X)
        logits: np.ndarray = self._session.run(None, {self._input_name: scaled})[0]
        return self._softmax(logits)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict binary class label (0 = normal, 1 = fault).

        Returns
        -------
        np.ndarray of shape (n_samples,) with values in {0, 1}.
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(np.int64)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scale(self, X: pd.DataFrame) -> np.ndarray:
        """Apply StandardScaler and return a float32 C-contiguous array."""
        arr = X.to_numpy(dtype=np.float32)
        scaled = self._scaler.transform(arr)
        # ONNX Runtime requires C-contiguous float32
        return np.ascontiguousarray(scaled, dtype=np.float32)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax along axis 1."""
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)
