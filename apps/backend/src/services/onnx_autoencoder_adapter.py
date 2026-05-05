"""
OnnxAutoencoderAdapter — backend adapter for the Conv1D Autoencoder (RF-10).

Anomaly detection via reconstruction error
------------------------------------------
Unlike classifier models that output class probabilities directly, the
autoencoder outputs a *reconstruction* of the input window.  The anomaly
score is the Mean Squared Error (MSE) between the input and the output:

    mse = mean((reconstruction - input)²)   over axes T and C

Windows from the healthy operational regime were seen during training and
are reconstructed with low MSE.  Windows from fault periods are *novel* to
the model and yield high MSE.

Score normalisation
-------------------
``ModelService`` expects probabilities in [0, 1] for the positive (fault)
class.  The adapter maps MSE → probability via a sigmoid centred on the
calibrated threshold:

    score = sigmoid((mse - mse_threshold) / scale)
    scale = mse_threshold / 3

This gives:
  * mse = mse_threshold  →  score = 0.5  (the decision boundary)
  * mse = 2 × threshold  →  score ≈ 0.95 (strong anomaly signal)
  * mse = 0              →  score ≈ 0.05 (very healthy)

The matching ``decision_threshold`` in the model card is 0.5 (the default),
so ``ModelService`` fires the alarm exactly when ``mse ≥ mse_threshold``.

Window assembly
---------------
Follows the same cold-start padding strategy as ``OnnxSequenceAdapter``:
* If X has ≥ window_size rows → use the last T rows as the inference window.
* If X has < window_size rows → replicate the oldest row leftward until T is
  reached (neutral, non-alarmist padding for the first few ticks).

ONNX artefact
-------------
The exported graph accepts (B, T, C) and returns (B, T, C).  MSE is
computed in scaled space (same space the model was trained in).
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

_RAW_CHANNELS: list[str] = [
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
]

# Sigmoid scale factor: mse = 2×threshold yields score ≈ 0.95.
_SIGMOID_SCALE_FACTOR: float = 3.0


class OnnxAutoencoderAdapter:
    """
    Production adapter wrapping the Conv1D Autoencoder ONNX graph.

    Exposes the same ``predict / predict_proba / feature_names_in_`` interface
    as all other adapters so ``ModelService`` can hot-swap this model without
    any router or service changes.

    Parameters
    ----------
    onnx_path :
        Path to ``autoencoder_v1.onnx``.
    scaler_path :
        Path to ``autoencoder_scaler.joblib`` (per-channel StandardScaler
        fitted on healthy training windows only).
    mse_threshold :
        The calibrated reconstruction error boundary.  Loaded from
        ``autoencoder_v1_card.json`` by ``model_service.load_model_by_name``
        so this class stays artefact-version-agnostic.
    window_size :
        Number of time steps the model expects (T axis).  Default 60.
    channel_names :
        Ordered list of sensor channel names.  Defaults to the 12 raw
        MetroPT-3 channels if not supplied.
    """

    feature_names_in_: np.ndarray

    def __init__(
        self,
        onnx_path: Path,
        scaler_path: Path,
        mse_threshold: float,
        window_size: int = 60,
        channel_names: list[str] | None = None,
    ) -> None:
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"Autoencoder ONNX artefact not found: {onnx_path}. "
                "Run `python src/train_autoencoder.py` first."
            )
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Autoencoder scaler artefact not found: {scaler_path}. "
                "Run `python src/train_autoencoder.py` first."
            )
        if mse_threshold <= 0.0:
            raise ValueError(
                f"mse_threshold must be positive, got {mse_threshold}. "
                "Check the model card."
            )

        sess_opts: ort.SessionOptions = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1

        self._session: ort.InferenceSession = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name: str = self._session.get_inputs()[0].name

        self._scaler: StandardScaler = joblib.load(scaler_path)
        self._mse_threshold: float = mse_threshold
        self._window_size: int = window_size
        self._channels: list[str] = (
            list(channel_names) if channel_names is not None else list(_RAW_CHANNELS)
        )

        self.feature_names_in_: np.ndarray = np.array(self._channels, dtype=object)

        logger.info(
            "[RF-10] OnnxAutoencoderAdapter loaded | onnx=%s | T=%d | C=%d | "
            "mse_threshold=%.6f",
            onnx_path.name,
            window_size,
            len(self._channels),
            mse_threshold,
        )

    # ------------------------------------------------------------------
    # sklearn-compatible inference interface
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute anomaly probability via reconstruction error.

        Parameters
        ----------
        X : pd.DataFrame
            Single-row snapshot (cold start) or multi-row DataFrame with the
            sensor history buffered by ``InferencePipelineService``.  The
            last ``window_size`` rows are used as the inference window;
            shorter inputs are padded on the left via cold-start replication.

        Returns
        -------
        np.ndarray of shape (1, 2)
            Column 0 = P(healthy), Column 1 = P(fault).
            Column 1 ≥ 0.5 ↔ mse ≥ mse_threshold.
        """
        window: np.ndarray = self._build_window(X)  # (1, T, C) raw values
        scaled: np.ndarray = self._scale(window)  # (1, T, C) normalised

        reconstruction: np.ndarray = self._session.run(
            None, {self._input_name: scaled}
        )[
            0
        ]  # (1, T, C)

        # Per-sample MSE in scaled space (same space the model was trained in).
        mse: float = float(np.mean((reconstruction - scaled) ** 2))

        fault_prob: float = self._sigmoid_score(mse)
        return np.array([[1.0 - fault_prob, fault_prob]], dtype=np.float32)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict binary fault class (0 = normal, 1 = fault).

        Returns
        -------
        np.ndarray of shape (1,) with values in {0, 1}.
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_window(self, X: pd.DataFrame) -> np.ndarray:
        """Assemble a (1, T, C) window from a sensor history DataFrame."""
        cols_present: list[str] = [c for c in self._channels if c in X.columns]
        arr: np.ndarray = X[cols_present].to_numpy(dtype=np.float32)

        t: int = self._window_size
        if len(arr) >= t:
            window: np.ndarray = arr[-t:]
        else:
            n_pad: int = t - len(arr)
            pad: np.ndarray = np.repeat(arr[:1], n_pad, axis=0)
            window = np.concatenate([pad, arr], axis=0)

        return window[np.newaxis].astype(np.float32)  # (1, T, C)

    def _scale(self, window: np.ndarray) -> np.ndarray:
        """Apply the per-channel StandardScaler fitted on healthy training windows."""
        b, t, c = window.shape
        scaled: np.ndarray = self._scaler.transform(window.reshape(-1, c))
        return np.ascontiguousarray(scaled.reshape(b, t, c), dtype=np.float32)

    def _sigmoid_score(self, mse: float) -> float:
        """
        Map a raw MSE value to a probability in (0, 1).

        The sigmoid is centred on ``mse_threshold`` so that:
          score(threshold) = 0.5  (the decision boundary)
          score(2×threshold) ≈ 0.95
          score(0) ≈ 0.05
        """
        scale: float = self._mse_threshold / _SIGMOID_SCALE_FACTOR
        x: float = (mse - self._mse_threshold) / max(scale, 1e-10)
        # Numerically stable sigmoid.
        if x >= 0:
            return float(1.0 / (1.0 + np.exp(-x)))
        exp_x = float(np.exp(x))
        return float(exp_x / (1.0 + exp_x))
