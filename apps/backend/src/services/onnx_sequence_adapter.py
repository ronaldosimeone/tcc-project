"""
OnnxSequenceAdapter — backend adapter for sequential ONNX models
(TCN, BiLSTM, PatchTST).

Interface contract
------------------
Exposes the same ``predict / predict_proba / feature_names_in_`` interface as
``OnnxMlpAdapter`` so ``ModelService`` swaps any model in without code changes.

Sliding-buffer strategy
-----------------------
Sequential models consume a window of T timesteps rather than a single row.
The adapter maintains a ``collections.deque`` of length ``window_size`` per
inference call.  ``ModelService.predict_from_features`` passes a DataFrame with
at least the last ``window_size`` rows from ``SensorBuffer``; the adapter
takes the last T rows and passes them as a single ``(1, T, C)`` window.

Cold-start padding
------------------
When fewer than ``window_size`` rows are available (buffer not yet warm), the
adapter pads the left side by replicating the oldest available row.  This is
the same "neutral replication" strategy used by ``ModelService._build_feature_row``
for the MLP — the prediction degrades gracefully until the buffer is warm.

ONNX optimisation options
--------------------------
Same session-level optimisations as ``OnnxMlpAdapter``:
* ``ORT_ENABLE_ALL`` graph optimisation (constant folding, op fusion).
* ``ORT_SEQUENTIAL`` execution mode + 1 intra-op thread for lowest p95
  latency on CPU-only single-row inference.
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

# Canonical 12 raw channels — must stay in sync with DataModule._RAW_CHANNELS.
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


class OnnxSequenceAdapter:
    """
    Production adapter wrapping a sequential ONNX model + per-channel scaler.

    Parameters
    ----------
    onnx_path :
        Path to ``<arch>_v1.onnx``.
    scaler_path :
        Path to ``<arch>_scaler.joblib`` (per-channel ``StandardScaler``).
    window_size :
        Number of time steps the model expects (T axis).  Read from the
        ``inference.window_size`` field of the model card by
        ``load_model_by_name``; defaults to 60 (the training default).
    channel_names :
        Ordered list of input channel names.  Read from ``feature_names`` in
        the model card so it survives feature-set changes without a code edit.
    """

    # Exposes the channel names so ``ModelService.__init__`` can build
    # ``_expected_features`` the same way it does for every other adapter.
    feature_names_in_: np.ndarray

    def __init__(
        self,
        onnx_path: Path,
        scaler_path: Path,
        window_size: int = 60,
        channel_names: list[str] | None = None,
    ) -> None:
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"Sequential ONNX artefact not found: {onnx_path}. "
                "Run `python src/train_sequential.py --arch <arch>` first."
            )
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Sequential scaler artefact not found: {scaler_path}. "
                "Run `python src/train_sequential.py --arch <arch>` first."
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
        self._window_size: int = window_size
        self._channels: list[str] = (
            list(channel_names) if channel_names is not None else list(_RAW_CHANNELS)
        )
        self.feature_names_in_: np.ndarray = np.array(self._channels, dtype=object)

        logger.info(
            "[RNF-24] OnnxSequenceAdapter loaded | onnx=%s | T=%d | C=%d",
            onnx_path.name,
            window_size,
            len(self._channels),
        )

    # ------------------------------------------------------------------
    # sklearn-compatible inference interface
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute class probabilities for a feature matrix.

        Parameters
        ----------
        X : pd.DataFrame
            Can be:
            * A single-row snapshot — padded to ``window_size`` via cold-start
              replication (used by ``ModelService.predict`` when the buffer
              is empty).
            * A multi-row DataFrame with at least ``window_size`` rows — the
              last ``window_size`` rows are consumed as the inference window
              (used by ``ModelService.predict_from_features`` once the sensor
              buffer is warm).

        Returns
        -------
        np.ndarray of shape (1, 2)
            ``[:, 1]`` is the fault probability consumed by ``ModelService``.
        """
        window: np.ndarray = self._build_window(X)  # (1, T, C) float32
        scaled: np.ndarray = self._scale(window)  # (1, T, C) float32 scaled
        logits: np.ndarray = self._session.run(None, {self._input_name: scaled})[0]
        return self._softmax(logits)

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
        """
        Convert a DataFrame into a ``(1, T, C)`` window tensor.

        Selects only the model's channel columns in training order; pads the
        left side with the oldest available row when fewer than ``window_size``
        rows are provided (cold-start replication).
        """
        # Keep only the contracted channels, in order.
        cols_present: list[str] = [c for c in self._channels if c in X.columns]
        arr: np.ndarray = X[cols_present].to_numpy(dtype=np.float32)

        t: int = self._window_size
        if len(arr) >= t:
            # Warm buffer: take the most recent T rows.
            window: np.ndarray = arr[-t:]
        else:
            # Cold start: pad left by repeating the oldest row.
            n_pad: int = t - len(arr)
            pad: np.ndarray = np.repeat(arr[:1], n_pad, axis=0)
            window = np.concatenate([pad, arr], axis=0)

        return window[np.newaxis].astype(np.float32)  # (1, T, C)

    def _scale(self, window: np.ndarray) -> np.ndarray:
        """Apply per-channel StandardScaler and return a C-contiguous float32 array."""
        b, t, c = window.shape
        scaled: np.ndarray = self._scaler.transform(window.reshape(-1, c))
        return np.ascontiguousarray(scaled.reshape(b, t, c), dtype=np.float32)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax along axis 1."""
        shifted: np.ndarray = logits - logits.max(axis=1, keepdims=True)
        exp: np.ndarray = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)
