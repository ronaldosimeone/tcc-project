"""
OnnxTreeAdapter — sklearn-compatible wrapper for tree-based ONNX models (RF-10).

Serves the V2 artefacts exported by ``train_random_forest.py`` (skl2onnx) and
``train_xgboost.py`` (onnxmltools).  Tree ensembles do not need input scaling,
so this adapter is intentionally simpler than ``OnnxMlpAdapter``: no scaler,
no softmax — the ONNX graph emits class probabilities directly.

Output handling
---------------
ONNX classifier graphs typically expose two outputs:

  ``label``         — predicted class (ndarray int64, shape (n,))
  ``probabilities`` — class probabilities, in one of two formats:
                       • ndarray of shape (n, 2)        (skl2onnx with zipmap=False)
                       • sequence of dict[int, float]   (onnxmltools default, ZipMap)

The adapter detects which format the session produced and normalises both into
an ``np.ndarray`` of shape (n, 2) so ``ModelService`` sees a uniform contract.

Feature contract
----------------
``feature_names_in_`` is supplied at construction time (read from the matching
model card by ``model_service.load_model_by_name``).  Tree ensembles are
trained on positional ndarrays — they care about column *order*, not names.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnxruntime as ort
import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)


class OnnxTreeAdapter:
    """
    Production adapter that wraps a tree-based ONNX model (RF V2, XGB V2)
    behind the sklearn predict / predict_proba interface.

    Parameters
    ----------
    onnx_path : Path
        Absolute path to the ``.onnx`` artefact (e.g. ``random_forest_v2.onnx``).
    feature_names : Sequence[str]
        Column order the model was trained on.  Read from the matching model
        card so this adapter stays version-agnostic (RF V2 has 80 features,
        XGB V2 has 34 — both work with the same class).
    """

    def __init__(self, onnx_path: Path, feature_names: Sequence[str]) -> None:
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX artefact not found at {onnx_path}. "
                "Re-run the training script to regenerate it."
            )
        if not feature_names:
            raise ValueError(
                "feature_names must be non-empty — pass the list from the model card."
            )

        # Same ORT tuning as OnnxMlpAdapter: single-row inference path,
        # graph optimisations on, no thread-pool churn.
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
        self._output_names: list[str] = [o.name for o in self._session.get_outputs()]

        # Public sklearn-like contract consumed by ModelService.
        self.feature_names_in_: np.ndarray = np.array(
            list(feature_names), dtype=object
        )

        logger.info(
            "[RF-10] OnnxTreeAdapter loaded | onnx=%s | features=%d | outputs=%s",
            onnx_path.name,
            len(feature_names),
            self._output_names,
        )

    # ------------------------------------------------------------------
    # Public sklearn-compatible interface
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute class probabilities.

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            Columns are [P(class=0), P(class=1)] in [0.0, 1.0], rows sum to 1.0.
        """
        arr = np.ascontiguousarray(X.to_numpy(dtype=np.float32))
        outputs: list[Any] = self._session.run(None, {self._input_name: arr})
        return self._extract_probabilities(outputs, n_rows=arr.shape[0])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary class label (0 = normal, 1 = fault) at threshold 0.5."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(np.int64)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_probabilities(outputs: Sequence[Any], n_rows: int) -> np.ndarray:
        """
        Locate the probability tensor in a session output list.

        Handles both shapes ONNX classifier graphs can emit:
          1. ``ndarray`` of shape (n, 2)         — skl2onnx with zipmap=False
          2. ``list[dict[int, float]]`` of len n — onnxmltools default ZipMap

        Strategy: prefer a 2-D float ndarray; otherwise convert a dict-sequence.
        """
        # Pass 1 — ndarray with the right shape (RF V2 case).
        for out in outputs:
            if (
                isinstance(out, np.ndarray)
                and out.ndim == 2
                and out.shape == (n_rows, 2)
                and np.issubdtype(out.dtype, np.floating)
            ):
                return out.astype(np.float32, copy=False)

        # Pass 2 — ZipMap sequence of dicts (XGB V2 default case).
        for out in outputs:
            if isinstance(out, list) and out and isinstance(out[0], dict):
                # Keys may be int or str depending on the converter.
                first = out[0]
                k0 = 0 if 0 in first else "0"
                k1 = 1 if 1 in first else "1"
                return np.array(
                    [[float(d[k0]), float(d[k1])] for d in out],
                    dtype=np.float32,
                )

        raise RuntimeError(
            f"Could not find a probability output among {len(outputs)} session "
            f"outputs (expected ndarray (n,2) or list[dict])."
        )
