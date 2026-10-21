"""
Leakage-free retrains served through the card-driven loader.

For every served ``*_v3`` artefact present on disk:
1. its card declares leakage-free validation and a ``serving`` block;
2. ``load_model_by_name`` builds the right adapter from the card alone;
3. a prediction through the REAL pipeline path (60-row buffer → V2 feature
   engineering → ``_infer_with_history``) returns a probability in [0, 1];
4. windowed models consume the whole buffer (no constant padded window).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.core.config import settings
from src.services.inference_pipeline import _infer_with_history
from src.services.model_service import load_model_by_name
from src.services.preprocessing import MetroPTPreprocessor

_NAMES = (
    "random_forest_v3",
    "xgboost_v3",
    "mlp_v3",
    "tcn_v3",
    "bilstm_v3",
    "patchtst_v3",
    "autoencoder_v3",
)
_MODELS = settings.models_dir
_AVAILABLE = [n for n in _NAMES if (_MODELS / f"{n}_card.json").exists()]

_SENSORS = (
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
)


def _buffer(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(5, 1, (n, len(_SENSORS)))
    df = pd.DataFrame(data, columns=list(_SENSORS))
    for c in ("COMP", "DV_eletric", "Towers", "MPG", "Oil_level"):
        df[c] = (df[c] > 5).astype(float)
    return df


@pytest.mark.parametrize("name", _AVAILABLE)
def test_card_declares_leakage_free_serving(name: str) -> None:
    card = json.loads((_MODELS / f"{name}_card.json").read_text(encoding="utf-8"))
    assert card["validation"]["leakage_free"] is True
    assert card["serving"]["adapter"] in {"tree", "mlp", "sequence", "autoencoder"}
    assert (_MODELS / card["serving"]["onnx"]).exists()
    assert not any(
        f.startswith(("Reservoirs", "Oil_level")) for f in card["feature_names"]
    )


@pytest.mark.parametrize("name", _AVAILABLE)
def test_predicts_through_the_pipeline_path(name: str) -> None:
    service = load_model_by_name(name)
    out = _infer_with_history(service, _buffer(), MetroPTPreprocessor())
    assert 0.0 <= out.failure_probability <= 1.0


@pytest.mark.parametrize(
    "name",
    [
        n
        for n in _AVAILABLE
        if n.split("_v")[0] in {"tcn", "bilstm", "patchtst", "autoencoder"}
    ],
)
def test_windowed_models_receive_the_real_history(name: str) -> None:
    """The window handed to ONNX must be the buffer itself, not the last row
    repeated 60 times (the serving bug fixed in this round). Probabilities are
    not compared: on synthetic inputs the TCN saturates at 0/1 even though its
    logits change with the history."""
    service = load_model_by_name(name)
    seen: list[pd.DataFrame] = []
    adapter = service._model  # noqa: SLF001
    original = adapter._build_window  # noqa: SLF001

    def spy(X: pd.DataFrame) -> np.ndarray:
        seen.append(X)
        return original(X)

    adapter._build_window = spy  # noqa: SLF001
    _infer_with_history(service, _buffer(seed=1), MetroPTPreprocessor())
    window = original(seen[-1])[0]
    assert len(seen[-1]) == 60
    assert np.ptp(window, axis=0).max() > 0  # rows differ over time
