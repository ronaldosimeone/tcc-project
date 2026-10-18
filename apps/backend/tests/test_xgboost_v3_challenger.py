"""
RF-33 challenger ``xgboost_v3`` (MetroPT-3 + MetroPT 2022, harmonised).

1. Loads through the same path as every tree model (OnnxTreeAdapter + card).
2. Its card excludes the sensors whose meaning differs across datasets.
3. The live pipeline's 80 V2 features are aligned by NAME to its 68 inputs —
   no extra pipeline needed to serve it as challenger.
4. The card records the nested selection and its out-of-time evidence.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.core.config import settings
from src.services.model_service import load_model_by_name
from src.services.preprocessing import MetroPTPreprocessor

_CARD = settings.model_path.parent / "xgboost_v3_card.json"

pytestmark = pytest.mark.skipif(
    not settings.xgboost_v3_onnx_path.exists() or not _CARD.exists(),
    reason="xgboost_v3 not built (python -m src.train_production --models xgboost)",
)

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


def _v2_window(n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame(rng.normal(5, 1, (n, len(_SENSORS))), columns=list(_SENSORS))
    return MetroPTPreprocessor().fit_transform(raw)


def test_card_excludes_semantically_mismatched_sensors() -> None:
    card = json.loads(_CARD.read_text(encoding="utf-8"))
    names: list[str] = card["feature_names"]
    assert card["dropped_sensors"] == ["Reservoirs", "Oil_level"]
    assert not any(n.startswith(("Reservoirs", "Oil_level")) for n in names)
    assert "reservoir_drop" not in names
    assert card["feature_count"] == len(names)


def test_card_records_nested_selection_and_evidence() -> None:
    card = json.loads(_CARD.read_text(encoding="utf-8"))
    assert card["status"] == "challenger"
    assert card["validation"]["leakage_free"] is True
    assert card["hyperparameters"]  # chosen by the nested inner search
    assert card["calibration"] is None or card["calibration"]["method"] == "platt"
    assert "failures_detected_mean_of_6" in card["evidence"]


def test_serves_from_the_live_v2_feature_row() -> None:
    service = load_model_by_name("xgboost_v3")
    row = _v2_window().iloc[[-1]]  # all 80 V2 features, as the pipeline builds
    out = service.predict_from_features(row)
    assert 0.0 <= out.failure_probability <= 1.0
    assert service.decision_threshold == pytest.approx(
        json.loads(_CARD.read_text(encoding="utf-8"))["decision_threshold"]
    )
