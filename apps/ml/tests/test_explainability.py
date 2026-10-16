"""
Unit tests for src/explainability.py — RF-32.

Guarantees verified
-------------------
1.  Every one of the 80 engineered features maps to at least one physical sensor.
2.  DV_pressure features are never attributed to another sensor (prefix clash).
3.  Sensor aggregation preserves the total contribution (SHAP additivity).
4.  TreeSHAP local explanation: base + Σ contributions ≈ model margin output.
5.  On a synthetic task driven by one sensor, that sensor is ranked first and
    deletion of top-k features beats random deletion (faithfulness).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.explainability import (
    RAW_SENSORS,
    TreeAlertExplainer,
    aggregate_by_sensor,
    deletion_faithfulness,
    feature_to_sensors,
)
from src.preprocessing import MetroPTPreprocessor


@pytest.fixture(scope="module")
def engineered() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    raw = pd.DataFrame({s: rng.normal(5, 1, 400) for s in RAW_SENSORS})
    return MetroPTPreprocessor().fit_transform(raw)


def test_every_feature_maps_to_a_sensor(engineered: pd.DataFrame) -> None:
    assert engineered.shape[1] == 80
    for feat in engineered.columns:
        assert set(feature_to_sensors(feat)) <= set(RAW_SENSORS)


def test_prefix_clash_resolved() -> None:
    assert feature_to_sensors("DV_pressure_ma_5") == ("DV_pressure",)
    assert feature_to_sensors("DV_eletric") == ("DV_eletric",)
    assert feature_to_sensors("TP2_TP3_diff") == ("TP2", "TP3")
    with pytest.raises(ValueError):
        feature_to_sensors("unknown_feature")


def test_aggregation_preserves_total(engineered: pd.DataFrame) -> None:
    contrib = np.random.default_rng(1).normal(size=engineered.shape[1])
    agg = aggregate_by_sensor(contrib, list(engineered.columns))
    assert sum(agg.values()) == pytest.approx(contrib.sum())


def test_tree_shap_driven_by_one_sensor(engineered: pd.DataFrame) -> None:
    X = engineered.iloc[20:].reset_index(drop=True)
    y = (X["Oil_temperature"] > 5.8).astype(int)
    model = XGBClassifier(n_estimators=50, max_depth=3, random_state=0).fit(X, y)

    explainer = TreeAlertExplainer(model)
    values, base = explainer.shap_values(X)
    margin = model.predict(X, output_margin=True)
    assert np.allclose(base + values.sum(axis=1), margin, atol=1e-3)

    positives = X[y == 1].head(20)
    exps = explainer.explain(positives, model.predict_proba(positives)[:, 1], k=3)
    assert all(e.top_sensors[0].name == "Oil_temperature" for e in exps)
    assert exps[0].to_prompt_lines()[0].startswith("- Oil_temperature: aumenta")

    pos_values, _ = explainer.shap_values(positives)
    faith = deletion_faithfulness(
        lambda d: model.predict_proba(d)[:, 1],
        positives,
        pos_values,
        X[y == 0].median(),
        ks=(1, 3),
        n_random=5,
    )
    assert faith["explained_drop"][1] > faith["random_drop"][1]
