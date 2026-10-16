"""
Per-alert explainability — RF-32 (and the XAI input of RF-35).

* TreeSHAP (Lundberg et al., 2020) for the tree models — exact Shapley values
  in polynomial time, so every alert can carry its own explanation.
* Feature -> sensor aggregation. The models see 80 engineered features
  (rolling means, lags, ranges, cross-sensor ratios); a maintenance
  technician — and the LLM in RF-35 — reasons in terms of the 12 physical
  sensors. Shapley values are additive, so summing the contributions of all
  features derived from a sensor gives that sensor's exact contribution.
  Cross-sensor features are split evenly between the sensors they combine.
* Deletion faithfulness (Samek et al., 2017): replace the top-k features by
  their background (normal-operation) median and measure how much the
  failure probability drops, versus replacing k random features. A faithful
  explanation must beat the random baseline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

RAW_SENSORS: tuple[str, ...] = (
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

# Engineered features that combine two physical sensors (see preprocessing.py).
_CROSS_FEATURES: dict[str, tuple[str, ...]] = {
    "TP2_TP3_diff": ("TP2", "TP3"),
    "TP2_TP3_ratio": ("TP2", "TP3"),
    "work_per_pressure": ("Motor_current", "TP2"),
    "reservoir_drop": ("Reservoirs", "TP3"),
}


def feature_to_sensors(feature: str) -> tuple[str, ...]:
    """Physical sensor(s) an engineered feature is derived from."""
    if feature in _CROSS_FEATURES:
        return _CROSS_FEATURES[feature]
    # Longest match first: "DV_pressure" must not be read as "DV_eletric"/"TP2".
    for sensor in sorted(RAW_SENSORS, key=len, reverse=True):
        if feature == sensor or feature.startswith(f"{sensor}_"):
            return (sensor,)
    raise ValueError(f"Cannot map feature {feature!r} to a sensor.")


def aggregate_by_sensor(
    contrib: np.ndarray, features: Sequence[str]
) -> dict[str, float]:
    """Sum per-feature contributions into per-sensor contributions (exact for SHAP)."""
    values: np.ndarray = np.asarray(contrib, dtype=float)
    if values.shape[-1] != len(features):
        raise ValueError("contrib and features length mismatch.")
    out: dict[str, float] = {s: 0.0 for s in RAW_SENSORS}
    for j, feat in enumerate(features):
        sensors = feature_to_sensors(feat)
        share: float = float(values[..., j].sum()) / len(sensors)
        for s in sensors:
            out[s] += share
    return out


@dataclass
class Contribution:
    name: str
    contribution: float
    value: float | None = None


@dataclass
class AlertExplanation:
    probability: float
    base_value: float
    top_features: list[Contribution]
    top_sensors: list[Contribution]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_prompt_lines(self) -> list[str]:
        """Compact, LLM-friendly rendering for RF-35 (sensor-level only)."""
        lines: list[str] = []
        for c in self.top_sensors:
            direction: str = "aumenta" if c.contribution > 0 else "reduz"
            lines.append(f"- {c.name}: {direction} o risco ({c.contribution:+.3f})")
        return lines


def _shap_matrix(explainer: Any, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """SHAP values for the positive class as an (n, n_features) matrix."""
    raw: Any = explainer.shap_values(X, check_additivity=False)
    base: Any = explainer.expected_value
    if isinstance(raw, list):  # older shap: one array per class
        raw, base = raw[1], base[1]
    arr: np.ndarray = np.asarray(raw)
    if arr.ndim == 3:  # (n, features, classes)
        arr = arr[:, :, 1]
        base = np.asarray(base).ravel()[-1]
    return arr, float(np.asarray(base).ravel()[-1])


class TreeAlertExplainer:
    """
    Exact TreeSHAP for the tree models.

    XGBoost: native ``pred_contribs=True`` (TreeSHAP built into the booster,
    margin/log-odds space) — ``shap`` <= 0.49 cannot parse the XGBoost 3.x
    ``base_score`` format, and the native path needs no extra dependency.
    scikit-learn forests: ``shap.TreeExplainer`` (probability space).
    """

    def __init__(self, model: Any, background: pd.DataFrame | None = None) -> None:
        self.features: list[str] = []
        self._model: Any = model
        self._native_xgb: bool = type(model).__name__ in {
            "XGBClassifier",
            "XGBRegressor",
        }
        self._explainer: Any = None
        if not self._native_xgb:
            import shap

            self._explainer = (
                shap.TreeExplainer(model)
                if background is None
                else shap.TreeExplainer(
                    model, data=background, feature_perturbation="interventional"
                )
            )

    def shap_values(self, X: pd.DataFrame) -> tuple[np.ndarray, float]:
        self.features = list(X.columns)
        if self._native_xgb:
            from xgboost import DMatrix

            contribs: np.ndarray = self._model.get_booster().predict(
                DMatrix(X), pred_contribs=True
            )
            return contribs[:, :-1], float(contribs[0, -1])  # last column = bias term
        return _shap_matrix(self._explainer, X)

    def explain(
        self, X: pd.DataFrame, prob: np.ndarray, k: int = 5
    ) -> list[AlertExplanation]:
        values, base = self.shap_values(X)
        out: list[AlertExplanation] = []
        for i in range(len(X)):
            row: np.ndarray = values[i]
            order: np.ndarray = np.argsort(-np.abs(row))[:k]
            sensors: dict[str, float] = aggregate_by_sensor(row, self.features)
            top_s = sorted(sensors.items(), key=lambda kv: -abs(kv[1]))[:k]
            out.append(
                AlertExplanation(
                    probability=float(prob[i]),
                    base_value=base,
                    top_features=[
                        Contribution(
                            self.features[j], float(row[j]), float(X.iloc[i, j])
                        )
                        for j in order
                    ],
                    top_sensors=[Contribution(s, float(v)) for s, v in top_s],
                )
            )
        return out


def deletion_faithfulness(
    predict: Any,
    X: pd.DataFrame,
    contrib: np.ndarray,
    background_median: pd.Series,
    ks: Sequence[int] = (1, 3, 5, 10, 20),
    n_random: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Mean drop in P(failure) after replacing the top-k features (by positive
    contribution) with the background median, versus k random features.
    ``predict`` maps a DataFrame to P(failure).
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    base_p: np.ndarray = predict(X)
    n_feat: int = X.shape[1]
    out: dict[str, Any] = {
        "k": list(ks),
        "explained_drop": [],
        "random_drop": [],
        "base_mean_p": float(base_p.mean()),
    }
    med: np.ndarray = background_median[X.columns].to_numpy()
    for k in ks:
        Xe: np.ndarray = X.to_numpy(copy=True)
        top: np.ndarray = np.argsort(-contrib, axis=1)[:, :k]
        rows: np.ndarray = np.arange(len(X))[:, None]
        Xe[rows, top] = med[top]
        drop_e: float = float(
            (base_p - predict(pd.DataFrame(Xe, columns=X.columns))).mean()
        )

        drops: list[float] = []
        for _ in range(n_random):
            Xr: np.ndarray = X.to_numpy(copy=True)
            rnd: np.ndarray = np.argsort(rng.random((len(X), n_feat)), axis=1)[:, :k]
            Xr[rows, rnd] = med[rnd]
            drops.append(
                float((base_p - predict(pd.DataFrame(Xr, columns=X.columns))).mean())
            )
        out["explained_drop"].append(drop_e)
        out["random_drop"].append(float(np.mean(drops)))
    return out
