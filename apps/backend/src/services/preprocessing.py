"""
Sensor feature engineering — *backend-side mirror* of ``apps/ml/src/preprocessing.py``.

Why this duplication exists
---------------------------
The training script lives in ``apps/ml`` and is packaged into its own Docker
image; the API runtime mounts only ``apps/ml/models`` (read-only) and does
**not** ship the ML source.  Importing ``apps/ml/src/preprocessing`` from the
backend works in local dev (shared filesystem) but breaks in production.

The two copies must therefore be kept *byte-for-byte equivalent* for the
feature contract to hold.  When you change the training preprocessor, mirror
the change here and re-train so the column order in the model card matches.

Used by ``InferencePipelineService`` to compute the same rolling / lag /
cross-sensor features at inference time as were seen at training time, once
the ``SensorBuffer`` is warm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

_DEFAULT_SENSOR_COLS: list[str] = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
]

_PRESSURE_COL: str = "TP2"
_RATIO_EPS: float = 1e-6


class MetroPTPreprocessor(BaseEstimator, TransformerMixin):
    """Mirror of ``apps/ml/src/preprocessing.MetroPTPreprocessor`` — see that
    module for the full docstring and design rationale."""

    def __init__(
        self,
        sensor_cols: list[str] | None = None,
        pressure_col: str = _PRESSURE_COL,
        window_std: int = 5,
        window_ma_short: int = 5,
        window_ma_long: int = 15,
        enable_v2_features: bool = True,
        lag_short: int = 5,
        lag_long: int = 15,
        window_minmax: int = 15,
    ) -> None:
        self.sensor_cols = sensor_cols
        self.pressure_col = pressure_col
        self.window_std = window_std
        self.window_ma_short = window_ma_short
        self.window_ma_long = window_ma_long
        self.enable_v2_features = enable_v2_features
        self.lag_short = lag_short
        self.lag_long = lag_long
        self.window_minmax = window_minmax

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "MetroPTPreprocessor":
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"MetroPTPreprocessor.transform expects a pandas DataFrame, "
                f"got {type(X).__name__!r} instead."
            )

        df: pd.DataFrame = X.copy()
        df = self._impute_nulls(df)
        df = self._add_pressure_delta(df)
        df = self._add_rolling_std(df)
        df = self._add_moving_averages(df)

        if self.enable_v2_features:
            df = self._add_cross_sensor_features(df)
            df = self._add_lags(df)
            df = self._add_rolling_minmax(df)

        return df

    # ------------------------------------------------------------------
    # V1 helpers
    # ------------------------------------------------------------------

    def _impute_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols: list[str] = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        return df

    def _add_pressure_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pressure_col not in df.columns:
            return df
        df[f"{self.pressure_col}_delta"] = df[self.pressure_col].diff().fillna(0.0)
        return df

    def _resolve_sensor_cols(self, df: pd.DataFrame) -> list[str]:
        candidates: list[str] = (
            self.sensor_cols if self.sensor_cols is not None else _DEFAULT_SENSOR_COLS
        )
        return [c for c in candidates if c in df.columns]

    def _add_rolling_std(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self._resolve_sensor_cols(df):
            df[f"{col}_std_{self.window_std}"] = (
                df[col]
                .rolling(window=self.window_std, min_periods=1)
                .std()
                .fillna(0.0)
            )
        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self._resolve_sensor_cols(df):
            df[f"{col}_ma_{self.window_ma_short}"] = (
                df[col].rolling(window=self.window_ma_short, min_periods=1).mean()
            )
            df[f"{col}_ma_{self.window_ma_long}"] = (
                df[col].rolling(window=self.window_ma_long, min_periods=1).mean()
            )
        return df

    # ------------------------------------------------------------------
    # V2 helpers
    # ------------------------------------------------------------------

    def _add_cross_sensor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if {"TP2", "TP3"}.issubset(df.columns):
            df["TP2_TP3_diff"] = df["TP2"] - df["TP3"]
            df["TP2_TP3_ratio"] = df["TP2"] / (df["TP3"] + _RATIO_EPS)
        if {"Motor_current", "TP2"}.issubset(df.columns):
            df["work_per_pressure"] = df["Motor_current"] / (df["TP2"] + _RATIO_EPS)
        if {"Reservoirs", "TP3"}.issubset(df.columns):
            df["reservoir_drop"] = df["Reservoirs"] - df["TP3"]
        return df

    def _add_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self._resolve_sensor_cols(df):
            lag_short_col = f"{col}_lag_{self.lag_short}"
            lag_long_col = f"{col}_lag_{self.lag_long}"
            df[lag_short_col] = df[col].shift(self.lag_short).bfill()
            df[lag_long_col] = df[col].shift(self.lag_long).bfill()
            df[f"{col}_roc_{self.lag_long}"] = (
                (df[col] - df[lag_long_col]) / float(self.lag_long)
            ).fillna(0.0)
        return df

    def _add_rolling_minmax(self, df: pd.DataFrame) -> pd.DataFrame:
        w = self.window_minmax
        for col in self._resolve_sensor_cols(df):
            min_col = f"{col}_min_{w}"
            max_col = f"{col}_max_{w}"
            df[min_col] = df[col].rolling(window=w, min_periods=1).min()
            df[max_col] = df[col].rolling(window=w, min_periods=1).max()
            df[f"{col}_range_{w}"] = df[max_col] - df[min_col]
        return df
