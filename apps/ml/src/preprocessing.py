"""
Feature Engineering pipeline for the MetroPT-3 Air Compressor dataset.

This module exposes a single public class, `MetroPTPreprocessor`, which is a
fully scikit-learn-compatible transformer (inherits BaseEstimator +
TransformerMixin).  It can be dropped into any sklearn Pipeline without
modification.

Domain context
--------------
The MetroPT-3 dataset records high-frequency readings from an air compressor
onboard a Porto metro train.  Raw sensor values alone are insufficient for
anomaly/fault detection because:

  • Compressor faults manifest as *changes in signal dynamics*, not just
    threshold violations on instantaneous readings.
  • Pressure cycles repeat at ~1 Hz; a window of 5 cycles captures one full
    load/unload phase, while 15 cycles cover the warm-up transient.

The features computed here are the minimum viable set required before any
classical or deep learning classifier can be trained (RF-02).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# Sensor columns present in the MetroPT-3 schema (excludes timestamp and
# binary indicator columns that should not be smoothed).
_DEFAULT_SENSOR_COLS: list[str] = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
]

# Column used to compute the pressure delta feature.
_PRESSURE_COL: str = "TP2"


class MetroPTPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn-compatible transformer that engineers time-series features
    from raw MetroPT-3 sensor readings.

    Parameters
    ----------
    sensor_cols : list[str] | None
        Sensor column names on which rolling statistics are computed.
        Defaults to the seven analogue sensor columns in the dataset schema.
        Pass a custom list to restrict or extend the feature set.
    pressure_col : str
        Column used to compute the instantaneous pressure delta.
        Defaults to ``"TP2"`` (delivery pressure after the compressor head).
    window_std : int
        Window size (in samples) for the rolling standard deviation feature.
        Default: 5.
    window_ma_short : int
        Window size for the short moving average. Default: 5.
    window_ma_long : int
        Window size for the long moving average. Default: 15.

    Notes
    -----
    *fit* does nothing (no parameters are estimated from training data) but is
    kept for sklearn API compliance.  The class is stateless between calls to
    *transform*.
    """

    def __init__(
        self,
        sensor_cols: list[str] | None = None,
        pressure_col: str = _PRESSURE_COL,
        window_std: int = 5,
        window_ma_short: int = 5,
        window_ma_long: int = 15,
    ) -> None:
        self.sensor_cols = sensor_cols
        self.pressure_col = pressure_col
        self.window_std = window_std
        self.window_ma_short = window_ma_short
        self.window_ma_long = window_ma_long

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "MetroPTPreprocessor":
        """No-op fit kept for sklearn Pipeline compatibility."""
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """
        Apply the full feature engineering pipeline.

        Steps (applied in order):
            1. Validate input type.
            2. Copy — never mutate the caller's DataFrame.
            3. Null imputation (ffill → bfill).
            4. Pressure delta.
            5. Rolling std (window_std samples).
            6. Rolling moving averages (window_ma_short, window_ma_long samples).

        Parameters
        ----------
        X : pd.DataFrame
            Raw sensor DataFrame.  Must contain at least the columns listed in
            *sensor_cols* and *pressure_col*.

        Returns
        -------
        pd.DataFrame
            New DataFrame with all original columns plus the engineered features.

        Raises
        ------
        TypeError
            If *X* is not a ``pd.DataFrame``.
        """
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

        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _impute_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward-fill then backward-fill numeric sensor columns.

        Forward-fill propagates the last valid reading forward in time (safe
        for sensor drop-outs lasting a few samples).  The subsequent
        backward-fill handles any remaining NaNs at the very start of the
        series where no prior value exists.
        """
        numeric_cols: list[str] = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        return df

    def _add_pressure_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the first-order difference of the pressure column.

        Δp(t) = p(t) − p(t−1)

        A sudden spike or drop in Δp is one of the earliest indicators of a
        valve failure or blockage in the air circuit.  The first row becomes
        NaN after diff(); we fill it with 0.0 (no change at series start).
        """
        if self.pressure_col not in df.columns:
            return df

        col_name: str = f"{self.pressure_col}_delta"
        df[col_name] = df[self.pressure_col].diff().fillna(0.0)
        return df

    def _resolve_sensor_cols(self, df: pd.DataFrame) -> list[str]:
        """Return the effective sensor column list, filtered to those present in *df*."""
        candidates: list[str] = (
            self.sensor_cols if self.sensor_cols is not None else _DEFAULT_SENSOR_COLS
        )
        return [c for c in candidates if c in df.columns]

    def _add_rolling_std(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling standard deviation over *window_std* samples per sensor column.

        std(t) = σ( x[t − w + 1 : t + 1] )

        A rising rolling std on pressure or motor current signals increasing
        mechanical instability (e.g., worn piston rings or bearing damage).
        The ``min_periods=1`` parameter avoids NaN for the first w−1 rows.
        """
        cols: list[str] = self._resolve_sensor_cols(df)
        for col in cols:
            df[f"{col}_std_{self.window_std}"] = (
                df[col].rolling(window=self.window_std, min_periods=1).std().fillna(0.0)
            )
        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Short and long exponential-free moving averages per sensor column.

        MA_k(t) = mean( x[t − k + 1 : t + 1] )

        The cross-over between MA_5 and MA_15 is a lightweight, interpretable
        trend signal used by the diagnostic layer to flag gradual drift
        (e.g., slow oil temperature rise before thermal shutdown).
        ``min_periods=1`` avoids leading NaNs.
        """
        cols: list[str] = self._resolve_sensor_cols(df)
        for col in cols:
            df[f"{col}_ma_{self.window_ma_short}"] = (
                df[col].rolling(window=self.window_ma_short, min_periods=1).mean()
            )
            df[f"{col}_ma_{self.window_ma_long}"] = (
                df[col].rolling(window=self.window_ma_long, min_periods=1).mean()
            )
        return df
