"""
Feature Engineering pipeline for the MetroPT-3 Air Compressor dataset.

This module exposes a single public class, ``MetroPTPreprocessor``, which is a
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

V2 features (added 2026-04-28)
------------------------------
* **Cross-sensor** — TP2/TP3 differential, work-per-pressure ratio,
  reservoir-drop.  Captures the discriminating signature of an Air Leak
  (pressure falls *while* motor current rises) which the per-sensor
  rolling features cannot represent on their own.
* **Lag features** — explicit ``*_lag_5`` and ``*_lag_15`` snapshots so
  classifiers can see the absolute trajectory, not just smoothed averages.
* **Rate-of-change** — ``*_roc_15`` = (now − 15s ago) / 15.  More robust to
  high-frequency noise than the single-sample ``TP2_delta``.
* **Rolling min/max/range** — ``*_min_15`` / ``*_max_15`` / ``*_range_15``.
  Detects peak excursions inside a window, which are invisible to the mean.

All V2 features are toggled by the ``enable_v2_features`` flag (default True)
so legacy training scripts that pin ``feature_count == 34`` keep working
during the migration.
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

# Numerical safety floor for ratio features to avoid div-by-zero.
_RATIO_EPS: float = 1e-6


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
    enable_v2_features : bool
        Toggle for the V2 feature additions (cross-sensor, lags, rolling
        min/max).  Default ``True``.  Set to ``False`` to reproduce the V1
        feature contract exactly (34 features).
    lag_short : int
        Short lag window in samples (V2). Default: 5.
    lag_long : int
        Long lag window in samples (V2). Default: 15.
    window_minmax : int
        Rolling window for min/max/range (V2). Default: 15.

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

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "MetroPTPreprocessor":
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
            7. (V2) Cross-sensor ratios and differentials.
            8. (V2) Explicit lag features and rate-of-change.
            9. (V2) Rolling min / max / range.

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

        if self.enable_v2_features:
            df = self._add_cross_sensor_features(df)
            df = self._add_lags(df)
            df = self._add_rolling_minmax(df)

        return df

    # ------------------------------------------------------------------
    # V1 — Private helpers
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
                df[col]
                .rolling(window=self.window_std, min_periods=1)
                .std()
                .fillna(0.0)
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

    # ------------------------------------------------------------------
    # V2 — Private helpers
    # ------------------------------------------------------------------

    def _add_cross_sensor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cross-sensor features that capture the joint behaviour of correlated
        sensors — specifically the discriminating signature of an Air Leak.

        * ``TP2_TP3_diff``      Direct pressure gap between the compressor head
          and the pneumatic panel.  A widening gap during normal compressor
          state indicates leakage between the two stages.
        * ``TP2_TP3_ratio``     Same idea expressed as a ratio.  Normalises
          out absolute pressure level and is more robust under load changes.
        * ``work_per_pressure`` Motor current divided by delivery pressure —
          rises sharply when the motor is doing more work to maintain a
          falling pressure (classic leak symptom).
        * ``reservoir_drop``    Reservoir minus pneumatic panel pressure.
          Goes negative just before reservoir collapse.
        """
        if {"TP2", "TP3"}.issubset(df.columns):
            df["TP2_TP3_diff"] = df["TP2"] - df["TP3"]
            df["TP2_TP3_ratio"] = df["TP2"] / (df["TP3"] + _RATIO_EPS)
        if {"Motor_current", "TP2"}.issubset(df.columns):
            df["work_per_pressure"] = df["Motor_current"] / (df["TP2"] + _RATIO_EPS)
        if {"Reservoirs", "TP3"}.issubset(df.columns):
            df["reservoir_drop"] = df["Reservoirs"] - df["TP3"]
        return df

    def _add_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Explicit lag snapshots and rate-of-change.

        x_lag_k(t) = x(t − k)
        x_roc_k(t) = ( x(t) − x(t − k) ) / k

        Lags let the classifier see the *absolute trajectory* rather than just
        the smoothed mean.  ROC is a denoised first derivative — more robust
        than the single-sample ``delta`` for anomalies that develop over
        seconds rather than instantaneously.

        ``shift().bfill()`` propagates the first valid value backwards so the
        leading rows do not produce NaNs.
        """
        cols: list[str] = self._resolve_sensor_cols(df)
        for col in cols:
            lag_short_col = f"{col}_lag_{self.lag_short}"
            lag_long_col = f"{col}_lag_{self.lag_long}"
            df[lag_short_col] = df[col].shift(self.lag_short).bfill()
            df[lag_long_col] = df[col].shift(self.lag_long).bfill()
            df[f"{col}_roc_{self.lag_long}"] = (
                (df[col] - df[lag_long_col]) / float(self.lag_long)
            ).fillna(0.0)
        return df

    def _add_rolling_minmax(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling minimum, maximum, and range over a configurable window.

        Captures peak excursions that the rolling mean smooths out — a brief
        pressure spike inside an otherwise stable window is an early
        indicator of valve flutter.
        """
        cols: list[str] = self._resolve_sensor_cols(df)
        w = self.window_minmax
        for col in cols:
            min_col = f"{col}_min_{w}"
            max_col = f"{col}_max_{w}"
            df[min_col] = df[col].rolling(window=w, min_periods=1).min()
            df[max_col] = df[col].rolling(window=w, min_periods=1).max()
            df[f"{col}_range_{w}"] = df[max_col] - df[min_col]
        return df
