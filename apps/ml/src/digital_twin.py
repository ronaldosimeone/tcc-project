"""
Grey-box digital twin of the APU air compressor — physics residuals.

The compressor alternates between LOADING (COMP = 0: intake valve open,
motor ~5.6 A, delivery pressure TP2 ~8.5 bar) and IDLE/offload (COMP = 1:
TP2 ≈ 0, motor ~1.4 A). Two first-principles relations hold in healthy
operation and break during an air leak:

1. Mass balance of the main reservoir, dP/dt = (Q_in − Q_out) / V.
   While loading the reservoir fills (≈ +0.096 bar / 10 s on MetroPT-3
   normal data, decreasing with back-pressure); while idle it decays slowly
   (train consumption ≈ −0.016 bar / 10 s). During the real failures the
   fill rate collapses to ≈ +0.0001 bar / 10 s: the compressor works but the
   air escapes. → ``fill_residual``, ``idle_decay_residual``.
2. Electrical work. Motor current while loading grows with the pressure the
   compressor pushes against (corr ≈ 0.83 with TP2) and oil viscosity
   (temperature). → ``current_residual``.
3. Duty cycle. A leaking system makes the compressor load more often to
   hold pressure. → ``duty_residual`` (share of loading samples in the last
   hour minus the healthy average).

Each expectation is a small linear model fitted ONLY on healthy training
rows (``fit``); ``transform`` returns the residuals (actual − expected),
smoothed over the last ``window`` samples of the relevant regime. Residuals
are interpretable in physical units, unlike the 80 statistical features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TWIN_FEATURES: tuple[str, ...] = (
    "fill_residual",
    "idle_decay_residual",
    "current_residual",
    "duty_residual",
)


def _lstsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def _predict(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    return coef[0] + X @ coef[1:]


@dataclass
class PhysicsTwin:
    window: int = 30
    duty_window: int = 360
    # Main-reservoir pressure used in the mass balance. On MetroPT-3
    # ``Reservoirs`` is a copy of ``TP3`` (r = 1.000); on MetroPT 2022
    # ``Reservoirs`` is a different ~1.6 bar signal, so cross-dataset work
    # uses ``TP3`` (identical on 2020, valid on 2022).
    pressure_col: str = "Reservoirs"
    fill_coef: np.ndarray = field(default_factory=lambda: np.zeros(2))
    current_coef: np.ndarray = field(default_factory=lambda: np.zeros(3))
    idle_decay: float = 0.0
    duty_mean: float = 0.0
    fitted: bool = False

    def _signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["loading"] = (df["COMP"] < 0.5).astype(float)
        out["dR"] = df[self.pressure_col].diff().fillna(0.0)
        return out

    def fit(self, df_normal: pd.DataFrame) -> "PhysicsTwin":
        """``df_normal`` = healthy, time-ordered training rows (raw sensors)."""
        sig = self._signals(df_normal)
        load = sig["loading"].to_numpy() == 1
        idle = ~load
        if load.sum() < 10 or idle.sum() < 10:
            raise ValueError("Need both loading and idle samples to fit the twin.")
        self.fill_coef = _lstsq(
            df_normal[self.pressure_col].to_numpy()[load, None],
            sig["dR"].to_numpy()[load],
        )
        self.current_coef = _lstsq(
            df_normal[["TP2", "Oil_temperature"]].to_numpy()[load],
            df_normal["Motor_current"].to_numpy()[load],
        )
        self.idle_decay = float(sig["dR"].to_numpy()[idle].mean())
        self.duty_mean = float(load.mean())
        self.fitted = True
        return self

    def _regime_mean(self, values: np.ndarray, mask: np.ndarray) -> pd.Series:
        """Rolling mean of ``values`` over the last ``window`` samples of the
        regime ``mask``; carried forward while the regime is inactive."""
        s = pd.Series(np.where(mask, values, np.nan))
        return s.rolling(self.window, min_periods=1).mean().ffill().fillna(0.0)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        sig = self._signals(df)
        load = sig["loading"].to_numpy() == 1
        dR = sig["dR"].to_numpy()

        fill_exp = _predict(self.fill_coef, df[self.pressure_col].to_numpy()[:, None])
        cur_exp = _predict(self.current_coef, df[["TP2", "Oil_temperature"]].to_numpy())

        out = pd.DataFrame(index=df.index)
        out["fill_residual"] = self._regime_mean(dR - fill_exp, load).to_numpy()
        out["idle_decay_residual"] = self._regime_mean(
            dR - self.idle_decay, ~load
        ).to_numpy()
        out["current_residual"] = self._regime_mean(
            df["Motor_current"].to_numpy() - cur_exp, load
        ).to_numpy()
        duty = (
            pd.Series(sig["loading"].to_numpy())
            .rolling(self.duty_window, min_periods=1)
            .mean()
        )
        out["duty_residual"] = (duty - self.duty_mean).to_numpy()
        return out
