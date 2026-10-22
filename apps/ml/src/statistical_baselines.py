"""
Statistical process-control baselines — CUSUM and EWMA — as learners of the
nested protocol.

Question they answer: does the complexity of the ML/DL models pay off with
7 failures? The public MetroPT benchmark (mimouniahmed/metro-anomaly-benchmark)
reports 4/4 failures with a CUSUM on the motor current, but it chose its
detectors looking at the same 4 failures it tested. Here every free parameter
(sensor, drift ``k`` / smoothing ``lambda``, reference) is a candidate of the
nested search: selected on inner leave-one-failure-out OOF cost, threshold
``h`` via Platt + RN-02 on the same OOF, never on the outer test fold.

Pipeline per sensor (all causal, gap-aware):

1. Smoothed signal: trailing 10-min mean inside each continuous segment
   (a gap > 60 s starts a segment, as in the ML preprocessor). Raw 10-s
   readings follow the compressor on/off cycle; the 10-min mean tracks the
   duty cycle, which is what a leak changes.
2. Standardisation z = (m - mu) / sd with either
   * ``static``: mu, sd of the smoothed signal on the HEALTHY TRAINING rows
     (the only fitted quantity — training data only), or
   * ``rolling``: mu, sd of the smoothed signal over [t - 7 d, t - 1 h)
     (no fit at all; adapts to regime shifts such as 2020 → 2022).
3. Statistic, reset at every segment start:
   * CUSUM (two-sided): S+ = max(0, S+ + z - k), S- = max(0, S- - z - k),
     score = max(S+, S-);
   * EWMA: e = lambda z + (1 - lambda) e, score = |e|.
   Several sensors → max of the per-sensor scores (OR of the alarms).
4. Raw score = log1p(score) — monotone, so Platt calibration and the RN-02
   threshold of the nested protocol apply unchanged.

The search spaces below were fixed BEFORE running anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numba
import numpy as np
import pandas as pd

BASELINES: list[str] = ["cusum", "ewma"]

SMOOTH_S: float = 600.0
GAP_S: float = 60.0
REF_WINDOW_S: float = 7 * 86400.0
REF_LAG_S: float = 3600.0
# Floor of the reference sd, as a fraction of the sensor's healthy sd, so a
# flat week (compressor idle) does not turn noise into huge z-scores.
SD_FLOOR_FRAC: float = 0.05

# Sensors: the two of the public benchmark's recommended detector plus the
# pressure that an air leak lowers. "+" = OR of both sensors.
_SENSORS: list[str] = [
    "Motor_current",
    "Oil_temperature",
    "TP2",
    "Motor_current+Oil_temperature",
]
SPACES: dict[str, list[dict[str, Any]]] = {
    "cusum": [
        {"sensors": s, "k": k, "reference": ref}
        for s in _SENSORS
        for k in (0.5, 1.0)
        for ref in ("static", "rolling")
    ],
    "ewma": [
        {"sensors": s, "lam": lam, "reference": ref}
        for s in _SENSORS
        for lam in (0.01, 0.1)
        for ref in ("static", "rolling")
    ],
}
# A-priori default (fallback with < 2 training failures) = the public
# benchmark's choice: motor current + oil temperature, k = 0.5, rolling 7 d.
SPACES["cusum"].insert(
    0,
    SPACES["cusum"].pop(
        SPACES["cusum"].index(
            {
                "sensors": "Motor_current+Oil_temperature",
                "k": 0.5,
                "reference": "rolling",
            }
        )
    ),
)
SPACES["ewma"].insert(
    0,
    SPACES["ewma"].pop(
        SPACES["ewma"].index(
            {
                "sensors": "Motor_current+Oil_temperature",
                "lam": 0.1,
                "reference": "rolling",
            }
        )
    ),
)


# ---------------------------------------------------------------------------
# Numba kernels (whole timeline, sorted by time)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _segment_mean(
    t: np.ndarray, x: np.ndarray, seg: np.ndarray, w: float
) -> np.ndarray:
    """Trailing mean over (t - w, t] that never crosses a segment start."""
    n = x.shape[0]
    out = np.empty(n)
    lo = 0
    acc = 0.0
    for i in range(n):
        if i > 0 and seg[i] != seg[i - 1]:
            lo = i
            acc = 0.0
        acc += x[i]
        while t[i] - t[lo] >= w:
            acc -= x[lo]
            lo += 1
        out[i] = acc / (i - lo + 1)
    return out


@numba.njit(cache=True)
def _rolling_ref(
    t: np.ndarray, x: np.ndarray, w: float, lag: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, sd and count of x over [t - w, t - lag) — time-based, causal."""
    n = x.shape[0]
    mu = np.empty(n)
    sd = np.empty(n)
    cnt = np.zeros(n)
    lo = 0
    hi = 0
    s1 = 0.0
    s2 = 0.0
    for i in range(n):
        while hi < n and t[hi] < t[i] - lag:
            s1 += x[hi]
            s2 += x[hi] * x[hi]
            hi += 1
        while lo < hi and t[lo] < t[i] - w:
            s1 -= x[lo]
            s2 -= x[lo] * x[lo]
            lo += 1
        c = hi - lo
        cnt[i] = c
        if c > 1:
            m = s1 / c
            mu[i] = m
            sd[i] = np.sqrt(max(s2 / c - m * m, 0.0))
        else:
            mu[i] = np.nan
            sd[i] = np.nan
    return mu, sd, cnt


@numba.njit(cache=True)
def _cusum(z: np.ndarray, seg: np.ndarray, k: float) -> np.ndarray:
    n = z.shape[0]
    out = np.empty(n)
    hi = 0.0
    lo = 0.0
    for i in range(n):
        if i > 0 and seg[i] != seg[i - 1]:
            hi = 0.0
            lo = 0.0
        zi = z[i]
        if np.isnan(zi):
            zi = 0.0
        hi = max(0.0, hi + zi - k)
        lo = max(0.0, lo - zi - k)
        out[i] = max(hi, lo)
    return out


@numba.njit(cache=True)
def _ewma(z: np.ndarray, seg: np.ndarray, lam: float) -> np.ndarray:
    n = z.shape[0]
    out = np.empty(n)
    e = 0.0
    for i in range(n):
        if i > 0 and seg[i] != seg[i - 1]:
            e = 0.0
        zi = z[i]
        if np.isnan(zi):
            zi = 0.0
        e = lam * zi + (1.0 - lam) * e
        out[i] = abs(e)
    return out


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------


class _Series:
    """Per-sensor smoothed signal and rolling reference, computed once per
    timeline (they depend on the data only, never on labels)."""

    def __init__(self, X: pd.DataFrame, ts: pd.Series) -> None:
        t = pd.to_datetime(ts).to_numpy().astype("datetime64[ns]").astype(np.int64)
        self.t = t.astype(np.float64) / 1e9
        if np.any(np.diff(self.t) < 0):
            raise ValueError("timeline must be sorted by time")
        self.seg = np.r_[0, np.cumsum(np.diff(self.t) > GAP_S)].astype(np.int64)
        self.X = X
        self._m: dict[str, np.ndarray] = {}
        self._ref: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def smoothed(self, sensor: str) -> np.ndarray:
        if sensor not in self._m:
            x = self.X[sensor].to_numpy(np.float64)
            self._m[sensor] = _segment_mean(self.t, x, self.seg, SMOOTH_S)
        return self._m[sensor]

    def rolling(self, sensor: str) -> tuple[np.ndarray, np.ndarray]:
        if sensor not in self._ref:
            mu, sd, _ = _rolling_ref(
                self.t, self.smoothed(sensor), REF_WINDOW_S, REF_LAG_S
            )
            self._ref[sensor] = (mu, sd)
        return self._ref[sensor]


_CACHE: dict[int, _Series] = {}


def _series(X: pd.DataFrame, ts: pd.Series) -> _Series:
    key = id(X)
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = _Series(X, ts)
    return _CACHE[key]


@dataclass
class Detector:
    name: str
    params: dict[str, Any]
    # per sensor: healthy-training mean and sd of the smoothed signal
    healthy: dict[str, tuple[float, float]]


def _sensors(params: dict[str, Any]) -> list[str]:
    return str(params["sensors"]).split("+")


def fit(
    name: str,
    params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    ts: pd.Series,
    train_idx: np.ndarray,
) -> Detector:
    """Only fitted quantity: mean/sd of the smoothed signal on healthy
    training rows (used by the static reference and as the sd floor)."""
    s = _series(X, ts)
    healthy_rows = train_idx[y.to_numpy()[train_idx] == 0]
    stats = {}
    for sensor in _sensors(params):
        m = s.smoothed(sensor)[healthy_rows]
        stats[sensor] = (float(np.mean(m)), float(np.std(m)) or 1.0)
    return Detector(name=name, params=dict(params), healthy=stats)


def raw_score(
    det: Detector, X: pd.DataFrame, ts: pd.Series, rows: np.ndarray
) -> np.ndarray:
    s = _series(X, ts)
    total: np.ndarray | None = None
    for sensor in _sensors(det.params):
        m = s.smoothed(sensor)
        mu_h, sd_h = det.healthy[sensor]
        if det.params["reference"] == "static":
            z = (m - mu_h) / sd_h
        else:
            mu, sd = s.rolling(sensor)
            z = (m - mu) / np.maximum(sd, SD_FLOOR_FRAC * sd_h)
        if det.name == "cusum":
            stat = _cusum(z, s.seg, float(det.params["k"]))
        else:
            stat = _ewma(z, s.seg, float(det.params["lam"]))
        total = stat if total is None else np.maximum(total, stat)
    assert total is not None
    return np.log1p(total[rows])
