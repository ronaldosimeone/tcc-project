"""
Streaming drift detectors — RF-34.

Production (RF-27) checks data drift once a day: PSI of the last 24 h vs a
reference sample, drift when PSI > 0.25. That is a batch test with a fixed
1-day granularity. This module adds two classical *sequential* detectors so
their detection delay and false-alarm rate can be compared under controlled,
synthetic drift (``evaluate_drift.py``):

* ``PageHinkley`` (Page, 1954) — cumulative deviation from the running mean;
  very fast on abrupt mean shifts, one direction per instance (two-sided
  wrapper below).
* ``Adwin`` (Bifet & Gavaldà, SDM 2007) — adaptive window: drops the oldest
  part of the window whenever two sub-windows have significantly different
  means (Hoeffding-style bound with confidence δ). Exact O(W) cut search,
  fine for hourly streams of a few thousand points.

Both take one value per step and return True on the step where drift is
signalled, then reset (Page-Hinkley) or shrink (ADWIN).
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


class PageHinkley:
    """One-sided Page-Hinkley for an INCREASE of the mean (use two instances,
    or ``TwoSidedPageHinkley``, for both directions)."""

    def __init__(
        self, delta: float = 0.005, threshold: float = 50.0, min_instances: int = 30
    ) -> None:
        self.delta = delta
        self.threshold = threshold
        self.min_instances = min_instances
        self.reset()

    def reset(self) -> None:
        self.n: int = 0
        self.mean: float = 0.0
        self.cum: float = 0.0
        self.min_cum: float = 0.0

    def update(self, x: float) -> bool:
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.cum += x - self.mean - self.delta
        self.min_cum = min(self.min_cum, self.cum)
        if self.n >= self.min_instances and self.cum - self.min_cum > self.threshold:
            self.reset()
            return True
        return False


class TwoSidedPageHinkley:
    def __init__(
        self, delta: float = 0.005, threshold: float = 50.0, min_instances: int = 30
    ) -> None:
        self._up = PageHinkley(delta, threshold, min_instances)
        self._down = PageHinkley(delta, threshold, min_instances)

    def update(self, x: float) -> bool:
        up = self._up.update(x)
        down = self._down.update(-x)
        if up or down:
            self._up.reset()
            self._down.reset()
            return True
        return False


class Adwin:
    """ADWIN with an exact cut search over a bounded window."""

    def __init__(
        self, delta: float = 0.002, max_window: int = 2000, min_side: int = 10
    ) -> None:
        self.delta = delta
        self.max_window = max_window
        self.min_side = min_side
        self.window: deque[float] = deque(maxlen=max_window)

    def _cut_found(self) -> int | None:
        w = np.fromiter(self.window, dtype=float)
        n = w.size
        if n < 2 * self.min_side:
            return None
        csum = np.cumsum(w)
        total = csum[-1]
        var = float(np.var(w)) + 1e-12
        delta_p = self.delta / math.log(n)  # Bonferroni over the candidate cuts
        for i in range(self.min_side, n - self.min_side + 1):
            n0, n1 = i, n - i
            mu0 = csum[i - 1] / n0
            mu1 = (total - csum[i - 1]) / n1
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            eps = math.sqrt(2.0 / m * var * math.log(2.0 / delta_p)) + 2.0 / (
                3.0 * m
            ) * math.log(2.0 / delta_p)
            if abs(mu0 - mu1) > eps:
                return i
        return None

    def update(self, x: float) -> bool:
        self.window.append(float(x))
        cut = self._cut_found()
        if cut is None:
            return False
        for _ in range(cut):  # drop the stale part of the window
            self.window.popleft()
        return True


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI with reference-quantile bins (same definition family as RF-27)."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / reference.size
    c = np.histogram(current, edges)[0] / max(current.size, 1)
    r = np.where(r == 0, 1e-4, r)
    c = np.where(c == 0, 1e-4, c)
    return float(np.sum((c - r) * np.log(c / r)))
