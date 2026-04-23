"""
SensorSimulator — RF-13 compliant multi-mode data generator.

Produces realistic air-compressor sensor readings that mimic the statistical
behaviour of the MetroPT-3 SCADA dataset using numpy distributions.

Modes
-----
NORMAL      Stationary Gaussian process around nominal operating values.
DEGRADATION Progressive drift driven by a step counter; each reading shifts
            the distribution toward failure territory over ~300 steps.
FAILURE     High-variance, out-of-range values that simulate active breakdown.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

import numpy as np
import structlog

from src.schemas.stream import SensorReading

log = structlog.get_logger(__name__)

_DEGRADATION_HORIZON: int = 300  # steps to reach full degradation (drift = 1.0)


class SimulatorMode(str, Enum):
    NORMAL = "NORMAL"
    DEGRADATION = "DEGRADATION"
    FAILURE = "FAILURE"


class SensorSimulator:
    """
    Stateful, mode-driven sensor-data generator.

    The mode can be switched at any time (RNF-29); the step counter resets on
    every mode change so each transition starts from the beginning of its
    distribution profile.

    Coroutine-safety: all access happens on the single asyncio event loop,
    so no locking is required.
    """

    def __init__(self, mode: SimulatorMode = SimulatorMode.NORMAL) -> None:
        self._mode: SimulatorMode = mode
        self._step: int = 0
        self._rng: np.random.Generator = np.random.default_rng()

    # ── Mode control (RNF-29) ─────────────────────────────────────────────

    @property
    def mode(self) -> SimulatorMode:
        return self._mode

    @mode.setter
    def mode(self, new_mode: SimulatorMode) -> None:
        if new_mode != self._mode:
            log.info(
                "simulator_mode_changed", from_=self._mode.value, to=new_mode.value
            )
            self._mode = new_mode
            self._step = 0

    # ── Public API ────────────────────────────────────────────────────────

    def generate_reading(self) -> SensorReading:
        """Return one SensorReading from the current mode's distribution."""
        self._step += 1
        ts = datetime.now(tz=timezone.utc)

        if self._mode == SimulatorMode.NORMAL:
            return self._normal(ts)
        if self._mode == SimulatorMode.DEGRADATION:
            return self._degradation(ts)
        return self._failure(ts)

    # ── Private helpers ───────────────────────────────────────────────────

    def _binary(self, p_off: float) -> float:
        """Return 0.0 or 1.0 with the given probability of being off."""
        p_off = float(np.clip(p_off, 0.0, 1.0))
        return float(self._rng.choice([0.0, 1.0], p=[p_off, 1.0 - p_off]))

    def _lerp(self, a: float, b: float, t: float) -> float:
        """Linear interpolation from a to b by factor t ∈ [0, 1]."""
        return a + (b - a) * t

    # ── Mode generators ───────────────────────────────────────────────────

    def _normal(self, ts: datetime) -> SensorReading:
        """Stable operation — low-noise Gaussians around nominal values."""
        g = self._rng
        return SensorReading(
            timestamp=ts,
            TP2=round(float(g.normal(5.90, 0.15)), 4),
            TP3=round(float(g.normal(3.40, 0.10)), 4),
            H1=round(float(g.normal(46.0, 2.00)), 4),
            DV_pressure=round(float(g.normal(0.05, 0.02)), 4),
            Reservoirs=round(float(g.normal(7.05, 0.12)), 4),
            Motor_current=round(float(g.normal(2.80, 0.15)), 4),
            Oil_temperature=round(float(g.normal(57.0, 1.50)), 4),
            COMP=self._binary(0.25),
            DV_eletric=self._binary(0.45),
            Towers=self._binary(0.35),
            MPG=self._binary(0.15),
            Oil_level=self._binary(0.05),
        )

    def _degradation(self, ts: datetime) -> SensorReading:
        """
        Progressive drift toward failure.

        drift ∈ [0, 1] grows linearly with _step up to _DEGRADATION_HORIZON.
        All distribution parameters interpolate between NORMAL and FAILURE values.
        """
        g = self._rng
        drift = min(self._step / _DEGRADATION_HORIZON, 1.0)
        lp = self._lerp

        return SensorReading(
            timestamp=ts,
            # Downstream pressure drops; noise widens
            TP2=round(float(g.normal(lp(5.90, 3.50, drift), lp(0.15, 1.50, drift))), 4),
            # Upstream pressure rises as compressor works harder
            TP3=round(float(g.normal(lp(3.40, 5.20, drift), lp(0.10, 0.80, drift))), 4),
            # Return pressure becomes erratic
            H1=round(float(g.normal(lp(46.0, 72.0, drift), lp(2.00, 15.0, drift))), 4),
            # Differential pressure worsens
            DV_pressure=round(
                float(g.normal(lp(0.05, 0.65, drift), lp(0.02, 0.30, drift))), 4
            ),
            # Reservoir drains faster
            Reservoirs=round(
                float(g.normal(lp(7.05, 5.50, drift), lp(0.12, 1.00, drift))), 4
            ),
            # Motor overworks
            Motor_current=round(
                float(g.normal(lp(2.80, 6.80, drift), lp(0.15, 1.80, drift))), 4
            ),
            # Oil heats up toward critical
            Oil_temperature=round(
                float(g.normal(lp(57.0, 86.0, drift), lp(1.50, 8.00, drift))), 4
            ),
            # Digital sensors begin flickering proportionally to drift
            COMP=self._binary(lp(0.25, 0.50, drift)),
            DV_eletric=self._binary(lp(0.45, 0.50, drift)),
            Towers=self._binary(lp(0.35, 0.50, drift)),
            MPG=self._binary(lp(0.15, 0.50, drift)),
            Oil_level=self._binary(lp(0.05, 0.40, drift)),
        )

    def _failure(self, ts: datetime) -> SensorReading:
        """Active breakdown — extreme values, high variance, erratic digital sensors."""
        g = self._rng
        return SensorReading(
            timestamp=ts,
            TP2=round(float(g.normal(3.50, 1.50)), 4),
            TP3=round(float(g.normal(5.20, 0.80)), 4),
            H1=round(float(g.normal(72.0, 15.0)), 4),
            DV_pressure=round(float(g.normal(0.65, 0.30)), 4),
            Reservoirs=round(float(g.normal(5.50, 1.00)), 4),
            Motor_current=round(float(g.normal(6.80, 1.80)), 4),
            Oil_temperature=round(float(g.normal(86.0, 8.00)), 4),
            COMP=self._binary(0.50),
            DV_eletric=self._binary(0.50),
            Towers=self._binary(0.50),
            MPG=self._binary(0.50),
            Oil_level=self._binary(0.40),
        )


# Module-level singleton — consumed by SensorStreamService and the simulator router.
_simulator = SensorSimulator()


def get_simulator() -> SensorSimulator:
    """FastAPI dependency returning the process-wide SensorSimulator singleton."""
    return _simulator
