"""
SensorSimulator — RF-13 compliant data streamer backed by real MetroPT-3 data.

Why real data instead of synthetic Gaussians
--------------------------------------------
The Conv1D Autoencoder was trained on the physical dynamics of the MetroPT-3
compressor (cyclic load/unload, pressure oscillations, correlated sensor
behaviour).  A flat np.random generator produces sequences that are
statistically unlike real operation and are therefore flagged as anomalies by
the autoencoder, producing constant false positives.

Streaming from the real dataset eliminates that distribution mismatch: the
model sees data drawn from the exact same manifold it was trained on.

Memory strategy
---------------
The parquet is read once at construction time.  Only the 12 sensor columns are
retained as a contiguous float32 ndarray.  Approximate footprint:
  12 columns × 1.5 M rows × 4 bytes ≈ 72 MB — well within FastAPI's budget.
No I/O occurs during streaming: `generate_reading` is a pure array lookup +
pointer increment.

Failure window detection
------------------------
Four known air-leak periods are isolated by timestamp range (provided by the
MetroPT-3 paper and domain-confirmed).  If the parquet timestamp column is
unavailable, the implementation falls back to the `anomaly` label column.
Everything outside the failure windows is treated as normal operation.

Modes
-----
NORMAL      Sequential read from df_normal rows — loops continuously.
FAILURE     Sequential read from df_failure rows — loops continuously.
DEGRADATION Linear interpolation (lerp) between the current normal row and
            the current failure row, with a drift factor that grows from 0 → 1
            over _DEGRADATION_HORIZON steps.  Both pointers advance each tick
            so the blended signal is temporally coherent.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from src.core.config import settings
from src.schemas.stream import SensorReading

log = structlog.get_logger(__name__)
logger = logging.getLogger(__name__)

_DEGRADATION_HORIZON: int = 300  # steps until drift reaches 1.0

# Known air-leak failure intervals from the MetroPT-3 paper.
# All timestamps are naive (UTC+0 / local Portuguese winter/summer time as
# published); comparisons use the parquet's native timezone representation.
_FAILURE_WINDOWS: list[tuple[str, str]] = [
    ("2020-04-18 00:00:00", "2020-04-18 23:59:00"),
    ("2020-05-29 23:30:00", "2020-05-30 06:00:00"),
    ("2020-06-05 10:00:00", "2020-06-07 14:30:00"),
    ("2020-07-15 14:30:00", "2020-07-15 19:00:00"),
]

# Column order must stay in sync with SensorReading field order.
_SENSOR_COLS: list[str] = [
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
]


class SimulatorMode(str, Enum):
    NORMAL = "NORMAL"
    DEGRADATION = "DEGRADATION"
    FAILURE = "FAILURE"


class SensorSimulator:
    """
    Stateful data streamer that replays real MetroPT-3 sensor readings.

    Parameters
    ----------
    mode :
        Initial operating mode.
    parquet_path :
        Override the parquet path (useful for tests).  Defaults to
        ``settings.simulator_parquet_path``.
    """

    def __init__(
        self,
        mode: SimulatorMode = SimulatorMode.NORMAL,
        parquet_path: Path | None = None,
    ) -> None:
        self._mode: SimulatorMode = mode
        self._step: int = 0

        resolved_path: Path = parquet_path or settings.simulator_parquet_path
        self._normal: np.ndarray
        self._failure: np.ndarray
        self._normal, self._failure = _load_and_split(resolved_path)

        self._idx_normal: int = 0
        self._idx_failure: int = 0

        log.info(
            "simulator_ready",
            mode=mode.value,
            normal_rows=len(self._normal),
            failure_rows=len(self._failure),
            parquet=str(resolved_path),
        )

    # ── Mode control (RNF-29) ─────────────────────────────────────────────

    @property
    def mode(self) -> SimulatorMode:
        return self._mode

    @mode.setter
    def mode(self, new_mode: SimulatorMode) -> None:
        if new_mode != self._mode:
            log.info(
                "simulator_mode_changed",
                from_=self._mode.value,
                to=new_mode.value,
            )
            self._mode = new_mode
            self._step = 0

    # ── Public API ────────────────────────────────────────────────────────

    def generate_reading(self) -> SensorReading:
        """Return one SensorReading from the current mode's data slice."""
        self._step += 1
        ts: datetime = datetime.now(tz=timezone.utc)

        if self._mode == SimulatorMode.NORMAL:
            return self._read_normal(ts)
        if self._mode == SimulatorMode.FAILURE:
            return self._read_failure(ts)
        return self._read_degradation(ts)

    # ── Private readers ───────────────────────────────────────────────────

    def _read_normal(self, ts: datetime) -> SensorReading:
        row: np.ndarray = self._normal[self._idx_normal]
        self._idx_normal = (self._idx_normal + 1) % len(self._normal)
        return _row_to_reading(row, ts)

    def _read_failure(self, ts: datetime) -> SensorReading:
        row: np.ndarray = self._failure[self._idx_failure]
        self._idx_failure = (self._idx_failure + 1) % len(self._failure)
        return _row_to_reading(row, ts)

    def _read_degradation(self, ts: datetime) -> SensorReading:
        """
        Blend the current normal and failure rows proportionally to drift.

        Both pointers advance each tick so the lerped signal is drawn from
        coherent temporal positions in each partition rather than a fixed
        snapshot — the blended reading represents the compressor transitioning
        through real operating states.
        """
        drift: float = min(self._step / _DEGRADATION_HORIZON, 1.0)

        normal_row: np.ndarray = self._normal[self._idx_normal]
        failure_row: np.ndarray = self._failure[self._idx_failure]

        self._idx_normal = (self._idx_normal + 1) % len(self._normal)
        self._idx_failure = (self._idx_failure + 1) % len(self._failure)

        # Vectorised lerp: blended = normal + drift * (failure - normal)
        blended: np.ndarray = normal_row + drift * (failure_row - normal_row)
        return _row_to_reading(blended, ts)


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions — no state)
# ---------------------------------------------------------------------------


def _build_failure_mask_from_timestamps(timestamps: pd.Series) -> np.ndarray:
    """
    Return a boolean array that is True for rows inside any failure window.

    Strips timezone from both sides so naive and tz-aware timestamps both work.
    """
    ts_naive: pd.Series = timestamps
    if hasattr(timestamps.dtype, "tz") and timestamps.dtype.tz is not None:
        ts_naive = timestamps.dt.tz_localize(None)

    mask: pd.Series = pd.Series(False, index=timestamps.index)
    for start_str, end_str in _FAILURE_WINDOWS:
        start: pd.Timestamp = pd.Timestamp(start_str)
        end: pd.Timestamp = pd.Timestamp(end_str)
        mask |= (ts_naive >= start) & (ts_naive <= end)

    return mask.to_numpy(dtype=bool)


def _load_and_split(parquet_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the MetroPT-3 parquet and return (normal_rows, failure_rows) as
    float32 ndarrays of shape (N, 12).

    Strategy
    --------
    1. Try the ``timestamp`` column to build the failure mask from the known
       air-leak windows.  This is robust even when the ``anomaly`` label has
       alignment issues.
    2. Fall back to the ``anomaly`` binary column if no timestamp is present.
    3. If neither is available, log a warning and treat the entire dataset
       as normal (safe degraded mode — no false failure data injected).
    """
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Simulator parquet not found at {parquet_path}. "
            "Generate it by running `python src/ingest_metropt.py` in apps/ml/, "
            "or override SIMULATOR_PARQUET_PATH in the environment."
        )

    # Read only the columns we need — avoids loading engineered feature columns
    # that may have been added by MetroPTPreprocessor.
    # pyarrow.parquet.read_schema reads only the file footer metadata (no row data).
    import pyarrow.parquet as pq

    all_cols: list[str] = pq.read_schema(parquet_path).names

    has_timestamp: bool = "timestamp" in all_cols
    has_anomaly: bool = "anomaly" in all_cols

    cols_to_read: list[str] = [c for c in _SENSOR_COLS if c in all_cols]
    if has_timestamp:
        cols_to_read.append("timestamp")
    if has_anomaly:
        cols_to_read.append("anomaly")

    df: pd.DataFrame = pd.read_parquet(
        parquet_path,
        columns=cols_to_read,
        engine="pyarrow",
    )

    # Build failure boolean mask.
    if has_timestamp:
        failure_mask: np.ndarray = _build_failure_mask_from_timestamps(df["timestamp"])
        logger.info(
            "Failure mask built from timestamp windows: %d failure rows, %d normal rows",
            failure_mask.sum(),
            (~failure_mask).sum(),
        )
    elif has_anomaly:
        failure_mask = df["anomaly"].to_numpy(dtype=bool)
        logger.info(
            "Failure mask built from anomaly column: %d failure rows, %d normal rows",
            failure_mask.sum(),
            (~failure_mask).sum(),
        )
    else:
        logger.warning(
            "Neither 'timestamp' nor 'anomaly' column found in parquet. "
            "Treating entire dataset as NORMAL — FAILURE mode will replay normal data."
        )
        failure_mask = np.zeros(len(df), dtype=bool)

    sensor_data: np.ndarray = df[_SENSOR_COLS].to_numpy(dtype=np.float32)

    normal_rows: np.ndarray = np.ascontiguousarray(sensor_data[~failure_mask])
    failure_rows: np.ndarray = np.ascontiguousarray(sensor_data[failure_mask])

    if len(failure_rows) == 0:
        logger.warning(
            "No failure rows found — FAILURE mode will replay NORMAL data as fallback."
        )
        failure_rows = normal_rows

    return normal_rows, failure_rows


def _row_to_reading(row: np.ndarray, ts: datetime) -> SensorReading:
    """
    Convert a 12-element float32 array to a SensorReading.

    Index order must match _SENSOR_COLS exactly.
    """
    return SensorReading(
        timestamp=ts,
        TP2=float(row[0]),
        TP3=float(row[1]),
        H1=float(row[2]),
        DV_pressure=float(row[3]),
        Reservoirs=float(row[4]),
        Motor_current=float(row[5]),
        Oil_temperature=float(row[6]),
        COMP=float(row[7]),
        DV_eletric=float(row[8]),
        Towers=float(row[9]),
        MPG=float(row[10]),
        Oil_level=float(row[11]),
    )


# ---------------------------------------------------------------------------
# FastAPI dependency — singleton instantiated lazily on first import
# ---------------------------------------------------------------------------

_simulator: SensorSimulator | None = None


def get_simulator() -> SensorSimulator:
    """
    FastAPI dependency returning the process-wide SensorSimulator singleton.

    Lazy initialisation avoids crashing the entire test suite when the parquet
    is absent; the error surfaces only when the simulator is actually used.
    """
    global _simulator
    if _simulator is None:
        _simulator = SensorSimulator()
    return _simulator
