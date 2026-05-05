"""
Stateful sensor buffer for production feature engineering (V2 — Phase 2).

Why this exists
---------------
Lag and rolling features (``TP2_lag_5``, ``TP2_std_5``, ``TP2_ma_15``) require
the *previous N samples* to compute correctly.  ``ModelService._build_feature_row``
runs in the request-handling path with access only to the current snapshot,
so it falls back to neutral defaults (0 for std, current value for MA, 0 for
lag).  That preserves a valid feature vector but throws away the trend signal
the V2 models were trained to consume.

This buffer holds the most recent ``window_size`` raw readings keyed to the
``MetroPTPreprocessor`` schema, so the same transform used at training time
can be re-applied at inference time once the window is warm.

Wiring strategy
---------------
The HTTP ``POST /predict/`` endpoint must remain stateless (each call is an
independent test or operator override; concurrent unrelated requests must not
contaminate each other's history).  The natural consumer is
``InferencePipelineService``, which pulls from a single ordered SSE stream at
1 Hz — exactly the cadence the rolling windows were trained on.

This module exposes the buffer as a utility; integration with the pipeline
service is a follow-up PR (see V2 plan, Phase 2).

Thread safety
-------------
``deque.append`` is atomic under the GIL but iteration through the snapshot
reader is not — we wrap mutators with ``threading.Lock`` so concurrent writers
in different worker threads don't see torn state.  Cost is negligible (single
acquire per 1 Hz tick).
"""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Any

import pandas as pd


class SensorBuffer:
    """
    Bounded FIFO of the most recent raw sensor snapshots.

    Parameters
    ----------
    window_size:
        Maximum number of readings retained.  Should equal or exceed the
        longest rolling/lag window used by the preprocessor (default 30 →
        covers ``ma_15`` plus a 15-sample ``lag_15`` lookback).
    warmup_size:
        Minimum number of samples required before
        :meth:`is_warm` returns ``True`` and stateful features can be trusted.
        Defaults to 15 — the longest window the V1 preprocessor uses.
    """

    def __init__(self, window_size: int = 30, warmup_size: int = 15) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if warmup_size <= 0 or warmup_size > window_size:
            raise ValueError("warmup_size must be in (0, window_size]")

        self._buf: deque[dict[str, float]] = deque(maxlen=window_size)
        self._lock: Lock = Lock()
        self._warmup_size: int = warmup_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, reading: dict[str, float]) -> None:
        """Append a single sensor snapshot.  Oldest entry is evicted automatically."""
        with self._lock:
            self._buf.append(dict(reading))  # defensive copy

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return a snapshot of the current buffer as a DataFrame.

        Rows are ordered oldest → newest, matching the temporal order expected
        by ``MetroPTPreprocessor`` (which uses ``rolling`` with the implicit
        assumption that index 0 is the oldest sample).
        """
        with self._lock:
            return pd.DataFrame(list(self._buf))

    def is_warm(self) -> bool:
        """``True`` once the buffer holds enough samples for stateful features."""
        with self._lock:
            return len(self._buf) >= self._warmup_size

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def clear(self) -> None:
        """Drop all buffered readings — useful on simulator mode change."""
        with self._lock:
            self._buf.clear()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        return self._buf.maxlen or 0

    @property
    def warmup_size(self) -> int:
        return self._warmup_size


# Module-level singleton — same pattern as ws_manager.manager.
buffer: SensorBuffer = SensorBuffer(window_size=30, warmup_size=15)


def get_sensor_buffer() -> SensorBuffer:
    """FastAPI dependency factory — exposes the process-wide buffer."""
    return buffer
