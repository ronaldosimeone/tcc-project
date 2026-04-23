"""
Pub/Sub broadcaster for real-time sensor data (RF-12, RNF-28).

Architecture
------------
One asyncio.Task runs `_broadcast_loop`, waking every BROADCAST_INTERVAL
seconds and pushing a fresh SensorReading into every subscriber's queue via
put_nowait().  Each SSE client owns exactly one asyncio.Queue; slow consumers
are evicted (queue full) rather than blocking the broadcaster.

This means:
  - RF-12: the clock is read exactly once per cycle → all clients see data
    at the same 1-second cadence.
  - RNF-28: 50+ clients share a single background task; adding a client is
    O(1) (one set.add + one put_nowait per cycle).
  - No memory leak: the task cancels itself when the subscriber set empties.
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone

import structlog

from src.schemas.stream import SensorReading

log = structlog.get_logger(__name__)

BROADCAST_INTERVAL: float = 1.0  # RF-12: exactly 1 second between events
_QUEUE_MAX_SIZE: int = 10  # backpressure cap; slow consumers are evicted


class SensorStreamService:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[SensorReading]] = set()
        self._broadcast_task: asyncio.Task[None] | None = None

    # ── Public API ────────────────────────────────────────────────────────

    def subscribe(self) -> asyncio.Queue[SensorReading]:
        """Register a new SSE client and return its dedicated queue."""
        queue: asyncio.Queue[SensorReading] = asyncio.Queue(maxsize=_QUEUE_MAX_SIZE)
        self._subscribers.add(queue)
        log.info("sse_subscribed", total=len(self._subscribers))
        self._ensure_broadcast()
        return queue

    def unsubscribe(self, queue: asyncio.Queue[SensorReading]) -> None:
        """Deregister a client queue (idempotent). Stops the task when empty."""
        self._subscribers.discard(queue)
        log.info("sse_unsubscribed", total=len(self._subscribers))
        if not self._subscribers:
            self._stop_broadcast()

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    # ── Broadcast task lifecycle ──────────────────────────────────────────

    def _ensure_broadcast(self) -> None:
        if self._broadcast_task is None or self._broadcast_task.done():
            self._broadcast_task = asyncio.create_task(
                self._broadcast_loop(), name="sse-sensor-broadcast"
            )
            log.debug("sse_broadcast_started")

    def _stop_broadcast(self) -> None:
        if self._broadcast_task and not self._broadcast_task.done():
            self._broadcast_task.cancel()
            log.debug("sse_broadcast_stopped")

    async def _broadcast_loop(self) -> None:
        try:
            while self._subscribers:
                reading = self._generate_reading()
                evicted: list[asyncio.Queue[SensorReading]] = []

                for queue in list(self._subscribers):
                    try:
                        queue.put_nowait(reading)
                    except asyncio.QueueFull:
                        evicted.append(queue)

                if evicted:
                    for q in evicted:
                        self._subscribers.discard(q)
                    log.warning("sse_slow_consumers_evicted", count=len(evicted))

                await asyncio.sleep(BROADCAST_INTERVAL)
        except asyncio.CancelledError:
            log.debug("sse_broadcast_loop_cancelled")

    # ── Data simulation ───────────────────────────────────────────────────

    def _generate_reading(self) -> SensorReading:
        """
        Simulate realistic SCADA readings for an air-compressor unit.
        Distributions derived from the MetroPT-3 dataset statistics.
        """
        return SensorReading(
            timestamp=datetime.now(tz=timezone.utc),
            TP2=round(random.gauss(5.90, 0.40), 4),
            TP3=round(random.gauss(3.40, 0.30), 4),
            H1=round(random.gauss(45.0, 8.00), 4),
            DV_pressure=round(random.gauss(0.10, 0.05), 4),
            Reservoirs=round(random.gauss(7.00, 0.50), 4),
            Motor_current=round(random.gauss(2.80, 0.60), 4),
            Oil_temperature=round(random.gauss(58.0, 5.00), 4),
            COMP=float(random.choices([0.0, 1.0], weights=[0.30, 0.70])[0]),
            DV_eletric=float(random.choices([0.0, 1.0], weights=[0.50, 0.50])[0]),
            Towers=float(random.choices([0.0, 1.0], weights=[0.40, 0.60])[0]),
            MPG=float(random.choices([0.0, 1.0], weights=[0.20, 0.80])[0]),
            Oil_level=float(random.choices([0.0, 1.0], weights=[0.10, 0.90])[0]),
        )


# Module-level singleton — same pattern as ws_manager.
_service = SensorStreamService()


def get_sensor_stream_service() -> SensorStreamService:
    """FastAPI dependency returning the process-wide singleton."""
    return _service
