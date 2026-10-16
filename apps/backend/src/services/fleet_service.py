"""
FleetService — N simulated compressors besides APU-Trem-042.

Each fleet asset is an independent MetroPT-3 replay:
    * shared data arrays (one parquet in memory for the whole fleet);
    * its own replay offset (assets are out of phase), its own sensor noise
      and RNG seed — two assets never emit the same reading;
    * its own operating-mode schedule (NORMAL → DEGRADATION → FAILURE →
      NORMAL …) with random durations, so failures appear at different times
      across the fleet;
    * its own feature buffer, and the same write path as the main pipeline
      (``AssetTracking``: prediction + RN-03 events + root cause).

Assets are registered in ``assets`` on start (ISO 14224 fields). Disabled when
``FLEET_SIZE=0`` (default), so the original single-asset deployment is
unchanged.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import structlog
from sqlalchemy import select

from src.core.database import AsyncSessionFactory
from src.models.asset import Asset
from src.models.prediction import DEFAULT_ASSET_ID
from src.schemas.stream import SensorReading
from src.services.asset_processor import AssetProcessor
from src.services.asset_tracking import AssetTracking
from src.services.model_registry import ModelRegistry
from src.services.shadow_service import is_canary
from src.services.simulator import SensorSimulator, SimulatorMode

log = structlog.get_logger(__name__)

TICK_SECONDS: float = 1.0
# Mode durations in ticks (seconds): (low, high).
_DURATIONS: dict[SimulatorMode, tuple[int, int]] = {
    SimulatorMode.NORMAL: (600, 3600),
    SimulatorMode.DEGRADATION: (120, 300),
    SimulatorMode.FAILURE: (60, 240),
}
_NEXT: dict[SimulatorMode, SimulatorMode] = {
    SimulatorMode.NORMAL: SimulatorMode.DEGRADATION,
    SimulatorMode.DEGRADATION: SimulatorMode.FAILURE,
    SimulatorMode.FAILURE: SimulatorMode.NORMAL,
}


def fleet_asset_ids(size: int) -> list[str]:
    """``APU-Trem-001`` … skipping the main asset ``APU-Trem-042``."""
    ids: list[str] = []
    n = 1
    while len(ids) < size:
        candidate = f"APU-Trem-{n:03d}"
        if candidate != DEFAULT_ASSET_ID:
            ids.append(candidate)
        n += 1
    return ids


@dataclass
class ModeSchedule:
    rng: np.random.Generator
    mode: SimulatorMode = SimulatorMode.NORMAL
    remaining: int = 0

    def __post_init__(self) -> None:
        # Desynchronise assets: start somewhere inside a NORMAL period.
        self.remaining = int(self.rng.integers(*_DURATIONS[SimulatorMode.NORMAL]))

    def tick(self) -> SimulatorMode:
        self.remaining -= 1
        if self.remaining <= 0:
            self.mode = _NEXT[self.mode]
            self.remaining = int(self.rng.integers(*_DURATIONS[self.mode]))
        return self.mode


class TelemetryPublisher(Protocol):
    async def publish(self, asset_id: str, reading: SensorReading) -> None: ...


@dataclass
class FleetAsset:
    asset_id: str
    simulator: SensorSimulator
    schedule: ModeSchedule
    processor: AssetProcessor


class FleetService:
    def __init__(
        self,
        size: int,
        registry: ModelRegistry,
        tracking: AssetTracking,
        base_simulator: SensorSimulator,
        noise_frac: float = 0.02,
        seed: int = 42,
        publisher: TelemetryPublisher | None = None,
        canary_registry: ModelRegistry | None = None,
        canary_fraction: float = 0.0,
    ) -> None:
        """``publisher`` set → readings go to the broker (MQTT) and are
        processed by the ingestor; None → processed in-process (default)."""
        self._registry = registry
        self._tracking = tracking
        self._publisher = publisher
        rng = np.random.default_rng(seed)
        data = base_simulator.data
        self.assets: list[FleetAsset] = [
            FleetAsset(
                asset_id=asset_id,
                simulator=SensorSimulator(
                    data=data,
                    start_offset=int(rng.integers(0, len(data[0]))),
                    noise_frac=noise_frac,
                    seed=int(rng.integers(0, 2**31)),
                ),
                schedule=ModeSchedule(
                    np.random.default_rng(int(rng.integers(0, 2**31)))
                ),
                processor=AssetProcessor(
                    asset_id,
                    (
                        canary_registry
                        if canary_registry is not None
                        and is_canary(asset_id, canary_fraction)
                        else registry
                    ),
                    tracking,
                ),
            )
            for asset_id in fleet_asset_ids(size)
        ]
        self._task: asyncio.Task[None] | None = None
        self.last_tick_ms: float | None = None

    async def register_assets(self) -> None:
        async with AsyncSessionFactory() as db:
            existing = set((await db.scalars(select(Asset.id))).all())
            for a in self.assets:
                if a.asset_id in existing:
                    continue
                db.add(
                    Asset(
                        id=a.asset_id,
                        name=f"Compressor de ar da APU — Trem {a.asset_id[-3:]}",
                        installation="Metro do Porto",
                        system="Unidade de produção de ar (APU)",
                        equipment_unit="Compressor de ar",
                        subunit="Compressor + secador de torre dupla",
                        manufacturer_model="MetroPT-3 (replay simulado)",
                        criticality=2,
                    )
                )
            await db.commit()

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="fleet-simulator")
        log.info("fleet_started", assets=len(self.assets))

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("fleet_stopped")

    async def _run(self) -> None:
        await self.register_assets()
        while True:
            started = time.perf_counter()
            await self.tick()
            self.last_tick_ms = (time.perf_counter() - started) * 1000
            await asyncio.sleep(max(TICK_SECONDS - self.last_tick_ms / 1000, 0.0))

    async def tick(self) -> None:
        for a in self.assets:
            a.simulator.mode = a.schedule.tick()
            reading = a.simulator.generate_reading()
            if self._publisher is not None:
                try:
                    await self._publisher.publish(a.asset_id, reading)
                except Exception as exc:  # broker down: the next tick retries
                    log.warning(
                        "fleet_publish_error", asset_id=a.asset_id, error=str(exc)
                    )
                continue
            await a.processor.process(reading)
