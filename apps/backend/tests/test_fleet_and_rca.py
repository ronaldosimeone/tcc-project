"""
Frota multi-ativo + causa raiz online.

Garantias verificadas
---------------------
1.  IDs da frota pulam o ativo principal APU-Trem-042.
2.  A agenda de modos percorre NORMAL → DEGRADATION → FAILURE → NORMAL e é
    determinística por seed.
3.  Simuladores com dados compartilhados mas offset/seed diferentes nunca
    emitem a mesma leitura; ruído afeta só os sensores analógicos.
4.  Oclusão por sensor atribui a contribuição ao sensor que move o score.
5.  AssetTracking: ao abrir um episódio grava o snapshot de causa raiz com as
    contribuições, no MESMO commit da predição.
6.  FleetService.tick: registra ativos, infere por ativo e abre eventos por
    ativo de forma independente.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import src.services.asset_tracking as tracking_module
import src.services.fleet_service as fleet_module
from src.core.database import Base
from src.models.asset import Asset
from src.models.failure_event import FailureEvent, RootCauseAnalysis
from src.models.prediction import Prediction
from src.schemas.predict import PredictRequest, PredictResponse
from src.services.asset_tracking import AssetTracking
from src.services.failure_event_service import FailureEventService
from src.services.fleet_service import FleetService, ModeSchedule, fleet_asset_ids
from src.services.root_cause_service import RootCauseService, sensor_occlusion
from src.services.simulator import SensorSimulator, SimulatorMode

SENSORS = (
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
)


def _data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    normal = rng.normal(5, 1, (500, 12)).astype(np.float32)
    normal[:, 7:] = rng.integers(0, 2, (500, 5))
    failure = normal[:50].copy()
    failure[:, :7] += 3  # only analogue sensors; binary ones stay 0/1
    return normal, failure


def test_fleet_ids_skip_main_asset() -> None:
    ids = fleet_asset_ids(45)
    assert len(ids) == 45 and "APU-Trem-042" not in ids and ids[-1] == "APU-Trem-046"


def test_mode_schedule_cycles_and_is_deterministic() -> None:
    def run(seed: int) -> list[SimulatorMode]:
        sched = ModeSchedule(np.random.default_rng(seed))
        return [sched.tick() for _ in range(12_000)]

    modes = run(1)
    assert modes == run(1)
    order = [m for i, m in enumerate(modes) if i == 0 or m != modes[i - 1]]
    assert order[:4] == [
        SimulatorMode.NORMAL,
        SimulatorMode.DEGRADATION,
        SimulatorMode.FAILURE,
        SimulatorMode.NORMAL,
    ]


def test_shared_data_offset_and_noise() -> None:
    data = _data()
    a = SensorSimulator(data=data, start_offset=0, noise_frac=0.1, seed=1)
    b = SensorSimulator(data=data, start_offset=0, noise_frac=0.1, seed=2)
    ra, rb = a.generate_reading(), b.generate_reading()
    assert ra.TP2 != rb.TP2  # different noise
    assert ra.COMP == rb.COMP == float(data[0][0, 7])  # binary sensors untouched
    c = SensorSimulator(data=data, start_offset=10)
    assert c.generate_reading().TP2 == pytest.approx(float(data[0][10, 0]))


def test_sensor_occlusion_attributes_to_driving_sensor() -> None:
    window = pd.DataFrame({s: np.full(30, 5.0) for s in SENSORS})
    window["Towers"] = 9.0

    def score(w: pd.DataFrame) -> float:
        return float(np.clip((w["Towers"].mean() - 5.0) / 4.0, 0, 1))

    contrib = sensor_occlusion(window, score, {s: 5.0 for s in SENSORS})
    assert contrib[0]["sensor"] == "Towers" and contrib[0][
        "contribution"
    ] == pytest.approx(1.0)
    assert all(c["contribution"] == 0 for c in contrib[1:])


@pytest.fixture()
async def session_factory(monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[Any, None]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )
    async with factory() as s:
        s.add(
            Asset(
                id="APU-Trem-042",
                name="x",
                installation="x",
                system="x",
                equipment_unit="x",
                subunit="x",
            )
        )
        await s.commit()
    monkeypatch.setattr(tracking_module, "AsyncSessionFactory", factory)
    monkeypatch.setattr(fleet_module, "AsyncSessionFactory", factory)
    yield factory
    await engine.dispose()


def _req() -> PredictRequest:
    return PredictRequest(**{s: 1.0 for s in SENSORS})


async def test_tracking_snapshots_rca_on_open(session_factory: Any) -> None:
    tracking = AssetTracking(
        FailureEventService(k_open=3),
        RootCauseService(),
        lambda: {s: 5.0 for s in SENSORS},
        llm_enabled=False,
    )
    window = pd.DataFrame({s: np.full(30, 5.0) for s in SENSORS}).assign(TP2=9.0)

    def scorer(w: pd.DataFrame) -> float:
        return 0.9 if w["TP2"].mean() > 6 else 0.1

    for i in range(4):
        result = PredictResponse(
            predicted_class=1,
            failure_probability=0.9,
            timestamp=datetime(2026, 1, 1, 0, 0, i, tzinfo=timezone.utc).isoformat(),
        )
        await tracking.record(
            "APU-Trem-042", "Compressor", _req(), result, "rf", window, scorer
        )

    async with session_factory() as db:
        assert len((await db.scalars(select(Prediction))).all()) == 4
        event = (await db.scalars(select(FailureEvent))).one()
        rca = (await db.scalars(select(RootCauseAnalysis))).one()
    assert event.n_samples == 4 and rca.event_id == event.id
    assert rca.method == "sensor_occlusion"
    assert rca.sensor_contributions[0] == {
        "sensor": "TP2",
        "contribution": pytest.approx(0.8),
    }


class _FakeModel:
    def predict_from_features(self, row: pd.DataFrame) -> PredictResponse:
        p = 0.95 if float(row["TP2"].iloc[0]) > 7 else 0.05
        return PredictResponse(
            predicted_class=int(p > 0.5),
            failure_probability=p,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


class _FakeRegistry:
    active_name = "fake"

    async def get(self) -> _FakeModel:
        return _FakeModel()


async def test_fleet_tick_tracks_assets_independently(session_factory: Any) -> None:
    data = _data()
    base = SensorSimulator(data=data)
    tracking = AssetTracking(
        FailureEventService(k_open=3, k_close=3),
        RootCauseService(),
        base.normal_medians,
        llm_enabled=False,
    )
    fleet = FleetService(2, _FakeRegistry(), tracking, base, noise_frac=0.0, seed=7)  # type: ignore[arg-type]
    await fleet.register_assets()
    # Asset 0 fails (failure rows have TP2 ≈ 8), asset 1 stays normal.
    fleet.assets[0].schedule.mode = SimulatorMode.FAILURE
    fleet.assets[0].schedule.remaining = 10_000
    fleet.assets[1].schedule.remaining = 10_000
    for _ in range(25):
        await fleet.tick()

    async with session_factory() as db:
        assets = set((await db.scalars(select(Asset.id))).all())
        events = (await db.scalars(select(FailureEvent))).all()
        per_asset = {
            aid: len(
                (
                    await db.scalars(
                        select(Prediction).where(Prediction.asset_id == aid)
                    )
                ).all()
            )
            for aid in assets
        }
    assert {"APU-Trem-001", "APU-Trem-002"} <= assets
    assert (
        per_asset["APU-Trem-001"] == per_asset["APU-Trem-002"] == 25 - 14
    )  # warm-up = 15 readings
    assert [e.asset_id for e in events] == ["APU-Trem-001"]
