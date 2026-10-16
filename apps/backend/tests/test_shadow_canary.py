"""
Champion × challenger: shadow deployment e canary.

Garantias verificadas
---------------------
1.  ``is_canary`` é determinístico, respeita os extremos (0 e 1) e a fração
    aproximada numa frota grande.
2.  ShadowEvaluator grava o par (champion, challenger) sobre a MESMA janela
    e nunca falha se o desafiante não estiver carregado.
3.  O relatório agrega concordância, |Δp| e alarmes exclusivos de cada modelo.
4.  FleetService envia ativos canary ao registry do desafiante.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import src.services.shadow_service as shadow_module
from src.core.database import Base
from src.core.exceptions import ModelNotAvailableError
from src.schemas.predict import PredictResponse
from src.services.fleet_service import FleetService
from src.services.shadow_service import ShadowEvaluator, is_canary, shadow_report
from src.services.simulator import SensorSimulator

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


def test_is_canary_deterministic_and_fractional() -> None:
    ids = [f"APU-Trem-{i:04d}" for i in range(4000)]
    assert [is_canary(a, 0.25) for a in ids] == [is_canary(a, 0.25) for a in ids]
    assert not any(is_canary(a, 0.0) for a in ids)
    assert all(is_canary(a, 1.0) for a in ids)
    assert 0.22 < np.mean([is_canary(a, 0.25) for a in ids]) < 0.28


class _Model:
    def __init__(self, p: float) -> None:
        self.p = p

    def predict_from_features(self, row: pd.DataFrame) -> PredictResponse:
        return PredictResponse(
            predicted_class=int(self.p >= 0.5),
            failure_probability=self.p,
            timestamp="2026-01-01T00:00:00+00:00",
        )


class _Registry:
    def __init__(self, p: float | None, name: str = "xgboost_v2") -> None:
        self._p = p
        self.active_name = name

    async def get(self) -> _Model:
        if self._p is None:
            raise ModelNotAvailableError("not loaded")
        return _Model(self._p)


@pytest.fixture()
async def factory(monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[Any, None]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    f = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(shadow_module, "AsyncSessionFactory", f)
    yield f
    await engine.dispose()


def _window() -> pd.DataFrame:
    return pd.DataFrame({s: np.full(30, 1.0) for s in SENSORS})


async def test_shadow_records_pairs_and_report(factory: Any) -> None:
    ev = ShadowEvaluator(_Registry(0.9))  # type: ignore[arg-type]
    now = datetime.now(timezone.utc)
    for i, champ_p in enumerate([0.2, 0.2, 0.8, 0.95]):
        champion = PredictResponse(
            predicted_class=int(champ_p >= 0.5),
            failure_probability=champ_p,
            timestamp=(now - timedelta(seconds=10 * i)).isoformat(),
        )
        await ev.observe("APU-Trem-042", _window(), champion, "random_forest_v2")

    async with factory() as db:
        report = await shadow_report(db, hours=1)
    assert report.n == 4
    assert (
        report.champion_model == "random_forest_v2"
        and report.challenger_model == "xgboost_v2"
    )
    assert report.class_agreement == pytest.approx(0.5)
    assert report.only_challenger_alarms == 2 and report.only_champion_alarms == 0
    assert report.mean_abs_probability_diff == pytest.approx(
        (0.7 + 0.7 + 0.1 + 0.05) / 4
    )


async def test_shadow_without_challenger_is_noop(factory: Any) -> None:
    champion = PredictResponse(
        predicted_class=0,
        failure_probability=0.1,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    assert await ShadowEvaluator(_Registry(None)).observe("A", _window(), champion, "rf") is None  # type: ignore[arg-type]
    async with factory() as db:
        assert (await shadow_report(db)).n == 0


def test_fleet_routes_canary_assets_to_challenger() -> None:
    rng = np.random.default_rng(0)
    normal = rng.normal(5, 1, (100, 12)).astype(np.float32)
    base = SensorSimulator(data=(normal, normal))
    champion, challenger = _Registry(0.1, "rf"), _Registry(0.9, "xgb")
    fleet = FleetService(
        20,
        cast(Any, champion),
        cast(Any, object()),
        base,
        canary_registry=cast(Any, challenger),
        canary_fraction=0.3,
    )
    for a in fleet.assets:
        expected = challenger if is_canary(a.asset_id, 0.3) else champion
        assert a.processor._registry is expected  # noqa: SLF001
    assert any(
        a.processor._registry is challenger for a in fleet.assets
    )  # noqa: SLF001
