"""
Shadow deployment and canary routing (champion vs challenger).

* ``ShadowEvaluator.observe`` — the challenger scores the champion's exact
  feature window; the pair is stored in ``shadow_predictions``. The
  challenger never triggers alerts or events.
* ``shadow_report`` — online agreement statistics over a time window:
  class agreement, mean |Δp|, positive rates of each model and the share of
  readings where only one of them would alarm.
* ``is_canary`` — deterministic canary assignment: an asset belongs to the
  canary group when the first 4 bytes of SHA-256(asset_id), scaled to [0, 1),
  fall below ``fraction``. Stable across restarts and replicas, so the same
  asset always runs the same model during the experiment.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timedelta, timezone

import pandas as pd
import structlog
from pydantic import BaseModel
from sqlalchemy import Integer, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.metrics import SHADOW_PREDICTIONS_TOTAL
from src.core.database import AsyncSessionFactory
from src.core.exceptions import ModelNotAvailableError
from src.models.shadow_prediction import ShadowPrediction
from src.schemas.predict import PredictResponse
from src.services.inference_pipeline import _infer_with_history
from src.services.model_registry import ModelRegistry
from src.services.preprocessing import MetroPTPreprocessor

log = structlog.get_logger(__name__)


def is_canary(asset_id: str, fraction: float) -> bool:
    if fraction <= 0:
        return False
    bucket = (
        int.from_bytes(hashlib.sha256(asset_id.encode()).digest()[:4], "big") / 2**32
    )
    return bucket < fraction


class ShadowReport(BaseModel):
    window_start: datetime
    window_end: datetime
    n: int
    champion_model: str | None
    challenger_model: str | None
    class_agreement: float | None
    mean_abs_probability_diff: float | None
    champion_positive_rate: float | None
    challenger_positive_rate: float | None
    only_champion_alarms: int
    only_challenger_alarms: int


class ShadowEvaluator:
    def __init__(
        self, challenger: ModelRegistry, preprocessor: MetroPTPreprocessor | None = None
    ) -> None:
        self._challenger = challenger
        self._preprocessor = preprocessor or MetroPTPreprocessor()

    async def observe(
        self,
        asset_id: str,
        window: pd.DataFrame,
        champion: PredictResponse,
        champion_model: str,
    ) -> ShadowPrediction | None:
        try:
            model = await self._challenger.get()
        except ModelNotAvailableError:
            return None
        try:
            result = await asyncio.to_thread(
                _infer_with_history, model, window, self._preprocessor
            )
        except Exception as exc:
            log.warning("shadow_inference_error", error=str(exc))
            return None
        row = ShadowPrediction(
            timestamp=datetime.fromisoformat(champion.timestamp),
            asset_id=asset_id,
            champion_model=champion_model,
            challenger_model=self._challenger.active_name,
            champion_probability=champion.failure_probability,
            challenger_probability=result.failure_probability,
            champion_class=champion.predicted_class,
            challenger_class=result.predicted_class,
        )
        async with AsyncSessionFactory() as db:
            try:
                db.add(row)
                await db.commit()
            except Exception as exc:
                await db.rollback()
                log.warning("shadow_save_error", error=str(exc))
                return None
        SHADOW_PREDICTIONS_TOTAL.labels(
            agreement=(
                "agree" if row.champion_class == row.challenger_class else "disagree"
            )
        ).inc()
        return row


async def shadow_report(db: AsyncSession, hours: int = 24) -> ShadowReport:
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    in_window = ShadowPrediction.timestamp >= start
    agg = (
        await db.execute(
            select(
                func.count(),
                func.avg(
                    (
                        ShadowPrediction.champion_class
                        == ShadowPrediction.challenger_class
                    ).cast(Integer)
                ),
                func.avg(
                    func.abs(
                        ShadowPrediction.champion_probability
                        - ShadowPrediction.challenger_probability
                    )
                ),
                func.avg(ShadowPrediction.champion_class),
                func.avg(ShadowPrediction.challenger_class),
                func.sum(
                    (
                        (ShadowPrediction.champion_class == 1)
                        & (ShadowPrediction.challenger_class == 0)
                    ).cast(Integer)
                ),
                func.sum(
                    (
                        (ShadowPrediction.champion_class == 0)
                        & (ShadowPrediction.challenger_class == 1)
                    ).cast(Integer)
                ),
            ).where(in_window)
        )
    ).one()
    names = (
        await db.execute(
            select(ShadowPrediction.champion_model, ShadowPrediction.challenger_model)
            .where(in_window)
            .order_by(ShadowPrediction.id.desc())
            .limit(1)
        )
    ).first()
    n = int(agg[0] or 0)

    def f(v: object) -> float | None:
        return None if n == 0 or v is None else float(v)  # type: ignore[arg-type]

    return ShadowReport(
        window_start=start,
        window_end=end,
        n=n,
        champion_model=names[0] if names else None,
        challenger_model=names[1] if names else None,
        class_agreement=f(agg[1]),
        mean_abs_probability_diff=f(agg[2]),
        champion_positive_rate=f(agg[3]),
        challenger_positive_rate=f(agg[4]),
        only_champion_alarms=int(agg[5] or 0),
        only_challenger_alarms=int(agg[6] or 0),
    )
