"""
AssetTracking — the per-reading write path shared by the main inference
pipeline (asset APU-Trem-042, which also feeds the live SSE dashboard) and
the fleet simulator (other assets).

For every inference result:
    1. store the prediction with its ``asset_id`` (RF-09);
    2. feed the RN-03 episode tracker (``FailureEventService``);
    3. when an episode OPENS, freeze the root-cause snapshot (sensor
       occlusion on that asset's own buffer window, RF-32) and schedule the
       RAG plan (RF-35) as a background task.

Everything after step 1 is best-effort: a failure in event tracking or RCA
is logged and never loses the prediction or blocks the stream.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime

import pandas as pd
import structlog

from src.core.config import settings
from src.core.database import AsyncSessionFactory
from src.schemas.predict import PredictRequest, PredictResponse
from src.services.failure_event_service import FailureEventService
from src.services.maintenance_suggestion_service import (
    build_maintenance_suggestion_service,
)
from src.services.prediction_service import save_prediction
from src.services.root_cause_service import (
    RootCauseService,
    WindowScorer,
    sensor_occlusion,
)

log = structlog.get_logger(__name__)


class AssetTracking:
    def __init__(
        self,
        events: FailureEventService,
        root_cause: RootCauseService,
        normal_medians: Callable[[], dict[str, float]],
        llm_enabled: bool | None = None,
    ) -> None:
        self._events = events
        self._rca = root_cause
        self._medians = normal_medians
        self._llm_enabled = (
            settings.rca_llm_enabled if llm_enabled is None else llm_enabled
        )
        self._tasks: set[asyncio.Task[None]] = set()

    async def record(
        self,
        asset_id: str,
        equipment_name: str,
        request: PredictRequest,
        result: PredictResponse,
        model_name: str | None,
        window: pd.DataFrame | None,
        scorer: WindowScorer | None,
        model_threshold: float | None = None,
    ) -> None:
        opened_id: int | None = None
        async with AsyncSessionFactory() as db:
            try:
                await save_prediction(db, request, result, asset_id)
                change = await self._events.process(
                    db,
                    asset_id,
                    datetime.fromisoformat(result.timestamp),
                    result.failure_probability,
                    model_name,
                    model_threshold,
                )
                if (
                    change
                    and change.kind == "opened"
                    and window is not None
                    and scorer is not None
                ):
                    contributions = await asyncio.to_thread(
                        sensor_occlusion, window, scorer, self._medians()
                    )
                    await self._rca.snapshot(db, change.event, contributions)
                    opened_id = change.event.id
                await db.commit()
            except Exception as exc:
                await db.rollback()
                log.warning("asset_tracking_error", asset_id=asset_id, error=str(exc))
                return
        if opened_id is not None and self._llm_enabled:
            task = asyncio.create_task(
                self._enrich(opened_id, equipment_name), name=f"rca-llm-{opened_id}"
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _enrich(self, event_id: int, equipment_name: str) -> None:
        async with AsyncSessionFactory() as db:
            await self._rca.enrich_with_suggestion(
                db, event_id, equipment_name, build_maintenance_suggestion_service(db)
            )

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
