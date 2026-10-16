"""
RootCauseService — per-event explanation (RF-32 online) + RAG suggestion (RF-35).

Online explanation by SENSOR OCCLUSION
--------------------------------------
Production serves the model as ONNX, where exact TreeSHAP is not available.
For each of the 12 MetroPT-3 sensors we replace that sensor's whole raw
window in the buffer by its normal-operation median, recompute every derived
feature (rolling means, lags, cross-sensor ratios) with the same
preprocessor, and measure the drop in P(failure):

    contribution(s) = P(failure | real window) − P(failure | s neutralised)

Positive = the sensor pushes towards failure. This is the sensor-level
analogue of the offline TreeSHAP aggregation (apps/ml/src/explainability.py)
and costs 13 model calls per OPENED event (not per reading). Its agreement
with TreeSHAP is measured offline (benchmarks/benchmark_occlusion_vs_shap.py).

The snapshot is FROZEN in ``root_cause_analyses`` when the event opens; the
RAG suggestion (MCP + Ollama, with the XAI-augmented query of RF-35 and the
RN-04 / RNF-84 guards) is filled in afterwards by a background task so a slow
or unavailable LLM never stalls inference.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.failure_event import FailureEvent, RootCauseAnalysis
from src.schemas.maintenance import MaintenanceSuggestionRequest, SensorContribution
from src.services.maintenance_suggestion_service import MaintenanceSuggestionService

log = structlog.get_logger(__name__)

SENSORS: tuple[str, ...] = (
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

# P(failure) for a raw buffer window (feature engineering + model inside).
WindowScorer = Callable[[pd.DataFrame], float]


def sensor_occlusion(
    window: pd.DataFrame, score: WindowScorer, normal_medians: dict[str, float]
) -> list[dict[str, float | str]]:
    """Per-sensor contribution, sorted by |contribution| (largest first)."""
    base = score(window)
    out: list[dict[str, float | str]] = []
    for sensor in SENSORS:
        if sensor not in window.columns or sensor not in normal_medians:
            continue
        neutral = window.copy()
        neutral[sensor] = normal_medians[sensor]
        out.append({"sensor": sensor, "contribution": round(base - score(neutral), 6)})
    return sorted(out, key=lambda c: -abs(float(c["contribution"])))


class RootCauseService:
    async def snapshot(
        self, db: AsyncSession, event: FailureEvent, contributions: list[dict[str, Any]]
    ) -> RootCauseAnalysis:
        rca = RootCauseAnalysis(
            event_id=event.id,
            method="sensor_occlusion",
            sensor_contributions=contributions,
            references=[],
        )
        db.add(rca)
        await db.flush()
        return rca

    async def enrich_with_suggestion(
        self,
        db: AsyncSession,
        event_id: int,
        equipment_name: str,
        suggestion_service: MaintenanceSuggestionService,
    ) -> None:
        """Fill the RAG plan of an existing snapshot. Errors are logged and
        swallowed: the explanation stays valid without the plan."""
        event = await db.get(FailureEvent, event_id)
        rca = await db.scalar(
            select(RootCauseAnalysis).where(RootCauseAnalysis.event_id == event_id)
        )
        if event is None or rca is None:
            return
        request = MaintenanceSuggestionRequest(
            failure_probability=min(max(event.peak_probability, 0.0), 1.0),
            equipment_name=equipment_name,
            sensor_contributions=[
                SensorContribution(
                    sensor=c["sensor"], contribution=float(c["contribution"])
                )
                for c in rca.sensor_contributions
            ],
        )
        try:
            resp = await suggestion_service.suggest(request)
        except Exception:
            log.warning("rca_suggestion_failed", event_id=event_id, exc_info=True)
            return
        rca.suggestion_markdown = resp.markdown
        rca.references = [r.model_dump() for r in resp.references]
        rca.grounding = resp.grounding.model_dump() if resp.grounding else None
        rca.llm_model = resp.model
        await db.commit()
        log.info(
            "rca_suggestion_saved",
            event_id=event_id,
            grounded=rca.grounding is not None,
        )
