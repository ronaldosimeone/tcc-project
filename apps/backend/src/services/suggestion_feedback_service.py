"""
SuggestionFeedbackService — RF-38 (human-in-the-loop).

Stores the technician's verdict on each maintenance plan and turns the
per-reference votes into a retrieval prior:

    prior(file, page) = (helpful − unhelpful) / (helpful + unhelpful + 2)

(Laplace-smoothed, in (−1, 1); a single vote moves it only to ±1/3, so one
technician cannot dominate the ranking).

Re-ranking (``rerank``) adds ``weight · prior`` to the cosine score ONLY to
order the chunks that already passed τ (RN-04). The grounding decision
always uses the original cosine score — feedback can reorder evidence, never
turn an ungrounded answer into a grounded one.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol, TypeVar

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.suggestion_feedback import SuggestionFeedback
from src.schemas.maintenance import (
    ReferencePrior,
    SuggestionFeedbackCreate,
    SuggestionFeedbackResponse,
)

log = structlog.get_logger(__name__)

RERANK_WEIGHT: float = 0.05

PriorKey = tuple[str, int]
PriorProvider = Callable[[str], Awaitable[dict[PriorKey, float]]]


class _Chunk(Protocol):
    file_name: str
    page: int
    score: float


ChunkT = TypeVar("ChunkT", bound=_Chunk)


def laplace_prior(helpful: int, unhelpful: int) -> float:
    return (helpful - unhelpful) / (helpful + unhelpful + 2)


def rerank(
    contexts: Sequence[ChunkT],
    priors: dict[PriorKey, float],
    weight: float = RERANK_WEIGHT,
) -> list[ChunkT]:
    """Stable sort by ``score + weight · prior`` (priors default to 0)."""
    return sorted(
        contexts,
        key=lambda c: c.score + weight * priors.get((c.file_name, c.page), 0.0),
        reverse=True,
    )


class SuggestionFeedbackService:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(
        self, payload: SuggestionFeedbackCreate
    ) -> SuggestionFeedbackResponse:
        row = SuggestionFeedback(
            equipment_name=payload.equipment_name,
            symptom_description=payload.symptom_description,
            query=payload.query,
            model=payload.model,
            grounding_status=payload.grounding_status,
            useful=payload.useful,
            reference_votes=[v.model_dump() for v in payload.reference_votes],
            comment=payload.comment,
        )
        self._db.add(row)
        await self._db.commit()
        await self._db.refresh(row)
        log.info(
            "suggestion_feedback_saved",
            id=row.id,
            useful=row.useful,
            votes=len(row.reference_votes),
        )
        return SuggestionFeedbackResponse(id=row.id, created_at=row.created_at)

    async def reference_priors(self, equipment_name: str) -> list[ReferencePrior]:
        result = await self._db.execute(
            select(SuggestionFeedback.reference_votes).where(
                SuggestionFeedback.equipment_name == equipment_name
            )
        )
        counts: dict[PriorKey, list[int]] = defaultdict(lambda: [0, 0])
        for (votes,) in result.all():
            for v in votes or []:
                key = (str(v["file_name"]), int(v["page"]))
                counts[key][0 if v["helpful"] else 1] += 1
        return [
            ReferencePrior(
                file_name=f,
                page=p,
                helpful_votes=h,
                unhelpful_votes=u,
                prior=laplace_prior(h, u),
            )
            for (f, p), (h, u) in sorted(counts.items())
        ]

    async def prior_map(self, equipment_name: str) -> dict[PriorKey, float]:
        return {
            (r.file_name, r.page): r.prior
            for r in await self.reference_priors(equipment_name)
        }
