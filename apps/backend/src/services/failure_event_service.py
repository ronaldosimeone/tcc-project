"""
FailureEventService — persists RN-03 alarm episodes per asset.

The inference pipeline calls ``process`` after every stored prediction. One
``EventTracker`` per asset lives in memory; the service turns its updates
into rows:

    opened   -> INSERT failure_events (status=open) + optional RCA hook
    extended -> UPDATE peak / mean / n_samples
    closed   -> UPDATE ended_at, status=closed

Restart safety: tracker state is in memory, so after a restart an event left
``open`` in the database can never be closed by the stream. The first time
an asset is seen, any open event of that asset is closed at its last update
(``updated_at``) — an honest "observation ended here" instead of an event
that stays open forever.

``rebuild`` recomputes the history of an asset from the stored predictions
with the same tracker (used to backfill and to change k/threshold); it
refuses to touch assets that already have work orders, because those carry
human labels that must never be silently discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import structlog
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import ConflictError
from src.models.failure_event import FailureEvent, RootCauseAnalysis, WorkOrder
from src.models.prediction import Prediction
from src.services.event_tracker import EpisodeStats, EventTracker, severity_for

log = structlog.get_logger(__name__)

ChangeKind = Literal["opened", "extended", "closed"]


@dataclass(frozen=True)
class EventChange:
    event: FailureEvent
    kind: ChangeKind


class FailureEventService:
    def __init__(
        self,
        threshold: float = 0.7,
        k_open: int = 6,
        k_close: int = 6,
        critical_at: float = 0.9,
    ) -> None:
        self.threshold = threshold
        self.k_open = k_open
        self.k_close = k_close
        self.critical_at = critical_at
        self._trackers: dict[str, EventTracker] = {}
        self._open_ids: dict[str, int] = {}

    def _tracker(self, asset_id: str) -> EventTracker:
        if asset_id not in self._trackers:
            self._trackers[asset_id] = EventTracker(
                self.threshold, self.k_open, self.k_close
            )
        return self._trackers[asset_id]

    async def _close_stale(self, db: AsyncSession, asset_id: str) -> None:
        await db.execute(
            update(FailureEvent)
            .where(FailureEvent.asset_id == asset_id, FailureEvent.status == "open")
            .values(status="closed", ended_at=FailureEvent.updated_at)
        )

    def _apply(self, event: FailureEvent, stats: EpisodeStats) -> None:
        event.peak_probability = stats.peak
        event.mean_probability = stats.mean
        event.n_samples = stats.n
        event.severity = severity_for(stats.peak, self.critical_at)

    async def process(
        self,
        db: AsyncSession,
        asset_id: str,
        ts: datetime,
        probability: float,
        model_name: str | None = None,
    ) -> EventChange | None:
        """Feed one prediction. Returns the event row touched and how, if any.
        The caller commits (and takes the RCA snapshot on ``opened``)."""
        if asset_id not in self._trackers:
            await self._close_stale(db, asset_id)
        upd = self._tracker(asset_id).update(ts, probability)

        if upd.opened:
            event = FailureEvent(
                asset_id=asset_id,
                started_at=upd.opened.started_at,
                status="open",
                severity=severity_for(upd.opened.peak, self.critical_at),
                peak_probability=upd.opened.peak,
                mean_probability=upd.opened.mean,
                n_samples=upd.opened.n,
                model_name=model_name,
                threshold=self.threshold,
            )
            db.add(event)
            await db.flush()
            self._open_ids[asset_id] = event.id
            log.info(
                "failure_event_opened",
                asset_id=asset_id,
                event_id=event.id,
                peak=upd.opened.peak,
            )
            return EventChange(event, "opened")

        stats = upd.extended or upd.closed
        event_id = self._open_ids.get(asset_id)
        if stats is None or event_id is None:
            return None
        row = await db.get(FailureEvent, event_id)
        if row is None:
            return None
        self._apply(row, stats)
        if upd.closed:
            row.status = "closed"
            row.ended_at = upd.closed.last_above_at
            self._open_ids.pop(asset_id, None)
            log.info(
                "failure_event_closed", asset_id=asset_id, event_id=row.id, n=stats.n
            )
            return EventChange(row, "closed")
        return EventChange(row, "extended")

    async def rebuild(self, db: AsyncSession, asset_id: str) -> int:
        """Recompute all events of ``asset_id`` from stored predictions.
        Returns the number of events created."""
        has_wo = await db.scalar(
            select(func.count())
            .select_from(WorkOrder)
            .where(WorkOrder.asset_id == asset_id)
        )
        if has_wo:
            raise ConflictError(
                f"Asset {asset_id} has work orders (human labels); rebuilding its events would discard them."
            )
        ids = select(FailureEvent.id).where(FailureEvent.asset_id == asset_id)
        await db.execute(
            delete(RootCauseAnalysis).where(RootCauseAnalysis.event_id.in_(ids))
        )
        await db.execute(delete(FailureEvent).where(FailureEvent.asset_id == asset_id))

        tracker = EventTracker(self.threshold, self.k_open, self.k_close)
        result = await db.stream(
            select(Prediction.timestamp, Prediction.failure_probability)
            .where(Prediction.asset_id == asset_id)
            .order_by(Prediction.timestamp)
        )
        created = 0
        current: FailureEvent | None = None
        async for ts, prob in result:
            upd = tracker.update(ts, prob)
            if upd.opened:
                current = FailureEvent(
                    asset_id=asset_id,
                    started_at=upd.opened.started_at,
                    status="open",
                    severity=severity_for(upd.opened.peak, self.critical_at),
                    peak_probability=upd.opened.peak,
                    mean_probability=upd.opened.mean,
                    n_samples=upd.opened.n,
                    threshold=self.threshold,
                )
                db.add(current)
                created += 1
            elif current is not None and (upd.extended or upd.closed):
                self._apply(current, upd.extended or upd.closed)  # type: ignore[arg-type]
                if upd.closed:
                    current.status = "closed"
                    current.ended_at = upd.closed.last_above_at
                    current = None
        # The online tracker for this asset no longer matches the rows.
        self._trackers.pop(asset_id, None)
        self._open_ids.pop(asset_id, None)
        await db.flush()
        return created
