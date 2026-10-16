"""
HistoryService — read side of the maintenance history + work orders.

Every number shown in the History dashboard comes from here (no mock):

* ``list_events``   — cursor pagination (started_at DESC, id DESC); filters by
                      asset, severity, status, time window.
* ``daily_counts``  — events per day split by severity (frequency chart).
* ``event_detail``  — event + frozen root cause + work order + the REAL
                      stored readings around it (−2 h … +30 min).
* ``kpis``          — reliability KPIs computed only from labelled data:
                      MTBF/MTTR/availability count confirmed failures (work
                      order closed with ``is_false_alarm = False``), never raw
                      alarms; OEE is reported as not computable (no
                      production counts), instead of being invented.
* work orders       — create from an event, validated status transitions,
                      confirmed cause (ground truth for re-training).
"""

from __future__ import annotations

import base64
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import BadRequestError, ConflictError, NotFoundError
from src.models.asset import Asset
from src.models.failure_event import FailureEvent, RootCauseAnalysis, WorkOrder
from src.models.prediction import Prediction
from src.schemas.history import (
    AssetOut,
    DailyEventCount,
    EventPage,
    FailureEventDetail,
    FailureEventOut,
    KpiReport,
    RootCauseOut,
    SensorContributionOut,
    SensorPoint,
    WorkOrderCreate,
    WorkOrderOut,
    WorkOrderUpdate,
)

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
WINDOW_BEFORE = timedelta(hours=2)
WINDOW_AFTER = timedelta(minutes=30)
MAX_WINDOW_POINTS = 500
TOP_SENSORS = 3

_TRANSITIONS: dict[str, set[str]] = {
    "open": {"in_progress", "cancelled"},
    "in_progress": {"closed", "cancelled"},
    "closed": set(),
    "cancelled": set(),
}


def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def encode_cursor(started_at: datetime, event_id: int) -> str:
    return base64.urlsafe_b64encode(
        f"{_utc(started_at).isoformat()}|{event_id}".encode()
    ).decode()


def decode_cursor(cursor: str) -> tuple[datetime, int]:
    try:
        raw = base64.urlsafe_b64decode(cursor.encode()).decode()
        ts, eid = raw.rsplit("|", 1)
        return datetime.fromisoformat(ts), int(eid)
    except (ValueError, UnicodeDecodeError) as exc:
        raise BadRequestError("Invalid pagination cursor.") from exc


class HistoryService:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ── assets ──────────────────────────────────────────────────────────

    async def list_assets(self) -> list[AssetOut]:
        rows = (
            await self._db.scalars(
                select(Asset).order_by(Asset.criticality.desc(), Asset.id)
            )
        ).all()
        return [AssetOut.model_validate(r) for r in rows]

    # ── events ──────────────────────────────────────────────────────────

    async def _to_out(self, events: list[FailureEvent]) -> list[FailureEventOut]:
        ids = [e.id for e in events]
        wos = (
            {
                w.event_id: w
                for w in (
                    await self._db.scalars(
                        select(WorkOrder).where(WorkOrder.event_id.in_(ids))
                    )
                ).all()
            }
            if ids
            else {}
        )
        rcas = (
            {
                r.event_id: r
                for r in (
                    await self._db.scalars(
                        select(RootCauseAnalysis).where(
                            RootCauseAnalysis.event_id.in_(ids)
                        )
                    )
                ).all()
            }
            if ids
            else {}
        )
        out: list[FailureEventOut] = []
        for e in events:
            wo = wos.get(e.id)
            rca = rcas.get(e.id)
            duration = (
                (_utc(e.ended_at) - _utc(e.started_at)).total_seconds() / 60
                if e.ended_at is not None
                else None
            )
            top = (
                sorted(
                    rca.sensor_contributions,
                    key=lambda c: -abs(float(c["contribution"])),
                )[:TOP_SENSORS]
                if rca
                else []
            )
            out.append(
                FailureEventOut(
                    id=e.id,
                    asset_id=e.asset_id,
                    started_at=_utc(e.started_at),
                    ended_at=_utc(e.ended_at) if e.ended_at else None,
                    status=e.status,  # type: ignore[arg-type]
                    severity=e.severity,  # type: ignore[arg-type]
                    peak_probability=e.peak_probability,
                    mean_probability=e.mean_probability,
                    n_samples=e.n_samples,
                    model_name=e.model_name,
                    threshold=e.threshold,
                    duration_minutes=duration,
                    work_order_status=wo.status if wo else None,  # type: ignore[arg-type]
                    confirmed_cause=wo.confirmed_cause if wo else None,  # type: ignore[arg-type]
                    is_false_alarm=wo.is_false_alarm if wo else None,
                    top_sensors=[
                        SensorContributionOut(
                            sensor=c["sensor"], contribution=float(c["contribution"])
                        )
                        for c in top
                    ],
                )
            )
        return out

    async def list_events(
        self,
        asset_id: str | None = None,
        severity: str | None = None,
        status: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> EventPage:
        q = select(FailureEvent)
        if asset_id:
            q = q.where(FailureEvent.asset_id == asset_id)
        if severity:
            q = q.where(FailureEvent.severity == severity)
        if status:
            q = q.where(FailureEvent.status == status)
        if start:
            q = q.where(FailureEvent.started_at >= start)
        if end:
            q = q.where(FailureEvent.started_at < end)
        if cursor:
            c_ts, c_id = decode_cursor(cursor)
            q = q.where(
                or_(
                    FailureEvent.started_at < c_ts,
                    and_(FailureEvent.started_at == c_ts, FailureEvent.id < c_id),
                )
            )
        rows = list(
            (
                await self._db.scalars(
                    q.order_by(
                        FailureEvent.started_at.desc(), FailureEvent.id.desc()
                    ).limit(limit + 1)
                )
            ).all()
        )
        has_more = len(rows) > limit
        rows = rows[:limit]
        next_cursor = (
            encode_cursor(rows[-1].started_at, rows[-1].id)
            if has_more and rows
            else None
        )
        return EventPage(items=await self._to_out(rows), next_cursor=next_cursor)

    async def daily_counts(
        self, start: datetime, end: datetime, asset_id: str | None = None
    ) -> list[DailyEventCount]:
        q = select(FailureEvent.started_at, FailureEvent.severity).where(
            FailureEvent.started_at >= start, FailureEvent.started_at < end
        )
        if asset_id:
            q = q.where(FailureEvent.asset_id == asset_id)
        counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"alerta": 0, "critico": 0}
        )
        for started_at, sev in (await self._db.execute(q)).all():
            counts[_utc(started_at).date().isoformat()][sev] += 1
        days: list[DailyEventCount] = []
        day = _utc(start).date()
        last = (_utc(end) - timedelta(microseconds=1)).date()  # `end` is exclusive
        while day <= last:
            c = counts.get(day.isoformat(), {"alerta": 0, "critico": 0})
            days.append(
                DailyEventCount(
                    date=day.isoformat(), alerta=c["alerta"], critico=c["critico"]
                )
            )
            day += timedelta(days=1)
        return days

    async def event_detail(self, event_id: int) -> FailureEventDetail:
        event = await self._db.get(FailureEvent, event_id)
        if event is None:
            raise NotFoundError(f"Failure event {event_id} not found.")
        out = (await self._to_out([event]))[0]
        rca = await self._db.scalar(
            select(RootCauseAnalysis).where(RootCauseAnalysis.event_id == event_id)
        )
        wo = await self._db.scalar(
            select(WorkOrder).where(WorkOrder.event_id == event_id)
        )

        lo = _utc(event.started_at) - WINDOW_BEFORE
        hi = (
            _utc(event.ended_at) if event.ended_at else datetime.now(timezone.utc)
        ) + WINDOW_AFTER
        rows = (
            await self._db.scalars(
                select(Prediction)
                .where(
                    Prediction.asset_id == event.asset_id,
                    Prediction.timestamp >= lo,
                    Prediction.timestamp <= hi,
                )
                .order_by(Prediction.timestamp)
            )
        ).all()
        step = max(
            1,
            len(rows) // MAX_WINDOW_POINTS
            + (1 if len(rows) % MAX_WINDOW_POINTS else 0),
        )
        window = [
            SensorPoint(
                timestamp=_utc(p.timestamp),
                failure_probability=p.failure_probability,
                values={s: float(getattr(p, s)) for s in SENSORS},
            )
            for p in rows[::step]
        ]
        return FailureEventDetail(
            event=out,
            root_cause=RootCauseOut.model_validate(rca) if rca else None,
            work_order=WorkOrderOut.model_validate(wo) if wo else None,
            sensor_window=window,
        )

    # ── KPIs ────────────────────────────────────────────────────────────

    async def kpis(
        self, start: datetime, end: datetime, asset_id: str | None = None
    ) -> KpiReport:
        span_q = select(
            Prediction.asset_id,
            func.min(Prediction.timestamp),
            func.max(Prediction.timestamp),
        ).where(Prediction.timestamp >= start, Prediction.timestamp < end)
        if asset_id:
            span_q = span_q.where(Prediction.asset_id == asset_id)
        observed_h = sum(
            (_utc(hi) - _utc(lo)).total_seconds() / 3600
            for _, lo, hi in (
                await self._db.execute(span_q.group_by(Prediction.asset_id))
            ).all()
            if lo is not None and hi is not None
        )

        ev_q = (
            select(FailureEvent, WorkOrder)
            .outerjoin(WorkOrder, WorkOrder.event_id == FailureEvent.id)
            .where(FailureEvent.started_at >= start, FailureEvent.started_at < end)
        )
        if asset_id:
            ev_q = ev_q.where(FailureEvent.asset_id == asset_id)
        pairs = (await self._db.execute(ev_q)).all()

        confirmed = [
            (e, w) for e, w in pairs if w is not None and w.is_false_alarm is False
        ]
        false_alarms = [
            (e, w) for e, w in pairs if w is not None and w.is_false_alarm is True
        ]
        unlabeled = len(pairs) - len(confirmed) - len(false_alarms)

        repair_h = [
            (_utc(w.closed_at) - _utc(w.started_at or w.opened_at)).total_seconds()
            / 3600
            for _, w in confirmed
            if w.status == "closed" and w.closed_at is not None
        ]
        mtbf = observed_h / len(confirmed) if confirmed and observed_h > 0 else None
        mttr = sum(repair_h) / len(repair_h) if repair_h else None
        labelled = len(confirmed) + len(false_alarms)
        weeks = max((_utc(end) - _utc(start)).total_seconds() / (7 * 86400), 1e-9)
        return KpiReport(
            asset_id=asset_id,
            window_start=_utc(start),
            window_end=_utc(end),
            observed_hours=observed_h,
            events_total=len(pairs),
            confirmed_failures=len(confirmed),
            false_alarms=len(false_alarms),
            unlabeled_events=unlabeled,
            alarm_precision=len(confirmed) / labelled if labelled else None,
            false_alarms_per_week=len(false_alarms) / weeks,
            mtbf_hours=mtbf,
            mttr_hours=mttr,
            availability=(
                mtbf / (mtbf + mttr) if mtbf is not None and mttr is not None else None
            ),
        )

    # ── work orders ─────────────────────────────────────────────────────

    async def create_work_order(self, payload: WorkOrderCreate) -> WorkOrderOut:
        event = await self._db.get(FailureEvent, payload.event_id)
        if event is None:
            raise NotFoundError(f"Failure event {payload.event_id} not found.")
        if await self._db.scalar(
            select(WorkOrder.id).where(WorkOrder.event_id == event.id)
        ):
            raise ConflictError(f"Event {event.id} already has a work order.")
        wo = WorkOrder(
            event_id=event.id,
            asset_id=event.asset_id,
            status="open",
            assignee=payload.assignee,
            notes=payload.notes,
            parts=[],
            opened_at=datetime.now(timezone.utc),
        )
        self._db.add(wo)
        await self._db.commit()
        await self._db.refresh(wo)
        return WorkOrderOut.model_validate(wo)

    async def update_work_order(
        self, wo_id: int, payload: WorkOrderUpdate
    ) -> WorkOrderOut:
        wo = await self._db.get(WorkOrder, wo_id)
        if wo is None:
            raise NotFoundError(f"Work order {wo_id} not found.")
        if wo.status in {"closed", "cancelled"}:
            raise ConflictError(
                f"Work order {wo_id} is {wo.status} and can no longer change."
            )
        now = datetime.now(timezone.utc)

        if payload.confirmed_cause is not None and payload.is_false_alarm:
            raise ConflictError("A false alarm cannot have a confirmed failure cause.")
        if payload.assignee is not None:
            wo.assignee = payload.assignee
        if payload.parts is not None:
            wo.parts = [p.model_dump() for p in payload.parts]
        if payload.notes is not None:
            wo.notes = payload.notes
        if payload.confirmed_cause is not None:
            wo.confirmed_cause = payload.confirmed_cause
            wo.is_false_alarm = False
        if payload.is_false_alarm is not None:
            wo.is_false_alarm = payload.is_false_alarm
            if payload.is_false_alarm:
                wo.confirmed_cause = None

        if payload.status is not None and payload.status != wo.status:
            if payload.status not in _TRANSITIONS[wo.status]:
                raise ConflictError(
                    f"Invalid transition {wo.status} → {payload.status}."
                )
            if payload.status == "closed" and wo.is_false_alarm is None:
                raise ConflictError(
                    "Closing requires a verdict: set confirmed_cause (real failure) or is_false_alarm=true."
                )
            wo.status = payload.status
            if payload.status == "in_progress":
                wo.started_at = now
            if payload.status in {"closed", "cancelled"}:
                wo.closed_at = now

        await self._db.commit()
        await self._db.refresh(wo)
        return WorkOrderOut.model_validate(wo)

    async def list_work_orders(
        self, asset_id: str | None = None, status: str | None = None
    ) -> list[WorkOrderOut]:
        q = select(WorkOrder)
        if asset_id:
            q = q.where(WorkOrder.asset_id == asset_id)
        if status:
            q = q.where(WorkOrder.status == status)
        rows = (await self._db.scalars(q.order_by(WorkOrder.opened_at.desc()))).all()
        return [WorkOrderOut.model_validate(r) for r in rows]
