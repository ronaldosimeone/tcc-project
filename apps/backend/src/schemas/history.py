"""
Pydantic v2 DTOs for the fleet / maintenance-history API.

Replaces the frontend mock (apps/frontend/lib/history-mock.ts) with real
data: assets, failure events (RN-03 episodes), work orders with confirmed
cause (ground truth), root-cause snapshots and reliability KPIs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["alerta", "critico"]
EventStatus = Literal["open", "closed"]
WorkOrderStatus = Literal["open", "in_progress", "closed", "cancelled"]
# Failure-cause taxonomy for the compressor (subset of ISO 14224 failure
# mechanisms that the MetroPT-3 APU can exhibit).
FailureCause = Literal[
    "air_leak",
    "oil_system",
    "air_drier",
    "valve",
    "electrical_motor",
    "sensor_fault",
    "other",
]


class AssetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    installation: str
    system: str
    equipment_unit: str
    subunit: str
    manufacturer_model: str | None
    criticality: int


class SensorContributionOut(BaseModel):
    sensor: str
    contribution: float


class RootCauseOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    method: str
    created_at: datetime
    sensor_contributions: list[SensorContributionOut]
    suggestion_markdown: str | None
    references: list[dict[str, object]]
    grounding: dict[str, object] | None
    llm_model: str | None


class WorkOrderOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    event_id: int
    asset_id: str
    status: WorkOrderStatus
    assignee: str | None
    opened_at: datetime
    started_at: datetime | None
    closed_at: datetime | None
    parts: list[dict[str, object]]
    confirmed_cause: FailureCause | None
    is_false_alarm: bool | None
    notes: str | None


class FailureEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    asset_id: str
    started_at: datetime
    ended_at: datetime | None
    status: EventStatus
    severity: Severity
    peak_probability: float
    mean_probability: float
    n_samples: int
    model_name: str | None
    threshold: float
    duration_minutes: float | None = Field(description="None while the event is open.")
    work_order_status: WorkOrderStatus | None = None
    confirmed_cause: FailureCause | None = None
    is_false_alarm: bool | None = None
    top_sensors: list[SensorContributionOut] = Field(default_factory=list)


class EventPage(BaseModel):
    items: list[FailureEventOut]
    next_cursor: str | None = Field(
        description="Opaque cursor for the next page; None on the last page."
    )


class SensorPoint(BaseModel):
    timestamp: datetime
    failure_probability: float
    values: dict[str, float]


class FailureEventDetail(BaseModel):
    event: FailureEventOut
    root_cause: RootCauseOut | None
    work_order: WorkOrderOut | None
    sensor_window: list[SensorPoint] = Field(
        description="Real stored readings from 2 h before the event to 30 min after it (≤ 500 points)."
    )


class DailyEventCount(BaseModel):
    date: str
    alerta: int
    critico: int


class KpiReport(BaseModel):
    asset_id: str | None
    window_start: datetime
    window_end: datetime
    observed_hours: float
    events_total: int
    confirmed_failures: int
    false_alarms: int
    unlabeled_events: int
    alarm_precision: float | None = Field(
        description="confirmed / (confirmed + false alarms); labelled events only."
    )
    false_alarms_per_week: float
    mtbf_hours: float | None = Field(description="observed hours / confirmed failures.")
    mttr_hours: float | None = Field(
        description="Mean (closed − started|opened) of closed work orders of confirmed failures."
    )
    availability: float | None = Field(description="MTBF / (MTBF + MTTR).")
    oee: float | None = None
    oee_note: str = (
        "OEE needs production counts (performance and quality rates), which a compressor "
        "APU does not record; only availability is computed from data."
    )


class WorkOrderCreate(BaseModel):
    event_id: int
    assignee: str | None = Field(default=None, max_length=100)
    notes: str | None = Field(default=None, max_length=5000)


class PartUsed(BaseModel):
    part: str = Field(min_length=1, max_length=200)
    quantity: int = Field(ge=1, le=1000)


class WorkOrderUpdate(BaseModel):
    """Partial update. Status transitions are validated by the service:
    open → in_progress → closed, and open|in_progress → cancelled."""

    status: WorkOrderStatus | None = None
    assignee: str | None = Field(default=None, max_length=100)
    parts: list[PartUsed] | None = Field(default=None, max_length=50)
    confirmed_cause: FailureCause | None = None
    is_false_alarm: bool | None = None
    notes: str | None = Field(default=None, max_length=5000)
