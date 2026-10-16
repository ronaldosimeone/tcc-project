"""
SQLAlchemy ORM models for the maintenance history: failure events, work
orders and root-cause analyses.

FailureEvent — one alarm EPISODE, not one prediction. Built online by
``EventTracker`` from the prediction stream with the RN-03 persistence rule:
an event opens after ``k_open`` consecutive predictions above the threshold
and closes after ``k_close`` consecutive predictions below it. This is the
same episode definition used in the offline evaluation
(apps/ml/src/event_metrics.py), so history in the UI and thesis metrics
count the same thing.

WorkOrder — the maintenance action opened from an event. When the technician
closes it with a confirmed cause (or marks it as a false alarm), the event
becomes a LABELLED example: the history turns into a dataset that feeds
re-training and the KPIs (MTBF/MTTR only count confirmed failures).

RootCauseAnalysis — immutable snapshot taken when the event opens: per-sensor
contributions (XAI) and, when available, the RAG suggestion with its
grounding report. Frozen on purpose: the explanation shown months later must
be the one that existed when the decision was made.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.core.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class FailureEvent(Base):
    __tablename__ = "failure_events"
    __table_args__ = (
        Index("ix_failure_events_asset_started", "asset_id", "started_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("assets.id"), nullable=False
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="open"
    )  # open | closed
    severity: Mapped[str] = mapped_column(
        String(16), nullable=False
    )  # alerta | critico
    peak_probability: Mapped[float] = mapped_column(Float, nullable=False)
    mean_probability: Mapped[float] = mapped_column(Float, nullable=False)
    n_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, onupdate=_now
    )


class WorkOrder(Base):
    __tablename__ = "work_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("failure_events.id"), nullable=False, unique=True
    )
    asset_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("assets.id"), nullable=False, index=True
    )
    # open -> in_progress -> closed | cancelled
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="open")
    assignee: Mapped[str | None] = mapped_column(String(100), nullable=True)
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    closed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    parts: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    # Ground truth (closes the loop): taxonomy in schemas/history.py.
    confirmed_cause: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_false_alarm: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class RootCauseAnalysis(Base):
    __tablename__ = "root_cause_analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("failure_events.id"), nullable=False, unique=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    method: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # e.g. "sensor_occlusion"
    # [{"sensor": "TP2", "contribution": 0.31}, ...] sorted by |contribution|
    sensor_contributions: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    suggestion_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    references: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    grounding: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
