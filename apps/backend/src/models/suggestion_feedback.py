"""
SQLAlchemy ORM model for suggestion_feedback — RF-38 (human-in-the-loop).

Each row is a technician's verdict on one maintenance plan: whether the plan
was useful and, per cited manual chunk (file + page), whether that chunk was
actually relevant. The per-reference votes feed the retrieval re-ranking
(``SuggestionFeedbackService.reference_priors``) and become ground truth for
the RAG benchmark (RF-36) — feedback collected in operation is the only
scalable source of expert labels.

No user accounts exist in the project (see models/alert_settings.py), so
feedback is anonymous; `equipment_name` scopes the priors.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.core.database import Base


class SuggestionFeedback(Base):
    __tablename__ = "suggestion_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    equipment_name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    symptom_description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    query: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    grounding_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    useful: Mapped[bool] = mapped_column(Boolean, nullable=False)
    # [{"file_name": str, "page": int, "helpful": bool}, ...]
    reference_votes: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    comment: Mapped[str | None] = mapped_column(String(1000), nullable=True)
