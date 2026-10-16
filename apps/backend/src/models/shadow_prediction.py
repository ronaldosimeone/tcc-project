"""
SQLAlchemy ORM model for shadow_predictions — champion vs challenger.

Shadow deployment: the challenger model scores the SAME feature window as
the champion for every reading of the main asset, but never acts (no
alerts, no events). Each pair is stored so agreement can be analysed online
(``GET /v1/models/shadow``) and offline, before any promotion (RN-05).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.core.database import Base


class ShadowPrediction(Base):
    __tablename__ = "shadow_predictions"
    __table_args__ = (Index("ix_shadow_predictions_ts", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    asset_id: Mapped[str] = mapped_column(String(64), nullable=False)
    champion_model: Mapped[str] = mapped_column(String(64), nullable=False)
    challenger_model: Mapped[str] = mapped_column(String(64), nullable=False)
    champion_probability: Mapped[float] = mapped_column(Float, nullable=False)
    challenger_probability: Mapped[float] = mapped_column(Float, nullable=False)
    champion_class: Mapped[int] = mapped_column(Integer, nullable=False)
    challenger_class: Mapped[int] = mapped_column(Integer, nullable=False)
