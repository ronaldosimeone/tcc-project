"""
SQLAlchemy ORM model for the predictions table — RF-09.

Every call to POST /predict that produces a successful inference is stored
here by PredictionService.save_prediction().  The timestamps column is
indexed to allow efficient newest-first ordering for the history endpoint.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer
from sqlalchemy.orm import Mapped, mapped_column

from src.core.database import Base


class Prediction(Base):
    """
    Persisted record of a single MetroPT-3 inference cycle — RF-09.

    Columns
    -------
    id                : Auto-incremented surrogate key.
    timestamp         : UTC timestamp of the inference (from PredictResponse).
    TP2 … Oil_level   : The 12 raw sensor inputs sent by the client.
    predicted_class   : Binary label returned by the Random Forest (0 or 1).
    failure_probability: Model confidence for the fault class (class 1).
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # UTC timestamp — indexed for ORDER BY timestamp DESC queries
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    # ── Analogue sensors ──────────────────────────────────────────────────
    TP2: Mapped[float] = mapped_column(Float, nullable=False)
    TP3: Mapped[float] = mapped_column(Float, nullable=False)
    H1: Mapped[float] = mapped_column(Float, nullable=False)
    DV_pressure: Mapped[float] = mapped_column(Float, nullable=False)
    Reservoirs: Mapped[float] = mapped_column(Float, nullable=False)
    Motor_current: Mapped[float] = mapped_column(Float, nullable=False)
    Oil_temperature: Mapped[float] = mapped_column(Float, nullable=False)

    # ── Digital / switch sensors ──────────────────────────────────────────
    COMP: Mapped[float] = mapped_column(Float, nullable=False)
    DV_eletric: Mapped[float] = mapped_column(Float, nullable=False)
    Towers: Mapped[float] = mapped_column(Float, nullable=False)
    MPG: Mapped[float] = mapped_column(Float, nullable=False)
    Oil_level: Mapped[float] = mapped_column(Float, nullable=False)

    # ── ML outputs ────────────────────────────────────────────────────────
    predicted_class: Mapped[int] = mapped_column(Integer, nullable=False)
    failure_probability: Mapped[float] = mapped_column(Float, nullable=False)
