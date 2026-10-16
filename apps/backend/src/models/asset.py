"""
SQLAlchemy ORM model for assets — fleet registry (ISO 14224 taxonomy).

ISO 14224 organises equipment data in a hierarchy; the levels kept here are
the ones the project actually needs to group failures and KPIs:

    installation (plant / depot)  ->  system  ->  equipment unit  ->  subunit

Each simulated compressor (one MetroPT-3 replay instance) is one asset. The
primary key is the operational tag shown in the UI (e.g. ``APU-Trem-042``),
so predictions, failure events and work orders reference it directly.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.core.database import Base


class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    installation: Mapped[str] = mapped_column(String(100), nullable=False)
    system: Mapped[str] = mapped_column(String(100), nullable=False)
    equipment_unit: Mapped[str] = mapped_column(String(100), nullable=False)
    subunit: Mapped[str] = mapped_column(String(100), nullable=False)
    manufacturer_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    # 1 = low … 3 = high (drives severity escalation and fleet ordering).
    criticality: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
