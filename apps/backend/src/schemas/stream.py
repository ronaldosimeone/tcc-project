from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """Snapshot of all 12 sensor features for one machine cycle."""

    timestamp: datetime

    # ── Analog sensors ────────────────────────────────────────────────────
    TP2: float = Field(description="Downstream pressure [bar]")
    TP3: float = Field(description="Evaporator outlet temperature [°C]")
    H1: float = Field(description="Return pressure [%]")
    DV_pressure: float = Field(description="Differential pressure [bar]")
    Reservoirs: float = Field(description="Reservoir pressure [bar]")
    Motor_current: float = Field(description="Motor current draw [A]")
    Oil_temperature: float = Field(description="Oil temperature [°C]")

    # ── Digital (binary) sensors ──────────────────────────────────────────
    COMP: float = Field(description="Compressor state [0=off, 1=on]")
    DV_eletric: float = Field(description="Electric valve state [binary]")
    Towers: float = Field(description="Tower valve state [binary]")
    MPG: float = Field(description="Minimum pressure governor state [binary]")
    Oil_level: float = Field(description="Oil level switch [binary]")
