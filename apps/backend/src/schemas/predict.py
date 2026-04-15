"""
Pydantic v2 DTOs for POST /predict.

PredictRequest  — 14 raw MetroPT-3 sensor readings from a single acquisition cycle.
PredictResponse — [RF-05] predicted_class, failure_probability, timestamp.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    """
    Raw sensor snapshot from a MetroPT-3 compressor acquisition cycle.

    All analogue values are in their native engineering units (bar, °C, A).
    Digital / switch columns are represented as 0.0 or 1.0.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "TP2": 5.02,
                "TP3": 9.21,
                "H1": 8.97,
                "DV_pressure": 2.10,
                "Reservoirs": 8.85,
                "Oil_temperature": 72.3,
                "Motor_current": 4.5,
                "COMP": 1.0,
                "DV_eletric": 0.0,
                "Towers": 1.0,
                "MPG": 1.0,
                "Pressure_switch": 1.0,
                "Oil_level": 1.0,
                "Caudal_impulses": 1.0,
            }
        }
    )

    # ── Analogue pressure / temperature / current sensors ─────────────────
    TP2: float = Field(
        ..., description="Delivery pressure downstream of compressor head (bar)"
    )
    TP3: float = Field(..., description="Pressure at pneumatic panel inlet (bar)")
    H1: float = Field(..., description="Pressure at consumer circuit (bar)")
    DV_pressure: float = Field(
        ..., description="Differential pressure across air dryer (bar)"
    )
    Reservoirs: float = Field(..., description="Air reservoir pressure (bar)")
    Oil_temperature: float = Field(..., description="Compressor oil temperature (°C)")
    Motor_current: float = Field(..., description="Electric motor current draw (A)")

    # ── Digital / switch sensors (0.0 = OFF, 1.0 = ON) ───────────────────
    COMP: float = Field(..., ge=0.0, le=1.0, description="Compressor active state")
    DV_eletric: float = Field(
        ..., ge=0.0, le=1.0, description="Electric discharge valve state"
    )
    Towers: float = Field(
        ..., ge=0.0, le=1.0, description="Desiccant tower switch state"
    )
    MPG: float = Field(..., ge=0.0, le=1.0, description="MPG valve state")
    Pressure_switch: float = Field(
        ..., ge=0.0, le=1.0, description="High-pressure switch state"
    )
    Oil_level: float = Field(..., ge=0.0, le=1.0, description="Oil-level switch state")
    Caudal_impulses: float = Field(
        ..., ge=0.0, le=1.0, description="Flow impulse counter state"
    )


class PredictResponse(BaseModel):
    """
    [RF-05] Fault prediction result.

    predicted_class    : 0 = Normal operation | 1 = Fault detected.
    failure_probability: Model confidence for class 1, range [0.0, 1.0].
    timestamp          : ISO 8601 UTC timestamp of the prediction.
    """

    predicted_class: int = Field(description="Predicted label: 0 (normal) or 1 (fault)")
    failure_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability assigned to the fault class (class 1)",
    )
    timestamp: str = Field(description="ISO 8601 UTC timestamp of inference")
