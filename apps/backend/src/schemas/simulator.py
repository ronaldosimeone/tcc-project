from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from src.services.simulator import SimulatorMode


class ModeRequest(BaseModel):
    mode: SimulatorMode


class ModeResponse(BaseModel):
    mode: SimulatorMode
    message: str


class ScenarioInfo(BaseModel):
    """One real failure that the simulator can replay."""

    id: str
    label: str
    failure_type: str
    dataset: str
    start: datetime
    end: datetime
    seen_in_training: bool = Field(
        description="True: served models were trained on it (demonstration). "
        "False: after every training cut-off (honest live test)."
    )
    available: bool = Field(description="Data file present on this server.")
    note: str


class ScenarioRequest(BaseModel):
    id: str | None = Field(
        default=None, description="Scenario id; null = legacy MetroPT-3 windows."
    )
    lead_minutes: int = Field(
        default=30,
        ge=0,
        le=360,
        description="Replay starts this long before the failure.",
    )


class ScenarioStatus(BaseModel):
    mode: SimulatorMode
    scenario: ScenarioInfo | None
    step: int | None = None
    total: int | None = None
    phase: str | None = Field(default=None, description="pre_failure | failure")
