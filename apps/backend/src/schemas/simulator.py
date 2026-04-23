from __future__ import annotations

from pydantic import BaseModel

from src.services.simulator import SimulatorMode


class ModeRequest(BaseModel):
    mode: SimulatorMode


class ModeResponse(BaseModel):
    mode: SimulatorMode
    message: str
