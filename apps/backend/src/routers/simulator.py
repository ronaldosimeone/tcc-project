"""
Simulator control endpoint (RNF-29).

PUT /simulator/mode  — switch the active simulation mode at runtime.
GET /simulator/mode  — query the current mode.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.schemas.simulator import ModeRequest, ModeResponse
from src.services.simulator import SensorSimulator, get_simulator

router = APIRouter(prefix="/simulator", tags=["simulator"])


@router.put(
    "/mode",
    response_model=ModeResponse,
    summary="Change simulator mode (RNF-29)",
)
async def set_mode(
    payload: ModeRequest,
    simulator: SensorSimulator = Depends(get_simulator),
) -> ModeResponse:
    """
    Switch the data-generation mode in real time.

    The change takes effect on the very next broadcast cycle (≤ 1 s).
    Accepted values: `NORMAL`, `DEGRADATION`, `FAILURE`.
    """
    simulator.mode = payload.mode
    return ModeResponse(
        mode=simulator.mode,
        message=f"Simulator mode set to {simulator.mode.value}.",
    )


@router.get(
    "/mode",
    response_model=ModeResponse,
    summary="Query current simulator mode",
)
async def get_mode(
    simulator: SensorSimulator = Depends(get_simulator),
) -> ModeResponse:
    """Return the currently active simulation mode."""
    return ModeResponse(
        mode=simulator.mode,
        message=f"Current simulator mode: {simulator.mode.value}.",
    )
