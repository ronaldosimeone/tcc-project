"""
Simulator control endpoint (RNF-29).

PUT /simulator/mode  — switch the active simulation mode at runtime.
GET /simulator/mode  — query the current mode.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.schemas.simulator import (
    ModeRequest,
    ModeResponse,
    ScenarioInfo,
    ScenarioRequest,
    ScenarioStatus,
)
from src.services import failure_scenarios as fs
from src.services.feature_buffer import SensorBuffer, get_sensor_buffer
from src.services.protocols import SensorSimulatorProtocol
from src.services.simulator import SensorSimulator, get_simulator

router = APIRouter(prefix="/simulator", tags=["simulator"])


@router.put(
    "/mode",
    response_model=ModeResponse,
    summary="Change simulator mode (RNF-29)",
)
async def set_mode(
    payload: ModeRequest,
    simulator: SensorSimulatorProtocol = Depends(get_simulator),
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
    simulator: SensorSimulatorProtocol = Depends(get_simulator),
) -> ModeResponse:
    """Return the currently active simulation mode."""
    return ModeResponse(
        mode=simulator.mode,
        message=f"Current simulator mode: {simulator.mode.value}.",
    )


def _info(scenario: fs.FailureScenario) -> ScenarioInfo:
    return ScenarioInfo(
        id=scenario.id,
        label=scenario.label,
        failure_type=scenario.failure_type,
        dataset=scenario.dataset,
        start=scenario.start,
        end=scenario.end,
        seen_in_training=scenario.seen_in_training,
        available=fs.available(scenario),
        note=scenario.note,
    )


@router.get(
    "/scenarios",
    response_model=list[ScenarioInfo],
    summary="Real failure scenarios that can be replayed",
)
async def list_scenarios() -> list[ScenarioInfo]:
    return [_info(s) for s in fs.SCENARIOS]


def _status(simulator: SensorSimulator) -> ScenarioStatus:
    st = simulator.scenario_status()
    if st is None:
        return ScenarioStatus(mode=simulator.mode, scenario=None)
    return ScenarioStatus(
        mode=simulator.mode,
        scenario=_info(fs.BY_ID[str(st["id"])]),
        step=int(str(st["step"])),
        total=int(str(st["total"])),
        phase=str(st["phase"]),
    )


@router.put(
    "/scenario",
    response_model=ScenarioStatus,
    summary="Replay a real failure (switches to FAILURE mode)",
    responses={
        400: {"description": "Data file missing"},
        404: {"description": "Unknown scenario"},
    },
)
async def set_scenario(
    payload: ScenarioRequest,
    simulator: SensorSimulator = Depends(get_simulator),
    buffer: SensorBuffer = Depends(get_sensor_buffer),
) -> ScenarioStatus:
    fs.start_scenario(simulator, buffer, payload.id, payload.lead_minutes)
    return _status(simulator)


@router.get(
    "/scenario",
    response_model=ScenarioStatus,
    summary="Current scenario and replay progress",
)
async def get_scenario(
    simulator: SensorSimulator = Depends(get_simulator),
) -> ScenarioStatus:
    return _status(simulator)
