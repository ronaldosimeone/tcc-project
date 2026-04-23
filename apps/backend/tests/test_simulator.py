"""
Tests for SensorSimulator (RF-13) and PUT/GET /simulator/mode (RNF-29).

Coverage matrix
---------------
SimulatorMode      Enum has exactly the 3 required values.
SensorSimulator    Default mode, mode setter, step counter reset, return type,
                   all 12 sensor fields present, statistical profile per mode,
                   drift mechanics in DEGRADATION.
Endpoint PUT       200 response, mode persisted, mode change reflected in GET,
                   invalid mode → 422 Unprocessable Entity.
Endpoint GET       200 response, returns current mode.
Integration        Mode change propagates to SensorStreamService readings.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.routers.simulator import router as simulator_router
from src.schemas.stream import SensorReading
from src.services.simulator import (
    SimulatorMode,
    SensorSimulator,
    get_simulator,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sim() -> SensorSimulator:
    """Fresh, isolated SensorSimulator for every test."""
    return SensorSimulator()


@pytest.fixture()
def app(sim: SensorSimulator) -> FastAPI:
    """Minimal FastAPI with the simulator router; singleton overridden."""
    application = FastAPI()
    application.include_router(simulator_router)
    application.dependency_overrides[get_simulator] = lambda: sim
    return application


@pytest.fixture()
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


# ===========================================================================
# SimulatorMode enum
# ===========================================================================


class TestSimulatorMode:

    def test_has_normal(self) -> None:
        assert SimulatorMode.NORMAL == "NORMAL"

    def test_has_degradation(self) -> None:
        assert SimulatorMode.DEGRADATION == "DEGRADATION"

    def test_has_failure(self) -> None:
        assert SimulatorMode.FAILURE == "FAILURE"

    def test_exactly_three_values(self) -> None:
        assert len(SimulatorMode) == 3


# ===========================================================================
# SensorSimulator — unit tests
# ===========================================================================


class TestSensorSimulator:

    # ── Initialisation ────────────────────────────────────────────────────

    def test_default_mode_is_normal(self, sim: SensorSimulator) -> None:
        assert sim.mode == SimulatorMode.NORMAL

    def test_custom_initial_mode(self) -> None:
        s = SensorSimulator(mode=SimulatorMode.FAILURE)
        assert s.mode == SimulatorMode.FAILURE

    def test_initial_step_is_zero(self, sim: SensorSimulator) -> None:
        assert sim._step == 0

    # ── Mode setter ───────────────────────────────────────────────────────

    def test_mode_setter_changes_mode(self, sim: SensorSimulator) -> None:
        sim.mode = SimulatorMode.DEGRADATION
        assert sim.mode == SimulatorMode.DEGRADATION

    def test_mode_setter_resets_step(self, sim: SensorSimulator) -> None:
        sim._step = 150
        sim.mode = SimulatorMode.FAILURE
        assert sim._step == 0

    def test_mode_setter_no_reset_when_same_mode(self, sim: SensorSimulator) -> None:
        sim._step = 50
        sim.mode = SimulatorMode.NORMAL  # same mode — must NOT reset
        assert sim._step == 50

    # ── generate_reading ──────────────────────────────────────────────────

    def test_returns_sensor_reading(self, sim: SensorSimulator) -> None:
        assert isinstance(sim.generate_reading(), SensorReading)

    def test_step_increments_on_each_call(self, sim: SensorSimulator) -> None:
        sim.generate_reading()
        sim.generate_reading()
        assert sim._step == 2

    def test_all_12_sensor_fields_present(self, sim: SensorSimulator) -> None:
        reading = sim.generate_reading()
        expected = {
            "TP2",
            "TP3",
            "H1",
            "DV_pressure",
            "Reservoirs",
            "Motor_current",
            "Oil_temperature",
            "COMP",
            "DV_eletric",
            "Towers",
            "MPG",
            "Oil_level",
        }
        assert expected.issubset(reading.model_dump().keys())

    def test_all_fields_are_float(self, sim: SensorSimulator) -> None:
        reading = sim.generate_reading()
        data = reading.model_dump()
        data.pop("timestamp")
        assert all(isinstance(v, float) for v in data.values())

    def test_binary_sensors_are_0_or_1_in_normal(self, sim: SensorSimulator) -> None:
        binary = {"COMP", "DV_eletric", "Towers", "MPG", "Oil_level"}
        for _ in range(30):
            r = sim.generate_reading()
            for f in binary:
                assert getattr(r, f) in (0.0, 1.0)

    # ── NORMAL mode statistical profile ──────────────────────────────────

    def test_normal_motor_current_around_nominal(self, sim: SensorSimulator) -> None:
        """100-sample mean must be within ±0.5 A of nominal (2.80 A)."""
        readings = [sim.generate_reading().Motor_current for _ in range(100)]
        assert abs(np.mean(readings) - 2.80) < 0.5

    def test_normal_oil_temperature_around_nominal(self, sim: SensorSimulator) -> None:
        readings = [sim.generate_reading().Oil_temperature for _ in range(100)]
        assert abs(np.mean(readings) - 57.0) < 3.0

    def test_normal_tp2_around_nominal(self, sim: SensorSimulator) -> None:
        readings = [sim.generate_reading().TP2 for _ in range(100)]
        assert abs(np.mean(readings) - 5.90) < 0.5

    # ── FAILURE mode statistical profile ─────────────────────────────────

    def test_failure_motor_current_far_above_normal(self) -> None:
        """FAILURE motor current mean (≈6.8 A) must exceed NORMAL mean (≈2.8 A) by > 2 A."""
        fail = SensorSimulator(mode=SimulatorMode.FAILURE)
        readings = [fail.generate_reading().Motor_current for _ in range(100)]
        assert np.mean(readings) > 5.0

    def test_failure_oil_temperature_critical(self) -> None:
        fail = SensorSimulator(mode=SimulatorMode.FAILURE)
        readings = [fail.generate_reading().Oil_temperature for _ in range(100)]
        assert np.mean(readings) > 70.0

    def test_failure_tp2_pressure_low(self) -> None:
        fail = SensorSimulator(mode=SimulatorMode.FAILURE)
        readings = [fail.generate_reading().TP2 for _ in range(100)]
        assert np.mean(readings) < 5.0

    # ── DEGRADATION mode drift mechanics ─────────────────────────────────

    def test_degradation_motor_current_higher_than_normal_at_full_drift(self) -> None:
        """At drift = 1.0 (step 300) motor current mean must exceed normal mean by > 2 A."""
        normal_sim = SensorSimulator(mode=SimulatorMode.NORMAL)
        deg_sim = SensorSimulator(mode=SimulatorMode.DEGRADATION)
        deg_sim._step = 299  # next generate_reading → step=300, drift=1.0

        normal_mean = np.mean(
            [normal_sim.generate_reading().Motor_current for _ in range(100)]
        )
        deg_mean = np.mean(
            [deg_sim.generate_reading().Motor_current for _ in range(100)]
        )
        assert deg_mean - normal_mean > 2.0

    def test_degradation_oil_temp_rises_with_drift(self) -> None:
        s = SensorSimulator(mode=SimulatorMode.DEGRADATION)

        s._step = 0
        early = np.mean([s.generate_reading().Oil_temperature for _ in range(50)])

        s._step = 250
        late = np.mean([s.generate_reading().Oil_temperature for _ in range(50)])

        assert late > early + 5.0

    def test_degradation_pressure_drops_with_drift(self) -> None:
        s = SensorSimulator(mode=SimulatorMode.DEGRADATION)

        s._step = 0
        early = np.mean([s.generate_reading().TP2 for _ in range(50)])

        s._step = 250
        late = np.mean([s.generate_reading().TP2 for _ in range(50)])

        assert late < early - 0.5

    def test_degradation_step_increases_each_call(self) -> None:
        s = SensorSimulator(mode=SimulatorMode.DEGRADATION)
        for i in range(1, 6):
            s.generate_reading()
            assert s._step == i

    # ── Mode transitions ──────────────────────────────────────────────────

    def test_transition_normal_to_failure_then_back(self) -> None:
        s = SensorSimulator()
        s.mode = SimulatorMode.FAILURE
        assert s.mode == SimulatorMode.FAILURE
        assert s._step == 0

        s.mode = SimulatorMode.NORMAL
        assert s.mode == SimulatorMode.NORMAL
        assert s._step == 0

    def test_reading_changes_statistical_profile_after_mode_switch(self) -> None:
        """Motor_current should be higher in FAILURE than NORMAL (100-sample t-test proxy)."""
        s = SensorSimulator(mode=SimulatorMode.NORMAL)
        normal_mean = np.mean([s.generate_reading().Motor_current for _ in range(100)])

        s.mode = SimulatorMode.FAILURE
        failure_mean = np.mean([s.generate_reading().Motor_current for _ in range(100)])

        assert failure_mean > normal_mean + 2.0


# ===========================================================================
# Endpoint PUT /simulator/mode
# ===========================================================================


class TestPutSimulatorMode:

    def test_returns_200(self, client: TestClient) -> None:
        response = client.put("/simulator/mode", json={"mode": "NORMAL"})
        assert response.status_code == 200

    def test_response_contains_mode(self, client: TestClient) -> None:
        response = client.put("/simulator/mode", json={"mode": "DEGRADATION"})
        assert response.json()["mode"] == "DEGRADATION"

    def test_response_contains_message(self, client: TestClient) -> None:
        response = client.put("/simulator/mode", json={"mode": "FAILURE"})
        assert "message" in response.json()

    def test_mode_persisted_on_simulator(
        self, client: TestClient, sim: SensorSimulator
    ) -> None:
        client.put("/simulator/mode", json={"mode": "FAILURE"})
        assert sim.mode == SimulatorMode.FAILURE

    def test_invalid_mode_returns_422(self, client: TestClient) -> None:
        response = client.put("/simulator/mode", json={"mode": "EXPLODE"})
        assert response.status_code == 422

    def test_missing_body_returns_422(self, client: TestClient) -> None:
        response = client.put("/simulator/mode")
        assert response.status_code == 422

    def test_all_three_modes_accepted(self, client: TestClient) -> None:
        for mode in ("NORMAL", "DEGRADATION", "FAILURE"):
            r = client.put("/simulator/mode", json={"mode": mode})
            assert r.status_code == 200, f"Expected 200 for mode={mode}"

    def test_step_reset_after_mode_change(
        self, client: TestClient, sim: SensorSimulator
    ) -> None:
        sim._step = 99
        client.put("/simulator/mode", json={"mode": "FAILURE"})
        assert sim._step == 0


# ===========================================================================
# Endpoint GET /simulator/mode
# ===========================================================================


class TestGetSimulatorMode:

    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/simulator/mode")
        assert response.status_code == 200

    def test_returns_default_normal_mode(
        self, client: TestClient, sim: SensorSimulator
    ) -> None:
        assert sim.mode == SimulatorMode.NORMAL
        response = client.get("/simulator/mode")
        assert response.json()["mode"] == "NORMAL"

    def test_reflects_mode_set_by_put(self, client: TestClient) -> None:
        client.put("/simulator/mode", json={"mode": "DEGRADATION"})
        response = client.get("/simulator/mode")
        assert response.json()["mode"] == "DEGRADATION"

    def test_response_schema(self, client: TestClient) -> None:
        response = client.get("/simulator/mode")
        body = response.json()
        assert "mode" in body
        assert "message" in body


# ===========================================================================
# Integration: mode change propagates through SensorStreamService
# ===========================================================================


class TestModeIntegration:

    def test_stream_service_uses_simulator_mode(self) -> None:
        """
        FAILURE-mode readings (Motor_current ≈ 6.8 A) must differ from
        NORMAL-mode readings (Motor_current ≈ 2.8 A) by > 2 A on average.
        """
        from src.services.sensor_stream_service import SensorStreamService

        sim = SensorSimulator(mode=SimulatorMode.NORMAL)
        service = SensorStreamService(simulator=sim)

        normal_readings = [
            service._generate_reading().Motor_current for _ in range(100)
        ]

        sim.mode = SimulatorMode.FAILURE
        failure_readings = [
            service._generate_reading().Motor_current for _ in range(100)
        ]

        assert np.mean(failure_readings) - np.mean(normal_readings) > 2.0
