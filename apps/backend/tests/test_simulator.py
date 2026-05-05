"""
Tests for SensorSimulator (RF-13) and PUT/GET /simulator/mode (RNF-29).

Coverage matrix
---------------
SimulatorMode      Enum has exactly the 3 required values.
SensorSimulator    Default mode, mode setter, step counter reset, return type,
                   all 12 sensor fields present, real-data range sanity,
                   relative statistical separation between NORMAL and FAILURE,
                   drift mechanics in DEGRADATION.
Endpoint PUT       200 response, mode persisted, mode change reflected in GET,
                   invalid mode → 422 Unprocessable Entity.
Endpoint GET       200 response, returns current mode.
Integration        Mode change propagates to SensorStreamService readings.

Data-dependency note
--------------------
Tests that require real sensor readings are skipped when the MetroPT-3
parquet is absent (CI environments without the full dataset).  Structural
and schema tests run unconditionally.
"""

from __future__ import annotations

from pathlib import Path

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

_PARQUET = (
    Path(__file__).resolve().parents[3]
    / "ml" / "data" / "processed" / "metropt3.parquet"
)


@pytest.fixture(scope="session")
def parquet_path() -> Path:
    if not _PARQUET.exists():
        pytest.skip("metropt3.parquet not found — skipping data-dependent tests")
    return _PARQUET


@pytest.fixture()
def sim(parquet_path: Path) -> SensorSimulator:
    """Fresh, isolated SensorSimulator backed by real MetroPT-3 data."""
    return SensorSimulator(parquet_path=parquet_path)


@pytest.fixture()
def sim_failure(parquet_path: Path) -> SensorSimulator:
    return SensorSimulator(mode=SimulatorMode.FAILURE, parquet_path=parquet_path)


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
# SimulatorMode enum — no parquet needed
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

    def test_custom_initial_mode(self, parquet_path: Path) -> None:
        s = SensorSimulator(mode=SimulatorMode.FAILURE, parquet_path=parquet_path)
        assert s.mode == SimulatorMode.FAILURE

    def test_initial_step_is_zero(self, sim: SensorSimulator) -> None:
        assert sim._step == 0

    def test_data_partitions_non_empty(self, sim: SensorSimulator) -> None:
        """Both normal and failure arrays must have at least one row."""
        assert len(sim._normal) > 0
        assert len(sim._failure) > 0

    def test_normal_larger_than_failure(self, sim: SensorSimulator) -> None:
        """The MetroPT-3 dataset is predominantly healthy — normal rows > failure rows."""
        assert len(sim._normal) > len(sim._failure)

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
            "TP2", "TP3", "H1", "DV_pressure", "Reservoirs",
            "Motor_current", "Oil_temperature",
            "COMP", "DV_eletric", "Towers", "MPG", "Oil_level",
        }
        assert expected.issubset(reading.model_dump().keys())

    def test_all_fields_are_float(self, sim: SensorSimulator) -> None:
        reading = sim.generate_reading()
        data = reading.model_dump()
        data.pop("timestamp")
        assert all(isinstance(v, float) for v in data.values())

    def test_binary_sensors_are_0_or_1_in_normal(self, sim: SensorSimulator) -> None:
        """Real MetroPT-3 binary sensors are stored as 0.0 / 1.0 — must be preserved."""
        binary = {"COMP", "DV_eletric", "Towers", "MPG", "Oil_level"}
        for _ in range(30):
            r = sim.generate_reading()
            for f in binary:
                assert getattr(r, f) in (0.0, 1.0)

    # ── NORMAL mode — physical range checks (real data) ───────────────────

    def test_normal_tp2_positive(self, sim: SensorSimulator) -> None:
        """Downstream pressure must be non-negative in healthy operation."""
        readings = [sim.generate_reading().TP2 for _ in range(50)]
        assert all(v >= 0.0 for v in readings)

    def test_normal_oil_temperature_plausible(self, sim: SensorSimulator) -> None:
        """Oil temperature in healthy operation stays within 20–120 °C."""
        readings = [sim.generate_reading().Oil_temperature for _ in range(50)]
        assert all(20.0 <= v <= 120.0 for v in readings)

    def test_normal_motor_current_positive(self, sim: SensorSimulator) -> None:
        readings = [sim.generate_reading().Motor_current for _ in range(50)]
        assert all(v >= 0.0 for v in readings)

    # ── FAILURE vs NORMAL statistical separation ──────────────────────────

    def test_failure_motor_current_differs_from_normal(
        self, sim: SensorSimulator, sim_failure: SensorSimulator
    ) -> None:
        """
        Air-leak periods show anomalous motor current.
        The means must differ by at least 0.5 A (conservative — avoids fragility
        from dataset-specific absolute values).
        """
        normal_mean = np.mean([sim.generate_reading().Motor_current for _ in range(200)])
        fail_mean = np.mean([sim_failure.generate_reading().Motor_current for _ in range(200)])
        assert abs(fail_mean - normal_mean) > 0.5

    def test_failure_tp2_differs_from_normal(
        self, sim: SensorSimulator, sim_failure: SensorSimulator
    ) -> None:
        normal_mean = np.mean([sim.generate_reading().TP2 for _ in range(200)])
        fail_mean = np.mean([sim_failure.generate_reading().TP2 for _ in range(200)])
        # Air leaks typically drop downstream pressure — means must differ.
        assert abs(fail_mean - normal_mean) > 0.2

    # ── DEGRADATION mode drift mechanics ─────────────────────────────────

    def test_degradation_step_increases_each_call(self, sim: SensorSimulator) -> None:
        sim.mode = SimulatorMode.DEGRADATION
        for i in range(1, 6):
            sim.generate_reading()
            assert sim._step == i

    def test_degradation_at_drift_zero_close_to_normal(
        self, sim: SensorSimulator, parquet_path: Path
    ) -> None:
        """
        At drift=0 (step=0), the degradation reading is identical to the
        corresponding normal row — no failure signal injected yet.
        """
        normal_clone = SensorSimulator(parquet_path=parquet_path)
        deg = SensorSimulator(mode=SimulatorMode.DEGRADATION, parquet_path=parquet_path)

        # Both start at index 0; drift = 1/300 ≈ 0 on the first tick.
        normal_val = normal_clone.generate_reading().Motor_current
        deg_val = deg.generate_reading().Motor_current

        # With drift ≈ 0.003, the blended value must be very close to normal.
        assert abs(deg_val - normal_val) < abs(normal_val) * 0.05  # within 5 %

    def test_degradation_at_full_drift_close_to_failure(
        self, parquet_path: Path
    ) -> None:
        """
        At drift=1.0 (step=300), the reading equals the failure-partition row.
        """
        fail_sim = SensorSimulator(mode=SimulatorMode.FAILURE, parquet_path=parquet_path)
        deg_sim = SensorSimulator(mode=SimulatorMode.DEGRADATION, parquet_path=parquet_path)
        deg_sim._step = 299  # next generate_reading → step=300, drift=1.0

        # Both pointers start at 0 → first row must match exactly at drift=1.
        fail_val = fail_sim.generate_reading().Motor_current
        deg_val = deg_sim.generate_reading().Motor_current
        assert abs(deg_val - fail_val) < 1e-4

    def test_degradation_blend_is_monotonic_over_time(
        self, parquet_path: Path
    ) -> None:
        """
        Summing 50 early (low-drift) and 50 late (high-drift) Motor_current
        readings: the late mean must be closer to the failure mean than the
        early mean (demonstrates the drift is working in the right direction).
        """
        deg_sim = SensorSimulator(mode=SimulatorMode.DEGRADATION, parquet_path=parquet_path)
        fail_sim = SensorSimulator(mode=SimulatorMode.FAILURE, parquet_path=parquet_path)

        fail_mean = np.mean([fail_sim.generate_reading().Motor_current for _ in range(50)])

        # Early drift (step 1→50)
        early_mean = np.mean([deg_sim.generate_reading().Motor_current for _ in range(50)])

        # Late drift (step 250→299 → drift≈0.83–1.0)
        deg_sim._step = 249
        late_mean = np.mean([deg_sim.generate_reading().Motor_current for _ in range(50)])

        # Distance from failure: late readings must be closer to failure than early ones.
        assert abs(late_mean - fail_mean) < abs(early_mean - fail_mean)

    # ── Mode transitions ──────────────────────────────────────────────────

    def test_transition_normal_to_failure_then_back(
        self, sim: SensorSimulator
    ) -> None:
        sim.mode = SimulatorMode.FAILURE
        assert sim.mode == SimulatorMode.FAILURE
        assert sim._step == 0

        sim.mode = SimulatorMode.NORMAL
        assert sim.mode == SimulatorMode.NORMAL
        assert sim._step == 0

    def test_index_not_reset_on_mode_change(self, sim: SensorSimulator) -> None:
        """Pointer continuity: switching modes does not rewind the data stream."""
        for _ in range(10):
            sim.generate_reading()
        idx_before = sim._idx_normal

        sim.mode = SimulatorMode.FAILURE
        sim.mode = SimulatorMode.NORMAL
        assert sim._idx_normal == idx_before

    def test_normal_loops_without_index_error(self, sim: SensorSimulator) -> None:
        """Streaming more rows than the partition size must wrap silently."""
        n_rows = len(sim._normal) + 5
        for _ in range(n_rows):
            sim.generate_reading()   # must not raise

    def test_failure_loops_without_index_error(self, sim_failure: SensorSimulator) -> None:
        n_rows = len(sim_failure._failure) + 5
        for _ in range(n_rows):
            sim_failure.generate_reading()

    def test_reading_changes_statistical_profile_after_mode_switch(
        self, sim: SensorSimulator
    ) -> None:
        """Motor_current distributions must differ between NORMAL and FAILURE."""
        normal_readings = [sim.generate_reading().Motor_current for _ in range(200)]
        sim.mode = SimulatorMode.FAILURE
        failure_readings = [sim.generate_reading().Motor_current for _ in range(200)]
        assert abs(np.mean(failure_readings) - np.mean(normal_readings)) > 0.5


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

    def test_stream_service_uses_simulator_mode(
        self, sim: SensorSimulator
    ) -> None:
        """
        Switching the simulator to FAILURE must produce readings that are
        statistically different from NORMAL (200-sample means differ by > 0.5 A).
        """
        from src.services.sensor_stream_service import SensorStreamService

        service = SensorStreamService(simulator=sim)

        normal_readings = [
            service._generate_reading().Motor_current for _ in range(200)
        ]

        sim.mode = SimulatorMode.FAILURE
        failure_readings = [
            service._generate_reading().Motor_current for _ in range(200)
        ]

        assert abs(np.mean(failure_readings) - np.mean(normal_readings)) > 0.5
