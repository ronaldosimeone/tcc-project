"""
Real failure scenarios in the simulator (demonstration vs honest live test).

1. Honesty flag: MetroPT2 failures (after every training cut-off) are the only
   ones marked unseen; every 2020/2022 failure is marked as seen in training.
2. The simulator replays the scenario rows in order, reports the phase
   (pre-failure → failure) and ``None`` restores the legacy windows.
3. API: list, select (404 unknown, 400 data missing) and progress.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.services.failure_scenarios as fs
from src.main import create_app
from src.services.feature_buffer import SensorBuffer, get_sensor_buffer
from src.services.simulator import SensorSimulator, SimulatorMode, get_simulator


def _sim() -> SensorSimulator:
    rng = np.random.default_rng(0)
    normal = rng.normal(5, 1, (50, 12))
    failure = rng.normal(9, 1, (20, 12))
    return SensorSimulator(data=(normal, failure))


@pytest.fixture()
def fake_rows(monkeypatch: pytest.MonkeyPatch) -> np.ndarray:
    rows = np.arange(5 * 12, dtype=float).reshape(5, 12)
    monkeypatch.setattr(fs, "load_rows", lambda sid, lead: rows)
    monkeypatch.setattr(fs, "load_history", lambda sid, lead, n: rows[:0])
    monkeypatch.setattr(fs, "available", lambda s: True)
    return rows


def test_only_metropt2_failures_are_unseen() -> None:
    unseen = {s.id for s in fs.SCENARIOS if not s.seen_in_training}
    assert unseen == {"air_leak_mt2_06", "oil_leak_mt2_07"}
    assert all(s.start > fs.TRAINING_END for s in fs.SCENARIOS if s.id in unseen)


def test_simulator_replays_scenario_in_order_with_phase(
    monkeypatch: pytest.MonkeyPatch, fake_rows: np.ndarray
) -> None:
    import src.services.failure_scenarios as module

    monkeypatch.setattr(module, "load_rows", lambda sid, lead: fake_rows)
    sim = _sim()
    sim.set_scenario("oil_leak_mt2_07", lead_minutes=0)
    assert sim.mode == SimulatorMode.FAILURE
    first = sim.generate_reading()
    assert first.TP2 == fake_rows[0, 0]
    st = sim.scenario_status()
    assert st is not None and st["step"] == 1 and st["phase"] == "failure"
    sim.set_scenario(None)
    assert sim.scenario_status() is None


def test_lead_window_is_reported_as_pre_failure(fake_rows: np.ndarray) -> None:
    sim = _sim()
    sim.set_scenario("air_leak_mt2_06", lead_minutes=10)  # 60 rows > 5 → clipped
    st = sim.scenario_status()
    assert st is not None and st["phase"] == "pre_failure"


@pytest.fixture()
def client(fake_rows: np.ndarray) -> Iterator[TestClient]:
    app = create_app()
    sim = _sim()
    app.dependency_overrides[get_simulator] = lambda: sim
    # Isolated buffer: the endpoint resets/primes it (never the global one).
    app.dependency_overrides[get_sensor_buffer] = lambda: SensorBuffer()
    with TestClient(app) as c:
        yield c


def test_api_lists_scenarios_with_honesty_flag(client: TestClient) -> None:
    body = client.get("/simulator/scenarios").json()
    assert {s["id"] for s in body} == set(fs.BY_ID)
    flags = {s["id"]: s["seen_in_training"] for s in body}
    assert flags["oil_leak_mt2_07"] is False and flags["oil_leak_2022_05"] is True


def test_api_select_and_progress(client: TestClient) -> None:
    r = client.put(
        "/simulator/scenario", json={"id": "air_leak_mt2_06", "lead_minutes": 0}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "FAILURE" and body["scenario"]["id"] == "air_leak_mt2_06"
    assert client.get("/simulator/scenario").json()["total"] == 5


def test_api_rejects_unknown_and_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert client.put("/simulator/scenario", json={"id": "nope"}).status_code == 404
    monkeypatch.setattr(fs, "available", lambda s: False)
    assert (
        client.put("/simulator/scenario", json={"id": "air_leak_mt2_06"}).status_code
        == 400
    )
