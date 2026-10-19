"""
Live serving must score the same inputs the offline evaluation scored.

Found live (simulator, real failure scenarios):
* sequence / autoencoder models were scored before the buffer held a full
  window — the adapter left-pads by repeating the oldest row, a constant
  segment never seen in training, and TCN raised a false alarm 19 s after
  every scenario start;
* switching scenario kept the previous recording in the 60-row buffer, so
  windows spliced two different assets.

1. A model that reads ``history_rows`` rows is not scored until the buffer
   holds that many real rows (no prediction, no alert).
2. Starting a scenario primes the inference buffer with the rows that precede
   the replay in the SAME recording (the history the real asset had).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import src.services.failure_scenarios as fs
from src.services.feature_buffer import SensorBuffer
from src.services.inference_pipeline import InferencePipelineService
from src.services.simulator import SensorSimulator, SimulatorMode
from tests.test_inference_pipeline import (
    NORMAL_READING,
    _make_predict_response,
    _make_stream_service,
)


def _sequence_registry(history: int) -> tuple[MagicMock, MagicMock]:
    ms = MagicMock()
    ms.history_rows = history
    ms.decision_threshold = 0.12
    ms.predict_from_features.return_value = _make_predict_response(0.5)
    ms.predict.return_value = _make_predict_response(0.5)
    registry = MagicMock()
    registry.get = AsyncMock(return_value=ms)
    return registry, ms


@pytest.mark.asyncio
async def test_sequence_model_waits_for_a_full_real_window() -> None:
    registry, ms = _sequence_registry(history=5)
    alert = MagicMock()
    alert.process_prediction = AsyncMock()
    svc = InferencePipelineService(
        _make_stream_service([]),
        registry,
        alert,
        sensor_buffer=SensorBuffer(window_size=10, warmup_size=2),
    )
    svc._save_only = AsyncMock()  # type: ignore[method-assign]
    for _ in range(4):
        await svc._process(NORMAL_READING)
    ms.predict.assert_not_called()
    ms.predict_from_features.assert_not_called()
    alert.process_prediction.assert_not_awaited()

    await svc._process(NORMAL_READING)  # 5th real row: window complete
    ms.predict_from_features.assert_called_once()
    assert len(ms.predict_from_features.call_args[0][0]) == 5


def test_history_rows_precede_the_replay_in_the_same_recording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    ts = pd.date_range("2022-06-04 09:00", periods=200, freq="10s")
    df = pd.DataFrame({"timestamp": ts})
    for i, col in enumerate(fs.SENSOR_COLS):
        df[col] = np.arange(200, dtype=float) + i * 1000
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: df)
    fs.load_rows.cache_clear()
    fs.load_history.cache_clear()
    scen = fs.FailureScenario(
        "x",
        "x",
        "air_leak",
        "d",
        "d.parquet",
        ts[150].to_pydatetime(),
        ts[199].to_pydatetime(),
        "",
    )
    monkeypatch.setitem(fs.BY_ID, "x", scen)
    replay = fs.load_rows("x", 10)  # starts at ts[150] - 10 min = ts[90]
    history = fs.load_history("x", 10, 60)
    assert replay[0, 0] == 90
    assert history.shape == (60, 12)
    assert history[-1, 0] == 89 and history[0, 0] == 30


def test_start_scenario_primes_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(0)
    sim = SensorSimulator(data=(rng.normal(5, 1, (50, 12)), rng.normal(9, 1, (20, 12))))
    rows = np.arange(5 * 12, dtype=float).reshape(5, 12)
    hist = np.full((60, 12), 7.0)
    monkeypatch.setattr(fs, "load_rows", lambda sid, lead: rows)
    monkeypatch.setattr(fs, "load_history", lambda sid, lead, n: hist[:n])
    monkeypatch.setattr(fs, "available", lambda s: True)
    buffer = SensorBuffer(window_size=60, warmup_size=15)
    buffer.append({c: 123.0 for c in fs.SENSOR_COLS})  # previous recording

    fs.start_scenario(sim, buffer, "air_leak_mt2_06", 10)

    assert sim.mode == SimulatorMode.FAILURE
    assert len(buffer) == 60
    assert (buffer.to_dataframe()[fs.SENSOR_COLS].to_numpy() == 7.0).all()


def test_start_scenario_none_clears_buffer() -> None:
    rng = np.random.default_rng(0)
    sim = SensorSimulator(data=(rng.normal(5, 1, (50, 12)), rng.normal(9, 1, (20, 12))))
    buffer = SensorBuffer(window_size=60, warmup_size=15)
    buffer.append({c: 1.0 for c in fs.SENSOR_COLS})
    fs.start_scenario(sim, buffer, None, 30)
    assert len(buffer) == 0
    assert sim.scenario_status() is None
