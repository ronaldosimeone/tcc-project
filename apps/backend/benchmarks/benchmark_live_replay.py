"""
Live-path replay of the real failure scenarios — training/serving parity.

Feeds every scenario of ``failure_scenarios.SCENARIOS`` through the EXACT
serving path used by the running app — ``SensorSimulator`` →
``failure_scenarios.start_scenario`` (buffer primed with the recording's own
history) → ``SensorBuffer`` → ``MetroPTPreprocessor`` → served ONNX model
(``load_model_by_name``, card threshold) — and reports, per model, when the
RN-03 persistent alarm (k = 6 consecutive readings ≥ threshold) first fires.
As offline, an alarm before the failure start but inside the early horizon
(6 h, longer than the replayed lead) counts as an early detection.

The simulator emits one 10 s row per second, so "replay s" is the wall-clock
time an audience waits after pressing Play.

Run inside the API container (models and data mounted):
    docker exec -w /app api python benchmarks/benchmark_live_replay.py
Output: benchmarks/live_replay_matrix.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_BACKEND: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

from src.services import failure_scenarios as fs  # noqa: E402
from src.services.feature_buffer import SensorBuffer  # noqa: E402
from src.services.inference_pipeline import (  # noqa: E402
    _infer_with_history,
    _reading_to_dict,
    has_full_window,
)
from src.services.model_service import load_model_by_name  # noqa: E402
from src.services.preprocessing import MetroPTPreprocessor  # noqa: E402
from src.services.simulator import get_simulator  # noqa: E402

OUT: Path = Path(__file__).resolve().parent / "live_replay_matrix.json"
MODELS: tuple[str, ...] = (
    "random_forest_v3",
    "xgboost_v3",
    "mlp_v3",
    "tcn_v3",
    "bilstm_v3",
    "patchtst_v3",
    "autoencoder_v3",
)
LEAD_MINUTES: int = 10
MAX_FAILURE_ROWS: int = 720  # first 2 h of each failure
K_OPEN: int = 6


def replay(model_name: str, scenario: fs.FailureScenario) -> dict[str, Any]:
    ms = load_model_by_name(model_name)
    sim = get_simulator()
    buffer = SensorBuffer(window_size=60, warmup_size=15)
    fs.start_scenario(sim, buffer, scenario.id, LEAD_MINUTES)
    status = sim.scenario_status()
    assert status is not None
    lead_rows = int(sim._scenario_lead_rows)  # noqa: SLF001
    total = status["total"]
    assert isinstance(total, int)
    n = min(total, lead_rows + MAX_FAILURE_ROWS)
    pre = MetroPTPreprocessor()
    probs = np.full(n, np.nan)
    for i in range(n):
        buffer.append(_reading_to_dict(sim.generate_reading()))
        if buffer.is_warm() and has_full_window(ms, len(buffer)):
            probs[i] = _infer_with_history(
                ms, buffer.to_dataframe(), pre
            ).failure_probability
    above = np.nan_to_num(probs, nan=-1.0) >= ms.decision_threshold
    run = np.convolve(above.astype(int), np.ones(K_OPEN, int), "full")[:n] >= K_OPEN
    first = np.flatnonzero(run)
    first_row = int(first[0]) if first.size else None
    return {
        "model": model_name,
        "scenario": scenario.id,
        "seen_in_training": scenario.seen_in_training,
        "threshold": ms.decision_threshold,
        "max_probability": float(np.nanmax(probs)),
        # Same rule as the offline protocol: an alarm inside the early horizon
        # (6 h ≥ the replayed lead) before the start counts as detection.
        "detected": first_row is not None,
        "early_warning": first_row is not None and first_row < lead_rows,
        "first_alarm_replay_s": first_row,
        "first_alarm_vs_failure_start_min": (
            None if first_row is None else round((first_row - lead_rows) * 10 / 60, 1)
        ),
    }


def main() -> None:
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for model_name in MODELS:
        for scenario in fs.SCENARIOS:
            if not fs.available(scenario):
                continue
            row = replay(model_name, scenario)
            rows.append(row)
            print(
                f"{model_name:16s} {scenario.id:18s} detected={row['detected']} "
                f"at={row['first_alarm_vs_failure_start_min']} min",
                flush=True,
            )
    OUT.write_text(
        json.dumps(
            {
                "lead_minutes": LEAD_MINUTES,
                "max_failure_rows": MAX_FAILURE_ROWS,
                "k_open": K_OPEN,
                "seconds": round(time.perf_counter() - t0, 1),
                "rows": rows,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
