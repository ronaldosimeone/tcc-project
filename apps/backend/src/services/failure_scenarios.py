"""
Catalogue of REAL failure scenarios for the simulator (demonstration and
honest live test).

Each scenario replays the true sensor sequence of one documented failure,
from its reported start (``lead_minutes`` = 0, the UI default): the platform
detects failures in progress, it does not predict them. Sources: MetroPT-3 (2020), MetroPT 2022 and
MetroPT2 (2022, Zenodo 7766691; labels from arXiv:2502.07394 Table 1).

``seen_in_training`` is the honesty flag shown in the UI:
* True  — the served models were trained on data containing this failure:
          replaying it is a DEMONSTRATION, not evidence of prediction;
* False — the failure is after every training cut-off (MetroPT2): replaying
          it is a genuine out-of-sample test, live.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from src.core.config import settings
from src.core.exceptions import BadRequestError, NotFoundError

# Last timestamp in any training set (MetroPT 2022 ends here).
TRAINING_END: datetime = datetime(2022, 6, 2, 15, 49, 50)

SENSOR_COLS: list[str] = [
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
]


@dataclass(frozen=True)
class FailureScenario:
    id: str
    label: str
    failure_type: str
    dataset: str
    parquet: str
    start: datetime
    end: datetime
    note: str

    @property
    def seen_in_training(self) -> bool:
        return self.end <= TRAINING_END


SCENARIOS: tuple[FailureScenario, ...] = (
    FailureScenario(
        "air_leak_2020_07",
        "Vazamento de ar — jul/2020",
        "air_leak",
        "MetroPT-3",
        "metropt3.parquet",
        datetime(2020, 7, 15, 14, 30),
        datetime(2020, 7, 15, 19, 0),
        "Falha de 2020 usada no treino.",
    ),
    FailureScenario(
        "air_leak_2022_02",
        "Vazamento de ar nos consumidores — fev/2022",
        "air_leak",
        "MetroPT 2022",
        "metropt2022.parquet",
        datetime(2022, 2, 28, 21, 53),
        datetime(2022, 3, 1, 2, 0),
        "Detectada fora do tempo por todos os modelos no walk-forward.",
    ),
    FailureScenario(
        "air_dryer_2022_03",
        "Válvula do secador de ar — mar/2022",
        "air_dryer",
        "MetroPT 2022",
        "metropt2022.parquet",
        datetime(2022, 3, 23, 14, 54),
        datetime(2022, 3, 23, 15, 24),
        "Nenhum modelo a detectou fora do tempo (tipo inédito no treino).",
    ),
    FailureScenario(
        "oil_leak_2022_05",
        "Vazamento de óleo — mai/2022",
        "oil_leak",
        "MetroPT 2022",
        "metropt2022.parquet",
        datetime(2022, 5, 30, 12, 0),
        datetime(2022, 6, 2, 6, 18),
        "Nenhum modelo a detectou fora do tempo (tipo inédito no treino).",
    ),
    FailureScenario(
        "air_leak_mt2_06",
        "Vazamento de ar — jun/2022 (MetroPT2)",
        "air_leak",
        "MetroPT2",
        "metropt2.parquet",
        datetime(2022, 6, 4, 10, 19, 24),
        datetime(2022, 6, 4, 14, 22, 39),
        "Posterior a todo o treino e de outra APU. Offline: 6 dos 7 modelos "
        "servidos a detectam antes do alarme LPS do trem (o autoencoder, depois).",
    ),
    FailureScenario(
        "oil_leak_mt2_07",
        "Vazamento de óleo — jul/2022 (MetroPT2)",
        "oil_leak",
        "MetroPT2",
        "metropt2.parquet",
        datetime(2022, 7, 11, 10, 10, 18),
        datetime(2022, 7, 14, 10, 22, 8),
        "Posterior a todo o treino e de outra APU. Offline: MLP/TCN/BiLSTM/"
        "PatchTST detectam ~57 h antes do LPS; RF, XGBoost e AE não.",
    ),
)
BY_ID: dict[str, FailureScenario] = {s.id: s for s in SCENARIOS}


def data_dir() -> Path:
    return settings.simulator_parquet_path.parent


def available(scenario: FailureScenario) -> bool:
    return (data_dir() / scenario.parquet).exists()


def _frame(scenario: FailureScenario) -> tuple[pd.DataFrame, pd.Series]:
    path = data_dir() / scenario.parquet
    df = pd.read_parquet(path, columns=["timestamp", *SENSOR_COLS])
    return df, pd.to_datetime(df["timestamp"])


@lru_cache(maxsize=8)
def load_rows(scenario_id: str, lead_minutes: int) -> np.ndarray:
    """Sensor rows (n, 12) from ``start − lead_minutes`` to ``end``."""
    scenario = BY_ID[scenario_id]
    df, ts = _frame(scenario)
    lo = pd.Timestamp(scenario.start - timedelta(minutes=lead_minutes))
    window = df[(ts >= lo) & (ts <= pd.Timestamp(scenario.end))]
    if window.empty:
        raise ValueError(f"No rows for scenario {scenario_id} ({scenario.parquet}).")
    return window[SENSOR_COLS].to_numpy(dtype=np.float64)


@lru_cache(maxsize=8)
def load_history(scenario_id: str, lead_minutes: int, n_rows: int) -> np.ndarray:
    """The ``n_rows`` rows of the SAME recording right before the replay
    starts — the history the real asset had at that moment."""
    scenario = BY_ID[scenario_id]
    df, ts = _frame(scenario)
    lo = pd.Timestamp(scenario.start - timedelta(minutes=lead_minutes))
    before = df[ts < lo].tail(n_rows)
    return before[SENSOR_COLS].to_numpy(dtype=np.float64)


class _Buffer(Protocol):
    @property
    def capacity(self) -> int: ...

    def clear(self) -> None: ...

    def append(self, reading: dict[str, float]) -> None: ...


class _Simulator(Protocol):
    def set_scenario(self, scenario_id: str | None, lead_minutes: int = 0) -> None: ...


def start_scenario(
    simulator: _Simulator,
    buffer: _Buffer,
    scenario_id: str | None,
    lead_minutes: int,
) -> None:
    """Replays a real failure (``None`` = back to the legacy windows).

    The inference buffer is reset to the replayed recording: its window is
    filled with the rows that precede the replay in the same file, so the
    first scored windows are real history of that asset — never a splice of
    two recordings nor a left-padded window (training/serving parity)."""
    if scenario_id is not None:
        scenario = BY_ID.get(scenario_id)
        if scenario is None:
            raise NotFoundError(f"Unknown scenario '{scenario_id}'.")
        if not available(scenario):
            raise BadRequestError(
                f"Data for '{scenario_id}' is not on this server ({scenario.parquet})."
            )
    simulator.set_scenario(scenario_id, lead_minutes)
    buffer.clear()
    if scenario_id is None:
        return
    for row in load_history(scenario_id, lead_minutes, buffer.capacity):
        buffer.append(dict(zip(SENSOR_COLS, map(float, row))))
