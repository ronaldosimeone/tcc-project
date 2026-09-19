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

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.services.simulator as simulator_module
from src.core.config import settings
from src.routers.simulator import router as simulator_router
from src.schemas.stream import SensorReading
from src.services.simulator import (
    _DEGRADATION_HORIZON,
    _build_failure_mask_from_timestamps,
    _load_and_split,
    _row_to_reading,
    SimulatorMode,
    SensorSimulator,
    get_simulator,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# RNF-65 — usa a MESMA fonte de verdade que a aplicação real
# (`settings.simulator_parquet_path`, apps/backend/src/core/config.py) em vez
# de recalcular o caminho a partir de `__file__`: um `Path(__file__).resolve()
# .parents[N]` hand-rolled aqui já teve o offset errado (contava a partir de
# `tests/`, não de `src/core/`, apontando um nível ACIMA da raiz do repo —
# `.../metropt3.parquet` em vez de `.../apps/ml/data/processed/metropt3.parquet`)
# e fazia as 40 classes de teste dependentes de dado real SEMPRE pular, mesmo
# com o parquet presente. Reusar `settings` elimina essa classe de bug.
@pytest.fixture(scope="session")
def parquet_path() -> Path:
    path = settings.simulator_parquet_path
    if not path.exists():
        pytest.skip("metropt3.parquet not found — skipping data-dependent tests")
    return path


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
        """Real MetroPT-3 binary sensors are stored as 0.0 / 1.0 — must be preserved."""
        binary = {"COMP", "DV_eletric", "Towers", "MPG", "Oil_level"}
        for _ in range(30):
            r = sim.generate_reading()
            for f in binary:
                assert getattr(r, f) in (0.0, 1.0)

    # ── NORMAL mode — physical range checks (real data) ───────────────────

    def test_normal_tp2_within_plausible_sensor_range(
        self, sim: SensorSimulator
    ) -> None:
        """
        TP2 (pressão a jusante) fica dentro da faixa plausível do sensor.

        Achado real desta task (RNF-65): a asserção original exigia
        `TP2 >= 0.0` sempre — falsa sobre o dataset MetroPT-3 real, onde
        1.275.474 das 1.516.948 linhas (84%) têm `TP2` levemente negativo
        (mín. -0.032), ruído normal de um sensor de pressão operando perto
        de zero/vácuo, não uma anomalia. O simulador repassa o dado real
        verbatim (RF-13) — o bug era a expectativa do teste, não o
        simulador. Faixa abaixo cobre o mín./máx. reais do dataset com
        folga (nunca visto fora de [-1, 12] nas 1.5M linhas).
        """
        readings = [sim.generate_reading().TP2 for _ in range(50)]
        assert all(-1.0 <= v <= 12.0 for v in readings)

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
        normal_mean = np.mean(
            [sim.generate_reading().Motor_current for _ in range(200)]
        )
        fail_mean = np.mean(
            [sim_failure.generate_reading().Motor_current for _ in range(200)]
        )
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
        fail_sim = SensorSimulator(
            mode=SimulatorMode.FAILURE, parquet_path=parquet_path
        )
        deg_sim = SensorSimulator(
            mode=SimulatorMode.DEGRADATION, parquet_path=parquet_path
        )
        deg_sim._step = 299  # next generate_reading → step=300, drift=1.0

        # Both pointers start at 0 → first row must match exactly at drift=1.
        fail_val = fail_sim.generate_reading().Motor_current
        deg_val = deg_sim.generate_reading().Motor_current
        assert abs(deg_val - fail_val) < 1e-4

    def test_degradation_blend_is_monotonic_over_time(self, parquet_path: Path) -> None:
        """
        Compara o MESMO par (normal_row, failure_row) blendado em dois
        valores de drift diferentes — o único jeito determinístico de provar
        que o blend se aproxima da falha conforme o drift cresce, sem
        depender da variação natural do dado real de um sensor cíclico
        (Motor_current liga/desliga com o compressor: uma janela de 50
        leituras SEQUENCIAIS pode cair inteira num pico ou num vale,
        mascarando o efeito do drift — não é o blend que seria testado, e
        sim a sorte de qual trecho do dataset caiu na janela).

        Achado real desta task (RNF-65): a versão anterior deste teste
        setava `deg_sim._step` diretamente para pular pra "tarde" (drift
        alto) SEM avançar `_idx_normal`/`_idx_failure` junto — os ponteiros
        de linha continuavam onde as primeiras 50 chamadas os deixaram
        (~linha 50), então o "late" lia linhas bem mais cedo do dataset que
        seu próprio drift alto sugeria, e por coincidência do dado real
        (`Motor_current` tem um pico de operação entre as linhas ~50-100 do
        MetroPT-3) o teste falhava — não porque o blend estivesse errado
        (`test_degradation_at_drift_zero_close_to_normal`/
        `..._at_full_drift_close_to_failure`, que testam os EXTREMOS do
        mesmo jeito determinístico abaixo, sempre passaram), mas porque a
        comparação em si media coisas de linhas diferentes do dataset.
        Corrigido consumindo `generate_reading()` de verdade até o step
        desejado (avança step E ponteiros juntos, do jeito real) em vez de
        sobrescrever `_step` isoladamente no meio de uma sequência de leituras.
        """
        deg_sim = SensorSimulator(
            mode=SimulatorMode.DEGRADATION, parquet_path=parquet_path
        )

        # step=1 (drift≈1/300≈0.003): blend praticamente = normal_row puro.
        low_drift_val = deg_sim.generate_reading().Motor_current

        # Avança step E ponteiros JUNTOS (consumindo leituras de verdade,
        # nunca sobrescrevendo `_step` isolado) até restar 1 tick para
        # step=300 (drift=1.0) — pega o MESMO par (normal_row, failure_row)
        # que seria lido em drift=1.0 a partir daqui.
        for _ in range(_DEGRADATION_HORIZON - 2):
            deg_sim.generate_reading()
        high_drift_val = deg_sim.generate_reading().Motor_current  # step=300, drift=1.0

        # Ponteiros consistentes com o próprio avanço acima — lê a MESMA
        # posição de failure_row que o `deg_sim` acabou de consumir via
        # blend, para comparar contra o valor puro de failure naquela linha.
        fail_sim = SensorSimulator(
            mode=SimulatorMode.FAILURE, parquet_path=parquet_path
        )
        for _ in range(_DEGRADATION_HORIZON - 1):
            fail_sim.generate_reading()
        pure_failure_at_same_row = fail_sim.generate_reading().Motor_current

        # A leitura de alto drift deve estar MUITO mais perto da falha pura
        # (mesma linha) do que a leitura de drift quase zero estava —
        # provando que o blend converge para failure_row conforme step→300,
        # sem depender de médias sobre um sinal real cíclico.
        assert abs(high_drift_val - pure_failure_at_same_row) < abs(
            low_drift_val - pure_failure_at_same_row
        )

    # ── Mode transitions ──────────────────────────────────────────────────

    def test_transition_normal_to_failure_then_back(self, sim: SensorSimulator) -> None:
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
            sim.generate_reading()  # must not raise

    def test_failure_loops_without_index_error(
        self, sim_failure: SensorSimulator
    ) -> None:
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

    # ── Índices de linha — inicialização e avanço exato ─────────────────────

    def test_idx_normal_and_idx_failure_start_at_zero(
        self, sim: SensorSimulator
    ) -> None:
        assert sim._idx_normal == 0
        assert sim._idx_failure == 0

    def test_idx_normal_advances_by_exactly_1_per_reading_with_wraparound(
        self, sim: SensorSimulator
    ) -> None:
        n = len(sim._normal)
        assert sim._idx_normal == 0
        sim.generate_reading()
        assert sim._idx_normal == 1
        sim.generate_reading()
        assert sim._idx_normal == 2
        # Avança exatamente até o fim da partição — o próximo dá a volta pra 0.
        for _ in range(n - 3):
            sim.generate_reading()
        assert sim._idx_normal == n - 1
        sim.generate_reading()
        assert sim._idx_normal == 0

    def test_idx_failure_advances_by_exactly_1_per_reading_with_wraparound(
        self, sim_failure: SensorSimulator
    ) -> None:
        n = len(sim_failure._failure)
        sim_failure.generate_reading()
        assert sim_failure._idx_failure == 1
        for _ in range(n - 2):
            sim_failure.generate_reading()
        assert sim_failure._idx_failure == n - 1
        sim_failure.generate_reading()
        assert sim_failure._idx_failure == 0

    # ── Fórmula de drift (DEGRADATION) ───────────────────────────────────────

    def test_drift_formula_is_step_divided_by_horizon_capped_at_1(
        self, parquet_path: Path
    ) -> None:
        """RNF-64: `drift = min(step / HORIZON, 1.0)` — testa um step
        intermediário exato (não só os extremos 0 e 1.0 já cobertos pelos
        testes acima), mata `*` no lugar de `/` e o cap `2.0` no lugar de
        `1.0`."""
        deg_sim = SensorSimulator(
            mode=SimulatorMode.DEGRADATION, parquet_path=parquet_path
        )
        fail_sim = SensorSimulator(
            mode=SimulatorMode.FAILURE, parquet_path=parquet_path
        )
        normal_sim = SensorSimulator(
            mode=SimulatorMode.NORMAL, parquet_path=parquet_path
        )

        half_horizon = _DEGRADATION_HORIZON // 2
        for _ in range(half_horizon - 1):
            deg_sim.generate_reading()
            fail_sim.generate_reading()
            normal_sim.generate_reading()
        blended = deg_sim.generate_reading().Motor_current
        pure_normal = normal_sim.generate_reading().Motor_current
        pure_failure = fail_sim.generate_reading().Motor_current

        expected_drift = half_horizon / _DEGRADATION_HORIZON  # == 0.5
        expected = pure_normal + expected_drift * (pure_failure - pure_normal)
        assert blended == pytest.approx(expected, rel=1e-5)


# ===========================================================================
# _row_to_reading — mapeamento exato coluna -> campo (RNF-64)
# ===========================================================================


class TestRowToReading:
    def test_every_field_maps_to_its_exact_column_index(self) -> None:
        """Array com 12 valores DISTINTOS (0..11) — qualquer troca de índice
        entre campos faz a asserção correspondente falhar."""
        row = np.array([float(i) for i in range(12)], dtype=np.float32)
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

        reading = _row_to_reading(row, ts)

        assert reading.TP2 == 0.0
        assert reading.TP3 == 1.0
        assert reading.H1 == 2.0
        assert reading.DV_pressure == 3.0
        assert reading.Reservoirs == 4.0
        assert reading.Motor_current == 5.0
        assert reading.Oil_temperature == 6.0
        assert reading.COMP == 7.0
        assert reading.DV_eletric == 8.0
        assert reading.Towers == 9.0
        assert reading.MPG == 10.0
        assert reading.Oil_level == 11.0
        assert reading.timestamp == ts


# ===========================================================================
# _build_failure_mask_from_timestamps — fronteiras exatas de janela (RNF-64)
# ===========================================================================


class TestBuildFailureMaskFromTimestamps:
    # Janela de falha real conhecida (mesma usada pelo simulador/PSI).
    _WINDOW_START = "2020-04-18 00:00:00"
    _WINDOW_END = "2020-04-18 23:59:00"

    def test_timestamp_exactly_at_window_start_is_inside_the_failure_window(
        self,
    ) -> None:
        """Fronteira: início da janela é INCLUSIVO (`>=`, não `>`)."""
        ts = pd.Series([pd.Timestamp(self._WINDOW_START)])
        mask = _build_failure_mask_from_timestamps(ts)
        assert mask[0] is np.True_ or bool(mask[0]) is True

    def test_timestamp_exactly_at_window_end_is_inside_the_failure_window(self) -> None:
        """Fronteira: fim da janela é INCLUSIVO (`<=`, não `<`)."""
        ts = pd.Series([pd.Timestamp(self._WINDOW_END)])
        mask = _build_failure_mask_from_timestamps(ts)
        assert bool(mask[0]) is True

    def test_timestamp_one_minute_before_window_start_is_outside(self) -> None:
        ts = pd.Series([pd.Timestamp(self._WINDOW_START) - pd.Timedelta(minutes=1)])
        mask = _build_failure_mask_from_timestamps(ts)
        assert bool(mask[0]) is False

    def test_timestamp_one_minute_after_window_end_is_outside(self) -> None:
        ts = pd.Series([pd.Timestamp(self._WINDOW_END) + pd.Timedelta(minutes=1)])
        mask = _build_failure_mask_from_timestamps(ts)
        assert bool(mask[0]) is False

    def test_timezone_aware_timestamps_are_stripped_before_comparison(self) -> None:
        """`hasattr(dtype, "tz")` + `.tz is not None` -> converte para naive
        antes de comparar contra as janelas (naive, paper MetroPT-3)."""
        ts = pd.Series([pd.Timestamp(self._WINDOW_START, tz="UTC")])
        mask = _build_failure_mask_from_timestamps(ts)
        assert bool(mask[0]) is True


# ===========================================================================
# _load_and_split — detecção de coluna timestamp/anomaly + fallback (RNF-64)
# ===========================================================================


class TestLoadAndSplit:
    _COLS = simulator_module._SENSOR_COLS  # noqa: SLF001

    def _write_parquet(self, tmp_path: Path, df: pd.DataFrame) -> Path:
        path = tmp_path / "synthetic.parquet"
        df.to_parquet(path, engine="pyarrow")
        return path

    def test_uses_timestamp_column_when_present_even_if_anomaly_also_present(
        self, tmp_path: Path
    ) -> None:
        n = 10
        data = {c: [float(i) for i in range(n)] for c in self._COLS}
        data["timestamp"] = [
            (
                pd.Timestamp("2020-04-18 00:30:00")
                if i < 3
                else pd.Timestamp(f"2019-01-{i+1:02d}")
            )
            for i in range(n)
        ]
        data["anomaly"] = [False] * n  # nunca deveria ser usado aqui
        path = self._write_parquet(tmp_path, pd.DataFrame(data))

        normal_rows, failure_rows = _load_and_split(path)
        assert len(failure_rows) == 3  # só as 3 dentro da janela de falha real
        assert len(normal_rows) == n - 3

    def test_falls_back_to_anomaly_column_when_no_timestamp_column(
        self, tmp_path: Path
    ) -> None:
        n = 10
        data = {c: [float(i) for i in range(n)] for c in self._COLS}
        data["anomaly"] = [i < 4 for i in range(n)]
        path = self._write_parquet(tmp_path, pd.DataFrame(data))

        normal_rows, failure_rows = _load_and_split(path)
        assert len(failure_rows) == 4
        assert len(normal_rows) == n - 4

    def test_treats_everything_as_normal_when_neither_column_present(
        self, tmp_path: Path
    ) -> None:
        n = 6
        data = {c: [float(i) for i in range(n)] for c in self._COLS}
        path = self._write_parquet(tmp_path, pd.DataFrame(data))

        normal_rows, failure_rows = _load_and_split(path)
        assert len(normal_rows) == n
        # Sem linhas de falha reais -> fallback: failure_rows == normal_rows.
        np.testing.assert_array_equal(failure_rows, normal_rows)

    def test_falls_back_to_normal_rows_when_anomaly_column_has_no_true_values(
        self, tmp_path: Path
    ) -> None:
        """Fronteira: `len(failure_rows) == 0` (não `== 1`) — coluna
        `anomaly` existe mas está inteira `False`."""
        n = 6
        data = {c: [float(i) for i in range(n)] for c in self._COLS}
        data["anomaly"] = [False] * n
        path = self._write_parquet(tmp_path, pd.DataFrame(data))

        normal_rows, failure_rows = _load_and_split(path)
        assert len(normal_rows) == n
        np.testing.assert_array_equal(failure_rows, normal_rows)


# ===========================================================================
# get_simulator() — singleton lazy (RNF-64)
# ===========================================================================


class TestGetSimulatorSingleton:
    @pytest.fixture(autouse=True)
    def _reset_singleton(self, parquet_path: Path):
        """Isola cada teste do estado global `_simulator` — nunca deixa um
        teste anterior/posterior enxergar o singleton criado aqui."""
        original = simulator_module._simulator
        simulator_module._simulator = None
        yield
        simulator_module._simulator = original

    def test_first_call_creates_a_real_instance_not_none(self) -> None:
        instance = get_simulator()
        assert instance is not None
        assert isinstance(instance, SensorSimulator)

    def test_second_call_returns_the_same_instance_not_a_new_one(self) -> None:
        first = get_simulator()
        second = get_simulator()
        assert first is second


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

    def test_stream_service_uses_simulator_mode(self, sim: SensorSimulator) -> None:
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
