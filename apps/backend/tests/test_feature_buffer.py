"""
Testes de `SensorBuffer` (feature engineering stateful, RNF-64/RNF-65).

Auditoria desta task: 76% de cobertura, zero teste dedicado — só exercitado
de raspão via `InferencePipelineService`. Validação de construtor
(`window_size`/`warmup_size`), FIFO com eviction, `is_warm()` (limiar
`>=`, não `>`), `to_dataframe()` (ordem cronológica), `clear()`.
"""

from __future__ import annotations

import pandas as pd
import pytest

import src.services.feature_buffer as feature_buffer_module
from src.services.feature_buffer import SensorBuffer


# ---------------------------------------------------------------------------
# Defaults do construtor + singleton de módulo (RNF-64) — nenhum teste
# pré-existente construía `SensorBuffer()` sem sobrescrever window_size/
# warmup_size, então os defaults reais (30/15) nunca eram verificados.
# ---------------------------------------------------------------------------


def test_constructor_defaults_are_window_30_warmup_15() -> None:
    buf = SensorBuffer()
    assert buf.capacity == 30
    assert buf.warmup_size == 15


def test_warmup_size_is_a_property_not_a_bound_method() -> None:
    """Mata o mutante que remove o decorator `@property` de `warmup_size`
    — sem ele, `buf.warmup_size` devolveria um método, não um `int`."""
    buf = SensorBuffer(window_size=10, warmup_size=4)
    assert buf.warmup_size == 4
    assert isinstance(buf.warmup_size, int)


def test_module_level_buffer_singleton_uses_the_documented_defaults() -> None:
    assert feature_buffer_module.buffer.capacity == 30  # noqa: SLF001
    assert feature_buffer_module.buffer.warmup_size == 15  # noqa: SLF001


# ---------------------------------------------------------------------------
# Construtor — validação
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("window_size", [0, -1, -100])
def test_constructor_rejects_non_positive_window_size(window_size: int) -> None:
    with pytest.raises(ValueError, match="window_size"):
        SensorBuffer(window_size=window_size, warmup_size=1)


@pytest.mark.parametrize("warmup_size", [0, -1])
def test_constructor_rejects_non_positive_warmup_size(warmup_size: int) -> None:
    with pytest.raises(ValueError, match="warmup_size"):
        SensorBuffer(window_size=10, warmup_size=warmup_size)


def test_constructor_rejects_warmup_size_larger_than_window_size() -> None:
    with pytest.raises(ValueError, match="warmup_size"):
        SensorBuffer(window_size=5, warmup_size=6)


def test_constructor_accepts_warmup_size_equal_to_window_size() -> None:
    """Fronteira: `warmup_size == window_size` é o limite VÁLIDO — só
    `warmup_size > window_size` deve ser rejeitado (`> window_size`, não
    `>=`, no código real)."""
    buf = SensorBuffer(window_size=5, warmup_size=5)
    assert buf.capacity == 5


# ---------------------------------------------------------------------------
# append / __len__ / capacity — FIFO com eviction
# ---------------------------------------------------------------------------


def test_append_increases_length_up_to_window_size() -> None:
    buf = SensorBuffer(window_size=3, warmup_size=1)
    assert len(buf) == 0
    buf.append({"TP2": 1.0})
    assert len(buf) == 1
    buf.append({"TP2": 2.0})
    buf.append({"TP2": 3.0})
    assert len(buf) == 3


def test_append_evicts_oldest_when_window_is_full() -> None:
    buf = SensorBuffer(window_size=2, warmup_size=1)
    buf.append({"TP2": 1.0})
    buf.append({"TP2": 2.0})
    buf.append({"TP2": 3.0})  # deve expulsar o TP2=1.0

    assert len(buf) == 2
    df = buf.to_dataframe()
    assert df["TP2"].tolist() == [2.0, 3.0]


def test_append_stores_a_defensive_copy_not_a_reference() -> None:
    """Mutar o dict original DEPOIS de `append` não deve alterar o que já
    foi armazenado — proteção contra um chamador reaproveitar o mesmo dict
    (padrão comum: `reading = {...}; buffer.append(reading); reading["x"] = 1`)."""
    buf = SensorBuffer(window_size=2, warmup_size=1)
    original = {"TP2": 1.0}
    buf.append(original)
    original["TP2"] = 999.0

    assert buf.to_dataframe()["TP2"].iloc[0] == 1.0


def test_capacity_reflects_configured_window_size() -> None:
    assert SensorBuffer(window_size=42, warmup_size=1).capacity == 42


# ---------------------------------------------------------------------------
# is_warm — limiar >= (nao >)
# ---------------------------------------------------------------------------


def test_is_warm_false_below_warmup_threshold() -> None:
    buf = SensorBuffer(window_size=10, warmup_size=3)
    buf.append({"TP2": 1.0})
    buf.append({"TP2": 2.0})
    assert buf.is_warm() is False


def test_is_warm_true_exactly_at_warmup_threshold() -> None:
    """Fronteira: `len(buf) == warmup_size` já deve ser `True` — a regra é
    `>=`, não `>` (aquecer exatamente no N-ésimo sample, não no N+1)."""
    buf = SensorBuffer(window_size=10, warmup_size=3)
    for _ in range(3):
        buf.append({"TP2": 1.0})
    assert buf.is_warm() is True


def test_is_warm_true_above_warmup_threshold() -> None:
    buf = SensorBuffer(window_size=10, warmup_size=3)
    for _ in range(5):
        buf.append({"TP2": 1.0})
    assert buf.is_warm() is True


def test_is_warm_false_when_empty() -> None:
    assert SensorBuffer(window_size=10, warmup_size=3).is_warm() is False


# ---------------------------------------------------------------------------
# to_dataframe — ordem cronológica (mais antigo primeiro)
# ---------------------------------------------------------------------------


def test_to_dataframe_preserves_insertion_order_oldest_first() -> None:
    buf = SensorBuffer(window_size=5, warmup_size=1)
    for v in (10.0, 20.0, 30.0):
        buf.append({"TP2": v})

    df = buf.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df["TP2"].tolist() == [10.0, 20.0, 30.0]


def test_to_dataframe_empty_buffer_returns_empty_dataframe() -> None:
    df = SensorBuffer(window_size=5, warmup_size=1).to_dataframe()
    assert df.empty


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


def test_clear_empties_the_buffer_and_resets_warm_state() -> None:
    buf = SensorBuffer(window_size=5, warmup_size=2)
    buf.append({"TP2": 1.0})
    buf.append({"TP2": 2.0})
    assert buf.is_warm() is True

    buf.clear()

    assert len(buf) == 0
    assert buf.is_warm() is False
    assert buf.to_dataframe().empty
