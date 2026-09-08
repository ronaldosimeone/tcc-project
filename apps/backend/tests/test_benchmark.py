"""
Tests for benchmark_models.py (RF-18, RNF-36).

Coverage
--------
Lógica pura de benchmark (sem depender de timing real / carregar modelos):
  • latency_stats — mean/p50/p95/p99 corretos para amostras conhecidas.
  • compute_f1 — F1 da classe 1 confere com sklearn para casos conhecidos.
  • is_eligible — regra RF-18 (p95 < 100ms).
  • select_winner — filtra não elegíveis, nunca escolhe p95 >= 100ms,
    desempate determinístico (F1 desc -> p95 asc -> memória asc), retorna
    None quando nenhum modelo é elegível.
  • find_largest_anomaly_run — localiza a maior sequência contígua de 1s.
  • update_env_active_model — substitui/adiciona ACTIVE_MODEL preservando
    o resto do arquivo.

Medições reais de performance (tempo/memória) NÃO são testadas aqui — são
inerentemente instáveis; a validação end-to-end é rodar
`python benchmark_models.py` de fato (ver seção de validação da task).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_models import (  # noqa: E402
    LatencyStats,
    ModelResult,
    build_eval_slice,
    compute_f1,
    find_largest_anomaly_run,
    is_eligible,
    latency_stats,
    select_winner,
    update_env_active_model,
)

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# latency_stats
# ---------------------------------------------------------------------------


def test_latency_stats_basic_percentiles() -> None:
    samples = [float(i) for i in range(1, 101)]  # 1..100
    stats = latency_stats(samples)

    assert stats.n_samples == 100
    assert stats.mean_ms == pytest.approx(50.5)
    assert stats.p50_ms == pytest.approx(np.percentile(samples, 50))
    assert stats.p95_ms == pytest.approx(np.percentile(samples, 95))
    assert stats.p99_ms == pytest.approx(np.percentile(samples, 99))


def test_latency_stats_empty_raises() -> None:
    with pytest.raises(ValueError):
        latency_stats([])


def test_latency_stats_p95_exists_and_orders_correctly() -> None:
    stats = latency_stats([10.0, 20.0, 30.0, 40.0, 200.0])
    assert stats.p50_ms <= stats.p95_ms <= stats.p99_ms


# ---------------------------------------------------------------------------
# compute_f1
# ---------------------------------------------------------------------------


def test_compute_f1_perfect_prediction() -> None:
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 1]
    assert compute_f1(y_true, y_pred) == pytest.approx(1.0)


def test_compute_f1_all_wrong_on_positive_class() -> None:
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 0, 0, 0, 0]
    assert compute_f1(y_true, y_pred) == pytest.approx(0.0)


def test_compute_f1_known_value() -> None:
    # precision=2/3, recall=2/2 -> F1 = 2*P*R/(P+R) = 0.8
    y_true = [1, 1, 0, 0]
    y_pred = [1, 1, 1, 0]
    assert compute_f1(y_true, y_pred) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# is_eligible (RF-18)
# ---------------------------------------------------------------------------


def test_is_eligible_below_threshold() -> None:
    assert is_eligible(99.99) is True


def test_is_eligible_at_threshold_is_not_eligible() -> None:
    assert is_eligible(100.0) is False


def test_is_eligible_above_threshold() -> None:
    assert is_eligible(150.0) is False


# ---------------------------------------------------------------------------
# select_winner (RF-18 — regra de seleção e desempate)
# ---------------------------------------------------------------------------


def _result(
    name: str,
    f1: float,
    p95: float,
    memory_mb: float = 100.0,
    eligible: bool | None = None,
) -> ModelResult:
    stats = LatencyStats(
        mean_ms=p95 * 0.6, p50_ms=p95 * 0.8, p95_ms=p95, p99_ms=p95 * 1.1, n_samples=100
    )
    return ModelResult(
        name=name,
        f1=f1,
        latency=stats,
        memory_mb=memory_mb,
        decision_threshold=0.5,
        n_eval=100,
        eligible=is_eligible(p95) if eligible is None else eligible,
    )


def test_select_winner_never_picks_ineligible_model() -> None:
    results = [
        _result("slow_high_f1", f1=0.99, p95=250.0),  # não elegível — nunca vence
        _result("fast_lower_f1", f1=0.80, p95=10.0),
    ]
    winner = select_winner(results)
    assert winner is not None
    assert winner.name == "fast_lower_f1"


def test_select_winner_none_when_all_ineligible() -> None:
    results = [
        _result("a", f1=0.99, p95=150.0),
        _result("b", f1=0.95, p95=120.0),
    ]
    assert select_winner(results) is None


def test_select_winner_deterministic_tiebreak_by_f1() -> None:
    # Scores praticamente equivalentes -> desempate por maior F1.
    results = [
        _result("a", f1=0.90, p95=10.0, memory_mb=50.0),
        _result("b", f1=0.95, p95=10.0, memory_mb=50.0),
    ]
    winner = select_winner(results)
    assert winner is not None
    assert winner.name == "b"  # maior F1 vence o empate


def test_select_winner_tiebreak_by_latency_when_f1_tied() -> None:
    results = [
        _result("a", f1=0.90, p95=20.0, memory_mb=50.0),
        _result("b", f1=0.90, p95=5.0, memory_mb=50.0),
    ]
    winner = select_winner(results)
    assert winner is not None
    assert winner.name == "b"  # mesmo F1, menor p95 vence


def test_select_winner_ignores_errored_results() -> None:
    ok = _result("ok", f1=0.9, p95=10.0)
    broken = _result("broken", f1=0.99, p95=1.0)
    broken.error = "artefato ausente"
    winner = select_winner([ok, broken])
    assert winner is not None
    assert winner.name == "ok"


def test_select_winner_higher_f1_wins_when_not_tied() -> None:
    results = [
        _result("low_f1", f1=0.5, p95=10.0, memory_mb=10.0),
        _result("high_f1", f1=0.99, p95=10.0, memory_mb=10.0),
    ]
    winner = select_winner(results)
    assert winner is not None
    assert winner.name == "high_f1"


# ---------------------------------------------------------------------------
# find_largest_anomaly_run
# ---------------------------------------------------------------------------


def test_find_largest_anomaly_run_picks_biggest_block() -> None:
    y = np.array([0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0])
    start, end = find_largest_anomaly_run(y)
    assert (start, end) == (5, 9)
    assert (y[start:end] == 1).all()


def test_find_largest_anomaly_run_run_at_edges() -> None:
    y = np.array([1, 1, 1, 0, 0, 1])
    start, end = find_largest_anomaly_run(y)
    assert (start, end) == (0, 3)


def test_find_largest_anomaly_run_no_positives_raises() -> None:
    with pytest.raises(ValueError):
        find_largest_anomaly_run(np.zeros(10))


# ---------------------------------------------------------------------------
# build_eval_slice — preserva ordem, não gera dados sintéticos
# ---------------------------------------------------------------------------


def test_build_eval_slice_is_contiguous_and_ordered() -> None:
    n = 5000
    y = np.zeros(n, dtype=int)
    y[3000:3200] = 1  # única falha real, contígua
    df = pd.DataFrame({"anomaly": y, "row_id": np.arange(n)})

    sliced = build_eval_slice(df, pre_context=50, eval_normal=100, eval_anomaly=150)

    # linha a linha idêntica ao dataset original na mesma faixa -> sem shuffle/sintético
    expected_start = 3000 - 50 - 100
    assert list(sliced["row_id"]) == list(range(expected_start, 3000 + 150))
    assert sliced["anomaly"].iloc[-1] == 1
    assert sliced["anomaly"].iloc[0] == 0


def test_build_eval_slice_raises_when_dataset_too_short() -> None:
    y = np.array([0, 0, 1, 1])
    df = pd.DataFrame({"anomaly": y})
    with pytest.raises(ValueError):
        build_eval_slice(df, pre_context=50, eval_normal=100, eval_anomaly=1)


# ---------------------------------------------------------------------------
# update_env_active_model
# ---------------------------------------------------------------------------


def test_update_env_active_model_replaces_existing_line(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "POSTGRES_USER=admin\nACTIVE_MODEL=xgboost\nOLLAMA_BASE_URL=http://x\n",
        encoding="utf-8",
    )

    changed = update_env_active_model(env_file, "mlp")

    assert changed is True
    content = env_file.read_text(encoding="utf-8")
    assert "ACTIVE_MODEL=mlp" in content
    assert "POSTGRES_USER=admin" in content  # outras linhas preservadas
    assert "OLLAMA_BASE_URL=http://x" in content
    assert content.count("ACTIVE_MODEL=") == 1


def test_update_env_active_model_appends_when_missing(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("POSTGRES_USER=admin\n", encoding="utf-8")

    update_env_active_model(env_file, "tcn")

    content = env_file.read_text(encoding="utf-8")
    assert "ACTIVE_MODEL=tcn" in content
    assert "POSTGRES_USER=admin" in content


def test_update_env_active_model_missing_file_returns_false(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.env"
    assert update_env_active_model(missing, "mlp") is False
