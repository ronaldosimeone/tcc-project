"""
Benchmark comparativo dos modelos de produção do PredictIQ (RF-18, RNF-36).

Objetivo
--------
Decidir, com dados objetivos medidos sob as mesmas condições, qual modelo
listado em ``ModelRegistry.KNOWN_MODELS`` deve ser o ``ACTIVE_MODEL`` de
produção. Critérios: F1 (classe 1 / falha), latência de inferência (p95) e
memória do artefato carregado.

RF-18 — o vencedor é o melhor equilíbrio entre F1, latência e memória,
        desde que p95 < 100 ms. Modelos com p95 >= 100 ms são marcados
        "não elegíveis" e nunca vencem.
RNF-36 — todo o processo roda com um único comando:

    python benchmark_models.py

Reaproveitamento de produção (não há lógica paralela)
-------------------------------------------------------
* Lista de candidatos: ``src.services.model_registry.KNOWN_MODELS`` (a mesma
  fonte usada pelo endpoint admin ``PUT /models/active``).
* Carregamento: ``src.services.model_service.load_model_by_name`` — o mesmo
  factory usado no startup da API (``load_active_model``) e no hot-swap.
* Inferência: ``src.services.inference_pipeline._infer_with_history`` — a
  MESMA função que o pipeline de produção chama por leitura (preprocess +
  predict). Isso inclui a peculiaridade real do sistema hoje: o pipeline só
  passa a ÚLTIMA linha do buffer para ``predict_from_features``, então os
  modelos sequenciais (TCN/BiLSTM/PatchTST) e o autoencoder recebem uma
  janela "cold-start" (linha repetida) em vez de uma janela temporal real.
  Este benchmark preserva esse comportamento (não é escopo desta task
  corrigi-lo) e documenta o efeito no relatório.
* Threshold de decisão: ``ModelService.decision_threshold`` (lido do model
  card, mesmo valor tunado usado em produção).
* Buffer: tamanho/warmup lidos de ``SensorBuffer()`` (mesmos defaults do
  ``feature_buffer`` de produção: capacity=30, warmup=15) — não duplicados
  como números mágicos.

Dataset e slice de avaliação
-----------------------------
``apps/ml/data/processed/metropt3.parquet`` (dataset original — NÃO
alterado). O slice de avaliação é determinístico e reprodutível: localiza a
maior janela contígua de falha real (``anomaly == 1``) no dataset, mais um
contexto de operação normal imediatamente anterior (ordem temporal
preservada — obrigatório para as features de rolling/lag). Não há
embaralhamento, amostragem aleatória ou dados sintéticos.

Uso
---
    cd apps/backend
    python benchmark_models.py [--eval-normal N] [--eval-anomaly N] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import tracemalloc
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

_HERE: Path = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # garante `import src...` independente do cwd

from src.services.feature_buffer import SensorBuffer  # noqa: E402
from src.services.inference_pipeline import _infer_with_history  # noqa: E402
from src.services.model_registry import KNOWN_MODELS  # noqa: E402
from src.services.model_service import ModelService, load_model_by_name  # noqa: E402
from src.services.preprocessing import MetroPTPreprocessor  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes (RF-18 / RNF-36)
# ---------------------------------------------------------------------------

_ML_MODELS_DIR: Path = _HERE.parent / "ml" / "models"
_DATASET_PATH: Path = _HERE.parent / "ml" / "data" / "processed" / "metropt3.parquet"
_ENV_PATH: Path = _HERE.parent.parent / ".env"
_ENV_EXAMPLE_PATH: Path = _HERE.parent.parent / ".env.example"

_RAW_SENSOR_COLS: list[str] = [
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

_LATENCY_SLA_MS: float = 100.0  # RF-18
_PRE_CONTEXT_ROWS: int = 60  # aquece o buffer antes da 1a leitura avaliada
_DEFAULT_EVAL_NORMAL: int = 700
_DEFAULT_EVAL_ANOMALY: int = 1300
_SCORE_WEIGHTS: tuple[float, float, float] = (0.6, 0.25, 0.15)  # F1, latência, memória
_TIE_EPSILON: float = 0.01  # "praticamente equivalentes"


# ---------------------------------------------------------------------------
# Estruturas de resultado
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencyStats:
    """Estatísticas de latência de inferência (ms), medidas com ``time.perf_counter``."""

    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    n_samples: int


@dataclass
class ModelResult:
    """Resultado consolidado do benchmark para um modelo."""

    name: str
    f1: float
    latency: LatencyStats
    memory_mb: float
    decision_threshold: float
    n_eval: int
    eligible: bool
    score: float | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Métricas — funções puras (testadas isoladamente em test_benchmark.py)
# ---------------------------------------------------------------------------


def latency_stats(samples_ms: list[float]) -> LatencyStats:
    """Calcula média, p50, p95 e p99 a partir de amostras de latência em ms."""
    if not samples_ms:
        raise ValueError("samples_ms não pode ser vazio")
    arr = np.asarray(samples_ms, dtype=np.float64)
    return LatencyStats(
        mean_ms=float(np.mean(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        n_samples=len(arr),
    )


def is_eligible(p95_ms: float, sla_ms: float = _LATENCY_SLA_MS) -> bool:
    """RF-18 — elegível para produção somente se p95 < sla_ms."""
    return p95_ms < sla_ms


def compute_f1(y_true: list[int], y_pred: list[int]) -> float:
    """F1 da classe 1 (falha) — mesma métrica usada nos model cards de treino."""
    return float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))


def _normalize(values: list[float], higher_is_better: bool) -> list[float]:
    """Min-max normaliza para [0, 1]. Lista constante vira todos 1.0 (sem penalidade)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    if higher_is_better:
        return [(v - lo) / (hi - lo) for v in values]
    return [(hi - v) / (hi - lo) for v in values]  # menor valor -> 1.0


def composite_score(f1_norm: float, latency_norm: float, memory_norm: float) -> float:
    """
    Score simples e documentado (nenhuma fórmula de equilíbrio pré-existia no
    projeto): pesos 0.6/0.25/0.15 priorizam F1 (qualidade de detecção,
    critério RF-04 já usado nos model cards), depois latência (sensibilidade
    do pipeline SSE/alertas em tempo real), depois memória (recurso menos
    restrito hoje — ver limits do serviço `api` no docker-compose.yml).
    """
    w_f1, w_lat, w_mem = _SCORE_WEIGHTS
    return w_f1 * f1_norm + w_lat * latency_norm + w_mem * memory_norm


def select_winner(results: list[ModelResult]) -> ModelResult | None:
    """
    Aplica RF-18: filtra elegíveis (p95 < 100ms), pontua e escolhe o vencedor.

    Critério de desempate determinístico quando scores estão "praticamente
    equivalentes" (diferença < ``_TIE_EPSILON``): maior F1 -> menor p95 ->
    menor memória.

    Retorna ``None`` se nenhum modelo for elegível (nunca escolhe um modelo
    com p95 >= 100ms).
    """
    eligible = [r for r in results if r.eligible and r.error is None]
    if not eligible:
        return None

    f1_n = _normalize([r.f1 for r in eligible], higher_is_better=True)
    lat_n = _normalize([r.latency.p95_ms for r in eligible], higher_is_better=False)
    mem_n = _normalize([r.memory_mb for r in eligible], higher_is_better=False)

    scores: list[float] = []
    for r, fn, ln, mn in zip(eligible, f1_n, lat_n, mem_n):
        s = composite_score(fn, ln, mn)
        r.score = s
        scores.append(s)

    best_score = max(scores)
    contenders = [r for r, s in zip(eligible, scores) if best_score - s < _TIE_EPSILON]

    contenders.sort(key=lambda r: (-r.f1, r.latency.p95_ms, r.memory_mb))
    return contenders[0]


# ---------------------------------------------------------------------------
# Seleção do slice de avaliação — determinístico, sem embaralhar, sem dados sintéticos
# ---------------------------------------------------------------------------


def find_largest_anomaly_run(y: np.ndarray) -> tuple[int, int]:
    """
    Retorna (start, end) [end exclusivo] da maior sequência contígua de
    ``y == 1`` em um array temporalmente ordenado.
    """
    y = np.asarray(y).astype(int)
    if y.sum() == 0:
        raise ValueError("Nenhuma amostra positiva (anomaly=1) encontrada no dataset.")

    diff = np.diff(y)
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0] + 1)
    if y[0] == 1:
        starts = [0] + starts
    if y[-1] == 1:
        ends = ends + [len(y)]

    runs = list(zip(starts, ends))
    best = max(runs, key=lambda se: se[1] - se[0])
    return best


def build_eval_slice(
    raw_df: pd.DataFrame,
    pre_context: int = _PRE_CONTEXT_ROWS,
    eval_normal: int = _DEFAULT_EVAL_NORMAL,
    eval_anomaly: int = _DEFAULT_EVAL_ANOMALY,
) -> pd.DataFrame:
    """
    Monta um slice contíguo e determinístico do dataset original:

        [contexto normal p/ aquecer rolling stats]
        [eval_normal linhas rotuladas 0]
        [eval_anomaly linhas rotuladas 1, início da maior falha real]

    A ordem temporal do dataset é preservada (obrigatório para
    std/ma/lag/roc/min/max). Nenhuma linha é sintética ou embaralhada.
    """
    run_start, run_end = find_largest_anomaly_run(raw_df["anomaly"].to_numpy())

    normal_start = run_start - pre_context - eval_normal
    if normal_start < 0:
        raise ValueError(
            "Dataset curto demais para o slice pedido "
            f"(precisa de {pre_context + eval_normal} linhas normais antes de {run_start})."
        )
    anomaly_end = min(run_end, run_start + eval_anomaly)
    if anomaly_end - run_start < eval_anomaly:
        log.warning(
            "Janela de falha real (%d linhas) menor que eval_anomaly pedido (%d); usando %d.",
            run_end - run_start,
            eval_anomaly,
            anomaly_end - run_start,
        )

    return raw_df.iloc[normal_start:anomaly_end].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Execução do benchmark por modelo
# ---------------------------------------------------------------------------


def _load_with_memory(model_name: str) -> tuple[ModelService, float]:
    """Carrega o modelo via o factory de produção, medindo memória com tracemalloc."""
    tracemalloc.start()
    try:
        service = load_model_by_name(model_name)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return service, peak / (1024 * 1024)


def run_model_benchmark(
    model_name: str,
    eval_slice: pd.DataFrame,
    buffer_window: int,
    n_context: int,
) -> ModelResult:
    """
    Avalia um único modelo sob as mesmas condições dos demais:

    * Carrega via ``load_model_by_name`` (produção) — memória via tracemalloc.
    * Para cada linha avaliada, replica o buffer real (últimas
      ``buffer_window`` leituras) e chama ``_infer_with_history`` — a mesma
      função de produção — medindo latência com ``time.perf_counter``.
    * F1 sobre as predições binárias vs. o rótulo real ``anomaly``.

    Qualquer exceção (artefato ausente, incompatibilidade de versão de lib,
    etc.) é capturada e vira um ``ModelResult`` de erro/não-elegível — um
    modelo quebrado nunca derruba o benchmark inteiro (RNF-36: o script
    precisa terminar e reportar todos os candidatos em uma única execução).
    """
    try:
        service, memory_mb = _load_with_memory(model_name)

        preprocessor = MetroPTPreprocessor()
        latencies_ms: list[float] = []
        y_true: list[int] = []
        y_pred: list[int] = []

        # Silencia warnings (ex.: UserWarning do backend paralelo do
        # RandomForest) durante a seção cronometrada — impressão em stderr a
        # cada chamada inflaria artificialmente a latência medida e
        # quebraria "mesmas condições" entre os modelos.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(n_context, len(eval_slice)):
                window_start = max(0, i - buffer_window + 1)
                buffer_df = eval_slice.iloc[window_start : i + 1][
                    _RAW_SENSOR_COLS
                ].reset_index(drop=True)

                t0 = time.perf_counter()
                result = _infer_with_history(service, buffer_df, preprocessor)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                latencies_ms.append(elapsed_ms)
                y_true.append(int(eval_slice.iloc[i]["anomaly"]))
                y_pred.append(result.predicted_class)

        stats = latency_stats(latencies_ms)
        f1 = compute_f1(y_true, y_pred)
        eligible = is_eligible(stats.p95_ms)

        return ModelResult(
            name=model_name,
            f1=f1,
            latency=stats,
            memory_mb=memory_mb,
            decision_threshold=service.decision_threshold,
            n_eval=len(latencies_ms),
            eligible=eligible,
        )
    except Exception as exc:  # noqa: BLE001 — isola falha de 1 candidato (RNF-36)
        log.exception("Falha ao avaliar '%s'", model_name)
        empty = LatencyStats(0.0, 0.0, 0.0, 0.0, 0)
        return ModelResult(
            name=model_name,
            f1=0.0,
            latency=empty,
            memory_mb=0.0,
            decision_threshold=0.0,
            n_eval=0,
            eligible=False,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Atualização de configuração — ACTIVE_MODEL é a única forma existente (RF-10)
# ---------------------------------------------------------------------------


def update_env_active_model(env_path: Path, model_name: str) -> bool:
    """
    Atualiza (ou adiciona) a linha ``ACTIVE_MODEL=<model_name>`` em um arquivo
    .env preservando todas as outras linhas/segredos inalterados.

    Retorna True se o arquivo foi alterado, False se não existir.
    """
    if not env_path.exists():
        log.warning(
            "%s não encontrado — pulando atualização de ACTIVE_MODEL.", env_path
        )
        return False

    text = env_path.read_text(encoding="utf-8")
    pattern = re.compile(r"^ACTIVE_MODEL=.*$", re.MULTILINE)
    new_line = f"ACTIVE_MODEL={model_name}"

    if pattern.search(text):
        text = pattern.sub(new_line, text, count=1)
    else:
        sep = "" if text.endswith("\n") or not text else "\n"
        text = f"{text}{sep}{new_line}\n"

    env_path.write_text(text, encoding="utf-8")
    log.info("%s -> ACTIVE_MODEL=%s", env_path, model_name)
    return True


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------


def _fmt(v: float, digits: int = 4) -> str:
    return f"{v:.{digits}f}"


def generate_report(
    results: list[ModelResult],
    winner: ModelResult | None,
    out_path: Path,
    n_eval_total: int,
) -> None:
    lines: list[str] = []
    lines.append("# Benchmark de Modelos de Produção — PredictIQ\n")
    lines.append(
        "Gerado automaticamente por `python benchmark_models.py`. Reproduzível: "
        "execute o mesmo comando para regenerar este relatório com os mesmos dados.\n"
    )

    lines.append("## Resumo executivo\n")
    if winner is None:
        lines.append(
            "**Nenhum modelo elegível.** Todos os candidatos violaram RF-18 "
            f"(p95 < {_LATENCY_SLA_MS:.0f}ms) ou falharam ao carregar. "
            "`ACTIVE_MODEL` NÃO foi alterado.\n"
        )
    else:
        lines.append(f"- **Modelo vencedor:** `{winner.name}`")
        lines.append(f"- **F1 (classe 1):** {_fmt(winner.f1)}")
        lines.append(f"- **Latência p95:** {_fmt(winner.latency.p95_ms, 3)} ms")
        lines.append(
            f"- **Memória (tracemalloc, carregamento):** {_fmt(winner.memory_mb, 2)} MB"
        )
        lines.append(
            "- **Justificativa:** melhor equilíbrio (score="
            f"{_fmt(winner.score or 0.0, 4)}) entre F1, latência e memória "
            f"dentre os modelos elegíveis (p95 < {_LATENCY_SLA_MS:.0f}ms).\n"
        )

    lines.append("## Tabela comparativa\n")
    lines.append(
        "| Modelo | F1 | Latência média (ms) | p50 (ms) | p95 (ms) | p99 (ms) | "
        "Memória (MB) | Elegível | Resultado |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|")
    for r in sorted(results, key=lambda x: (not x.eligible, -(x.score or -1))):
        if r.error:
            lines.append(
                f"| `{r.name}` | — | — | — | — | — | — | ❌ | Erro: {r.error} |"
            )
            continue
        elig = "✅" if r.eligible else "❌"
        outcome = (
            "🏆 Vencedor"
            if winner is not None and r.name == winner.name
            else (
                "Elegível"
                if r.eligible
                else f"Não elegível (p95 ≥ {_LATENCY_SLA_MS:.0f}ms)"
            )
        )
        lines.append(
            f"| `{r.name}` | {_fmt(r.f1)} | {_fmt(r.latency.mean_ms, 3)} | "
            f"{_fmt(r.latency.p50_ms, 3)} | {_fmt(r.latency.p95_ms, 3)} | "
            f"{_fmt(r.latency.p99_ms, 3)} | {_fmt(r.memory_mb, 2)} | {elig} | {outcome} |"
        )
    lines.append("")

    valid = [r for r in results if r.error is None]
    lines.append("## Análise\n")
    if valid:

        def _tied_names(key_fn: Callable[[ModelResult], float], best_val: float) -> str:
            names = [f"`{r.name}`" for r in valid if key_fn(r) == best_val]
            return (
                " (empate com " + ", ".join(names[1:]) + ")" if len(names) > 1 else ""
            )

        best_f1_val = max(r.f1 for r in valid)
        best_lat_val = min(r.latency.p95_ms for r in valid)
        best_mem_val = min(r.memory_mb for r in valid)
        best_f1 = next(r for r in valid if r.f1 == best_f1_val)
        best_lat = next(r for r in valid if r.latency.p95_ms == best_lat_val)
        best_mem = next(r for r in valid if r.memory_mb == best_mem_val)
        eliminated = [r.name for r in valid if not r.eligible]
        lines.append(
            f"- Melhor F1: `{best_f1.name}` ({_fmt(best_f1.f1)})"
            f"{_tied_names(lambda r: r.f1, best_f1_val)}."
        )
        lines.append(
            f"- Menor latência p95: `{best_lat.name}` ({_fmt(best_lat.latency.p95_ms, 3)} ms)"
            f"{_tied_names(lambda r: r.latency.p95_ms, best_lat_val)}."
        )
        lines.append(
            f"- Menor memória: `{best_mem.name}` ({_fmt(best_mem.memory_mb, 2)} MB)"
            f"{_tied_names(lambda r: r.memory_mb, best_mem_val)}."
        )
        if eliminated:
            lines.append(
                f"- Eliminados pelo limite de {_LATENCY_SLA_MS:.0f}ms (RF-18): "
                f"{', '.join(f'`{n}`' for n in eliminated)}."
            )
        else:
            lines.append(f"- Nenhum modelo violou o limite de {_LATENCY_SLA_MS:.0f}ms.")
        if winner is not None:
            lines.append(
                f"\n`{winner.name}` representa o melhor equilíbrio: entre os modelos "
                "elegíveis, combina F1, p95 e memória segundo o score documentado em "
                "`composite_score()` (pesos 0.6 F1 / 0.25 latência / 0.15 memória)."
            )
        lines.append(
            "\n**Nota metodológica:** modelos sequenciais (TCN, BiLSTM, PatchTST) e o "
            "autoencoder são avaliados com o mesmo caminho de inferência real do "
            "PredictIQ (`InferencePipelineService._infer_with_history`), que hoje passa "
            "apenas a última leitura para `predict_from_features` — a janela temporal "
            "que esses modelos recebem em produção é, portanto, uma repetição "
            "cold-start da última leitura, não uma janela histórica real. Isso é uma "
            "característica já existente do sistema (não introduzida por este "
            "benchmark) e penaliza o F1 desses modelos aqui de forma consistente com "
            "o que acontece em produção hoje."
        )
    lines.append("")

    lines.append("## Critério\n")
    lines.append(
        f"**RF-18** — o modelo vencedor deve ter latência p95 < {_LATENCY_SLA_MS:.0f}ms. "
        "Modelos que violam essa condição são marcados não elegíveis e nunca são "
        "escolhidos, independentemente do F1. Entre os elegíveis, vence o melhor "
        "equilíbrio F1/latência/memória (`select_winner()`), com desempate "
        "determinístico (maior F1 → menor p95 → menor memória) quando os scores "
        f"diferem por menos de {_TIE_EPSILON}.\n"
    )
    lines.append(
        "**RNF-36** — todo este processo (carregar modelos, medir, selecionar, "
        "gerar relatório, atualizar `ACTIVE_MODEL`) executa com um único comando: "
        "`python benchmark_models.py`.\n"
    )
    lines.append(
        f"Slice de avaliação: {n_eval_total} leituras (dataset original "
        "`apps/ml/data/processed/metropt3.parquet`, ordem temporal preservada, "
        "sem embaralhar, sem dados sintéticos) — contexto normal + a maior janela "
        "de falha real contígua do dataset.\n"
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Relatório salvo -> %s", out_path)


# ---------------------------------------------------------------------------
# Orquestração (RNF-36 — comando único)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-normal", type=int, default=_DEFAULT_EVAL_NORMAL)
    parser.add_argument("--eval-anomaly", type=int, default=_DEFAULT_EVAL_ANOMALY)
    parser.add_argument("--out", type=Path, default=_HERE)
    parser.add_argument(
        "--skip-config-update",
        action="store_true",
        help="Não escreve ACTIVE_MODEL em .env/.env.example (só gera o relatório).",
    )
    args = parser.parse_args(argv)

    if not _DATASET_PATH.exists():
        log.error("Dataset não encontrado: %s", _DATASET_PATH)
        return 1

    log.info("Carregando dataset %s ...", _DATASET_PATH)
    raw_df = pd.read_parquet(_DATASET_PATH)

    buffer_defaults = (
        SensorBuffer()
    )  # mesmos defaults de produção (capacity=30, warmup=15)
    eval_slice = build_eval_slice(
        raw_df, eval_normal=args.eval_normal, eval_anomaly=args.eval_anomaly
    )
    log.info(
        "Slice de avaliação: %d linhas (%d normais + contexto, %d de falha real)",
        len(eval_slice),
        args.eval_normal + _PRE_CONTEXT_ROWS,
        args.eval_anomaly,
    )

    candidates = sorted(KNOWN_MODELS)
    results: list[ModelResult] = []
    for name in candidates:
        log.info("=" * 60)
        log.info("Avaliando: %s", name)
        result = run_model_benchmark(
            name,
            eval_slice,
            buffer_window=buffer_defaults.capacity,
            n_context=_PRE_CONTEXT_ROWS,
        )
        results.append(result)
        if result.error:
            log.warning("  %s: ERRO — %s", name, result.error)
        else:
            log.info(
                "  %s: F1=%.4f p95=%.2fms mem=%.2fMB elegível=%s",
                name,
                result.f1,
                result.latency.p95_ms,
                result.memory_mb,
                result.eligible,
            )

    winner = select_winner(results)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "benchmark_report.md"
    generate_report(results, winner, report_path, len(eval_slice) - _PRE_CONTEXT_ROWS)

    results_json = out_dir / "benchmark_results.json"
    results_json.write_text(
        json.dumps(
            [
                {
                    "name": r.name,
                    "f1": r.f1,
                    "latency_mean_ms": r.latency.mean_ms,
                    "latency_p50_ms": r.latency.p50_ms,
                    "latency_p95_ms": r.latency.p95_ms,
                    "latency_p99_ms": r.latency.p99_ms,
                    "memory_mb": r.memory_mb,
                    "eligible": r.eligible,
                    "score": r.score,
                    "error": r.error,
                }
                for r in results
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    if winner is None:
        log.error(
            "RF-18: nenhum modelo elegível (p95 < %.0fms). ACTIVE_MODEL não foi alterado.",
            _LATENCY_SLA_MS,
        )
        return 1

    log.info(
        "VENCEDOR: %s | F1=%.4f | p95=%.2fms | mem=%.2fMB | score=%.4f",
        winner.name,
        winner.f1,
        winner.latency.p95_ms,
        winner.memory_mb,
        winner.score,
    )

    if not args.skip_config_update:
        update_env_active_model(_ENV_PATH, winner.name)
        if _ENV_EXAMPLE_PATH.exists():
            update_env_active_model(_ENV_EXAMPLE_PATH, winner.name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
