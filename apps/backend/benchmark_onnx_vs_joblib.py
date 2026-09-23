"""
Benchmark Joblib vs ONNX — RNF-72 (single prediction) / auditoria Fase 1 e 7.

Objetivo
--------
Medir, sob as MESMAS condições (mesmos dados de entrada reais do MetroPT-3,
mesmo preprocessing de produção — ``ModelService._build_feature_row``, mesma
máquina, mesmo número de execuções, mesmo warm-up), a latência de UMA
predição via:

  * Joblib  — ``sklearn.ensemble.RandomForestClassifier`` / ``xgboost.XGBClassifier``
    carregados via ``joblib.load`` (random_forest / xgboost).
  * ONNX    — ``OnnxTreeAdapter`` sobre ONNX Runtime, CPU execution provider
    (random_forest_v2 / xgboost_v2).

Achado de auditoria (documentado, não escondido): ``random_forest_v2.onnx``/
``xgboost_v2.onnx`` NÃO são modelos diferentes dos `.joblib` — são o MESMO
modelo treinado, exportado nos dois formatos pela MESMA execução de
``train_random_forest.py``/``train_xgboost.py`` (mesmo ``model_card.json``,
mesmo ``feature_count``/``feature_names``). Não há conversão em runtime
nesta task — os artefatos ONNX já existem, versionados em
``apps/ml/models/``.

Achado de auditoria #2 — XGBoost V1 (joblib) foi treinado em ndarray puro,
SEM nomes de coluna (``train_xgboost.py`` linha ~337: exigência do
conversor ONNX onnxmltools). Por isso ``xgboost_v1.joblib`` não tem
``feature_names_in_`` e ``ModelService(model=...)`` não consegue instanciá-lo
normalmente (``AttributeError``) — reproduzido também pelo
``benchmark_models.py`` já existente no projeto (ver seu ``benchmark_results.json``:
``"xgboost"`` aparece com ``error``). Este script contorna ISSO SÓ para
poder medir o baseline Joblib do XGBoost: carrega o modelo via
``joblib.load`` e ordena as colunas manualmente usando
``xgboost_v1_card.json::feature_names`` (a mesma ordem gravada no
treinamento) — não reimplementa preprocessing, reusa
``ModelService._build_feature_row`` (função pura, não depende de
``self._model``) para gerar as features a partir do request real.

Uso
---
    cd apps/backend
    python benchmark_onnx_vs_joblib.py [--n-samples 300] [--warmup 20] [--out .]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

_HERE: Path = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from src.core.config import settings  # noqa: E402
from src.schemas.predict import PredictRequest  # noqa: E402
from src.services.model_service import load_model_by_name  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log: logging.Logger = logging.getLogger(__name__)

_DATASET_PATH: Path = _HERE.parent / "ml" / "data" / "processed" / "metropt3.parquet"
_MODELS_DIR: Path = _HERE.parent / "ml" / "models"

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


@dataclass(frozen=True)
class LatencyStats:
    n_samples: int
    warmup: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    total_s: float
    throughput_per_s: float


def latency_stats(samples_ms: list[float], warmup: int) -> LatencyStats:
    arr = np.asarray(samples_ms, dtype=np.float64)
    total_s = float(arr.sum() / 1000.0)
    return LatencyStats(
        n_samples=len(arr),
        warmup=warmup,
        mean_ms=float(np.mean(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        total_s=total_s,
        throughput_per_s=float(len(arr) / total_s) if total_s > 0 else 0.0,
    )


# ---------------------------------------------------------------------------
# Amostras reais — sensores brutos do dataset original, sem sintético
# ---------------------------------------------------------------------------


def load_real_requests(n_samples: int) -> list[PredictRequest]:
    """
    ``n_samples`` linhas reais, igualmente espaçadas ao longo do dataset
    original (mistura leituras normais e de falha, sem embaralhar, sem
    amostragem aleatória — determinístico e reproduzível).
    """
    df = pd.read_parquet(_DATASET_PATH, columns=_RAW_SENSOR_COLS)
    idx = np.linspace(0, len(df) - 1, num=n_samples, dtype=int)
    rows = df.iloc[idx].reset_index(drop=True)
    return [
        PredictRequest(**{col: float(row[col]) for col in _RAW_SENSOR_COLS})
        for _, row in rows.iterrows()
    ]


# ---------------------------------------------------------------------------
# Execução cronometrada — mesmo padrão para todos os caminhos
# ---------------------------------------------------------------------------


def run_timed(
    predict_one: Callable[[PredictRequest], float],
    requests: list[PredictRequest],
    warmup: int,
) -> LatencyStats:
    """
    ``predict_one`` recebe um PredictRequest e devolve failure_probability
    (usado só para forçar a leitura do resultado — evita eliminação por
    otimização/lazy-eval e serve de prova de que a chamada realmente rodou).
    Warm-up ANTES da medição (não contado) — evita contaminar por
    carregamento/JIT/cache frio da sessão ONNX.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for req in requests[:warmup]:
            predict_one(req)

        latencies_ms: list[float] = []
        for req in requests:
            t0 = time.perf_counter()
            predict_one(req)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    return latency_stats(latencies_ms, warmup=warmup)


# ---------------------------------------------------------------------------
# Caminhos Joblib / ONNX por modelo
# ---------------------------------------------------------------------------


def _rf_joblib_predict_one() -> Callable[[PredictRequest], float]:
    service = load_model_by_name("random_forest")  # ModelService(joblib RF)

    def _predict(req: PredictRequest) -> float:
        return service.predict(req).failure_probability

    return _predict


def _rf_onnx_predict_one() -> Callable[[PredictRequest], float]:
    service = load_model_by_name("random_forest_v2")  # ModelService(OnnxTreeAdapter)

    def _predict(req: PredictRequest) -> float:
        return service.predict(req).failure_probability

    return _predict


def _xgb_onnx_predict_one() -> Callable[[PredictRequest], float]:
    service = load_model_by_name("xgboost_v2")  # ModelService(OnnxTreeAdapter)

    def _predict(req: PredictRequest) -> float:
        return service.predict(req).failure_probability

    return _predict


def _xgb_joblib_predict_one() -> Callable[[PredictRequest], float]:
    """
    Contorno documentado (ver docstring do módulo): xgboost_v1.joblib não
    tem `feature_names_in_`, então ModelService não o instancia. Carrega o
    modelo bruto + reusa `ModelService._build_feature_row` (função pura) do
    RF (mesma lógica, não depende do modelo) para gerar as features reais,
    depois reordena pelas colunas do card (mesma ordem gravada no treino) e
    chama `.predict_proba` direto no XGBClassifier — o MESMO objeto treinado
    que o benchmark_models.py tenta (e falha) carregar via ModelService.
    """
    import joblib

    raw_model = joblib.load(settings.xgboost_model_path)
    card = json.loads(
        (_MODELS_DIR / "xgboost_v1_card.json").read_text(encoding="utf-8")
    )
    feature_names: list[str] = card["feature_names"]

    # Reusa a construção de features de produção (pure function) via uma
    # instância RF já funcional — `_build_feature_row` não usa `self._model`.
    rf_service = load_model_by_name("random_forest")

    def _predict(req: PredictRequest) -> float:
        X = rf_service._build_feature_row(req)
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[feature_names]
        proba = raw_model.predict_proba(X.to_numpy(dtype=np.float32))
        return float(proba[0][1])

    return _predict


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--out", type=Path, default=_HERE)
    args = parser.parse_args(argv)

    if not _DATASET_PATH.exists():
        log.error("Dataset não encontrado: %s", _DATASET_PATH)
        return 1

    log.info("Carregando %d amostras reais do dataset ...", args.n_samples)
    requests = load_real_requests(args.n_samples)

    builders: dict[str, Callable[[], Callable[[PredictRequest], float]]] = {
        "random_forest (joblib)": _rf_joblib_predict_one,
        "random_forest_v2 (onnx)": _rf_onnx_predict_one,
        "xgboost (joblib)": _xgb_joblib_predict_one,
        "xgboost_v2 (onnx)": _xgb_onnx_predict_one,
    }

    results: dict[str, Any] = {}
    for name, build in builders.items():
        log.info("=" * 60)
        log.info("Medindo: %s", name)
        try:
            predict_one = build()
            stats = run_timed(predict_one, requests, warmup=args.warmup)
            results[name] = asdict(stats)
            log.info(
                "  %s: mean=%.4fms p50=%.4fms p95=%.4fms p99=%.4fms throughput=%.1f/s",
                name,
                stats.mean_ms,
                stats.p50_ms,
                stats.p95_ms,
                stats.p99_ms,
                stats.throughput_per_s,
            )
        except Exception as exc:  # noqa: BLE001 — isola falha de 1 candidato
            log.exception("Falha ao medir '%s'", name)
            results[name] = {"error": str(exc)}

    def _speedup(joblib_key: str, onnx_key: str, field: str) -> float | None:
        j, o = results.get(joblib_key), results.get(onnx_key)
        if not j or not o or "error" in j or "error" in o:
            return None
        return j[field] / o[field] if o[field] > 0 else None

    speedups = {
        "random_forest": {
            "speedup_mean": _speedup(
                "random_forest (joblib)", "random_forest_v2 (onnx)", "mean_ms"
            ),
            "speedup_p50": _speedup(
                "random_forest (joblib)", "random_forest_v2 (onnx)", "p50_ms"
            ),
            "speedup_p95": _speedup(
                "random_forest (joblib)", "random_forest_v2 (onnx)", "p95_ms"
            ),
        },
        "xgboost": {
            "speedup_mean": _speedup(
                "xgboost (joblib)", "xgboost_v2 (onnx)", "mean_ms"
            ),
            "speedup_p50": _speedup("xgboost (joblib)", "xgboost_v2 (onnx)", "p50_ms"),
            "speedup_p95": _speedup("xgboost (joblib)", "xgboost_v2 (onnx)", "p95_ms"),
        },
    }

    out_path = args.out / "benchmark_onnx_vs_joblib_results.json"
    out_path.write_text(
        json.dumps({"results": results, "speedups": speedups}, indent=2),
        encoding="utf-8",
    )
    log.info("Resultados salvos -> %s", out_path)

    log.info("=" * 60)
    for model, sp in speedups.items():
        log.info("%s speedup: %s", model, sp)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
