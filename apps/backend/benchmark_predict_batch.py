"""
Benchmark HTTP real — RNF-73 Fase 8: POST /predict/batch (100 amostras).

Mede, contra a API REAL rodando (não chamada Python interna):

  A. ``POST /predict/batch`` com 100 amostras — latência, p50/p95/p99,
     throughput (amostras/s efetivo).
  B. Comparação: tempo de parede para obter 100 predições via
     ``100 x POST /predict/`` (respeitando RNF-19, 100/min) vs.
     ``1 x POST /predict/batch`` com 100 amostras (respeitando RNF-73,
     20/min) — demonstra que o endpoint de fato aproveita inferência batch
     (não é um loop disfarçado).

Rate limits respeitados, nunca contornados (RNF-72/73 §16): single =
100/min (RNF-19), batch = 20/min (RNF-73, src/core/rate_limit.py). O script
pausa entre requisições para nunca estourar o limite configurado.

Uso
---
    python benchmark_predict_batch.py --host http://localhost:8000 \\
           --n-batch-requests 30
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log: logging.Logger = logging.getLogger(__name__)

_BASE_PAYLOAD: dict[str, float] = {
    "TP2": 5.02,
    "TP3": 9.21,
    "H1": 8.97,
    "DV_pressure": 2.10,
    "Reservoirs": 8.85,
    "Motor_current": 4.5,
    "Oil_temperature": 72.3,
    "COMP": 1.0,
    "DV_eletric": 0.0,
    "Towers": 1.0,
    "MPG": 1.0,
    "Oil_level": 1.0,
}


def _sample(seed_offset: float) -> dict[str, float]:
    p = dict(_BASE_PAYLOAD)
    p["TP2"] = round(p["TP2"] + seed_offset, 4)
    return p


@dataclass(frozen=True)
class LatencyStats:
    n_samples: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    failures: int


def latency_stats(samples_ms: list[float], failures: int) -> LatencyStats:
    arr = np.asarray(samples_ms, dtype=np.float64)
    return LatencyStats(
        n_samples=len(arr),
        mean_ms=float(np.mean(arr)) if len(arr) else 0.0,
        p50_ms=float(np.percentile(arr, 50)) if len(arr) else 0.0,
        p95_ms=float(np.percentile(arr, 95)) if len(arr) else 0.0,
        p99_ms=float(np.percentile(arr, 99)) if len(arr) else 0.0,
        min_ms=float(np.min(arr)) if len(arr) else 0.0,
        max_ms=float(np.max(arr)) if len(arr) else 0.0,
        failures=failures,
    )


def bench_predict_batch(
    client: httpx.Client, host: str, n_requests: int, batch_size: int, wait_s: float
) -> LatencyStats:
    url = f"{host}/predict/batch"
    latencies_ms: list[float] = []
    failures = 0

    # warm-up — não contado
    warmup_payload = {"samples": [_sample(i * 0.01) for i in range(batch_size)]}
    client.post(url, json=warmup_payload, timeout=30.0)
    time.sleep(wait_s)

    for i in range(n_requests):
        payload = {
            "samples": [_sample(i * 0.01 + j * 0.001) for j in range(batch_size)]
        }
        t0 = time.perf_counter()
        try:
            resp = client.post(url, json=payload, timeout=30.0)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if resp.status_code != 200:
                failures += 1
                log.warning("batch request %d: HTTP %d", i, resp.status_code)
            else:
                latencies_ms.append(elapsed_ms)
        except Exception:
            failures += 1
            log.exception("batch request %d falhou", i)
        time.sleep(wait_s)

    return latency_stats(latencies_ms, failures)


def bench_single_x100(
    client: httpx.Client, host: str, wait_s: float
) -> dict[str, float]:
    """Tempo de parede pra 100 predições via 100x POST /predict/ (respeita
    100/min, RNF-19)."""
    url = f"{host}/predict/"
    t_start = time.perf_counter()
    failures = 0
    for i in range(100):
        try:
            resp = client.post(url, json=_sample(i * 0.01), timeout=30.0)
            if resp.status_code != 200:
                failures += 1
        except Exception:
            failures += 1
        time.sleep(wait_s)
    total_s = time.perf_counter() - t_start
    return {"total_s": total_s, "failures": failures}


def bench_batch_x1_of_100(client: httpx.Client, host: str) -> dict[str, float]:
    """Tempo de parede pra 100 predições via 1x POST /predict/batch(100)."""
    url = f"{host}/predict/batch"
    payload = {"samples": [_sample(i * 0.01) for i in range(100)]}
    t_start = time.perf_counter()
    resp = client.post(url, json=payload, timeout=30.0)
    total_s = time.perf_counter() - t_start
    return {"total_s": total_s, "failures": 0 if resp.status_code == 200 else 1}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--n-batch-requests", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)

    with httpx.Client() as client:
        log.info("=" * 60)
        log.info(
            "A. POST /predict/batch (100 amostras) x %d requisições ...",
            args.n_batch_requests,
        )
        # 20/min (RNF-73) => >=3s entre requests para nunca estourar.
        batch_stats = bench_predict_batch(
            client,
            args.host,
            n_requests=args.n_batch_requests,
            batch_size=100,
            wait_s=3.05,
        )
        log.info(
            "   mean=%.2fms p50=%.2fms p95=%.2fms p99=%.2fms failures=%d",
            batch_stats.mean_ms,
            batch_stats.p50_ms,
            batch_stats.p95_ms,
            batch_stats.p99_ms,
            batch_stats.failures,
        )

        log.info("=" * 60)
        log.info("B1. 100x POST /predict/ (sequencial, respeitando 100/min) ...")
        single_x100 = bench_single_x100(client, args.host, wait_s=0.62)
        log.info(
            "    tempo total: %.2fs | falhas: %d",
            single_x100["total_s"],
            single_x100["failures"],
        )

        log.info("=" * 60)
        log.info("B2. 1x POST /predict/batch com 100 amostras ...")
        time.sleep(3.5)  # respeita o limite de 20/min após as medições acima
        batch_x1 = bench_batch_x1_of_100(client, args.host)
        log.info(
            "    tempo total: %.4fs | falhas: %d",
            batch_x1["total_s"],
            batch_x1["failures"],
        )

    results = {
        "predict_batch_100_samples": asdict(batch_stats),
        "comparison_100_predictions": {
            "100x_predict_sequential": single_x100,
            "1x_predict_batch_100": batch_x1,
            "speedup_wall_time": (
                single_x100["total_s"] / batch_x1["total_s"]
                if batch_x1["total_s"] > 0
                else None
            ),
        },
    }

    out_path = args.out / "benchmark_predict_batch_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Resultados salvos -> %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
