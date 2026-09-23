"""
Load test — POST /predict/batch (RNF-73), 100 amostras por requisição.

Respeita o rate limit real do endpoint (20/min, RNF-73,
src/core/rate_limit.py::PREDICT_BATCH_RATE_LIMIT) — nunca contornado.
``PREDICT_BATCH_WAIT_SECONDS`` (default 3.05s) mantém o agregado abaixo de
20/min mesmo com múltiplos usuários virtuais (o limite é por IP, compartilhado
entre todos — mesmo raciocínio de ``locust_predict.py``).

Uso
---
    pip install locust
    locust -f locust_predict_batch.py --host http://localhost:8000 \\
           --headless -u 1 -r 1 --run-time 90s --csv=predict-batch

Veredito impresso ao final: requests, failures, p50/p95/p99, throughput em
AMOSTRAS/s (não só requisições/s — cada requisição processa 100 amostras).
"""

from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from locust import HttpUser, between, events
from locust import task as locust_task

_PATH: str = os.environ.get("PREDICT_BATCH_PATH", "/predict/batch")
_WAIT_SECONDS: float = float(os.environ.get("PREDICT_BATCH_WAIT_SECONDS", "3.05"))
_BATCH_SIZE: int = int(os.environ.get("PREDICT_BATCH_SIZE", "100"))

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


def _batch_payload(size: int) -> dict:
    samples = []
    for i in range(size):
        s = dict(_BASE_PAYLOAD)
        s["TP2"] = round(_BASE_PAYLOAD["TP2"] + (i % 20) * 0.05, 4)
        samples.append(s)
    return {"samples": samples}


class PredictBatchUser(HttpUser):
    wait_time = between(_WAIT_SECONDS, _WAIT_SECONDS)

    @locust_task
    def predict_batch(self) -> None:
        self.client.post(
            _PATH,
            json=_batch_payload(_BATCH_SIZE),
            name=f"{_PATH} ({_BATCH_SIZE} samples)",
        )


@events.test_stop.add_listener
def _print_rnf73_verdict(environment, **kwargs) -> None:  # type: ignore[no-untyped-def]
    name = f"{_PATH} ({_BATCH_SIZE} samples)"
    stats = environment.runner.stats.get(name, "POST")
    if stats is None or stats.num_requests == 0:
        print(f"\nRNF-73: sem amostras de '{name}' — verifique o host/endpoint.")
        return

    p50 = stats.get_response_time_percentile(0.50)
    p95 = stats.get_response_time_percentile(0.95)
    p99 = stats.get_response_time_percentile(0.99)
    total_samples_processed = stats.num_requests * _BATCH_SIZE

    print(f"\n{'=' * 60}")
    print(f"RNF-73 — POST /predict/batch ({_BATCH_SIZE} amostras/requisição)")
    print(f"{'=' * 60}")
    print(f"  Requests (batches)     : {stats.num_requests}")
    print(f"  Amostras processadas   : {total_samples_processed}")
    print(f"  Failures               : {stats.num_failures}")
    print(f"  p50                    : {p50:.1f} ms")
    print(f"  p95                    : {p95:.1f} ms")
    print(f"  p99                    : {p99:.1f} ms")
    print(f"  Resultado              : {'PASS' if stats.num_failures == 0 else 'FAIL'}")
    print(f"{'=' * 60}\n")
