"""
Load test — POST /predict (RNF-70/RNF-71).

Dois modos, mesma requisição real de sempre (sem mock, sem atalho):

1. PREDICT_MODE=random (default) — cada requisição usa um payload novo
   (sensores aleatórios dentro da faixa real do MetroPT-3). Mede a API como
   ela se comporta HOJE, sem nenhum cache-hit — usado como baseline "antes"
   (Fase 2) e continua valendo "depois" como referência de tráfego que nunca
   repete (cache sempre MISS por design de request, não por cache
   desabilitado).

2. PREDICT_MODE=cache_hit — todo usuário virtual reenvia o MESMO payload
   fixo (chave de cache idêntica). Um listener `test_start` faz 1 requisição
   de "priming" ANTES do load começar (fora das estatísticas do Locust) para
   garantir cache MISS -> SET já resolvido; a partir daí, toda requisição
   medida pelo Locust já é esperada como HIT. Não mistura miss/hit no mesmo
   número (RNF-70 §10).

Uso
---
    pip install locust
    # baseline (Fase 2, sem cache):
    locust -f locust_predict.py --host http://localhost:8000 \\
           --headless -u 20 -r 5 --run-time 30s --csv=predict-baseline

    # RNF-70 (Fase 5, cache-hit):
    PREDICT_MODE=cache_hit \\
    locust -f locust_predict.py --host http://localhost:8000 \\
           --headless -u 20 -r 5 --run-time 30s --csv=predict-cache-hit

O veredito RNF-70 (p95 < 50ms) é impresso automaticamente ao final de uma
run headless em PREDICT_MODE=cache_hit (hook `test_stop`).

Rate limit (RNF-19)
-------------------
POST /predict/ é limitado a 100 req/min POR IP (slowapi, ver
src/core/rate_limit.py::PREDICT_RATE_LIMIT) — o Locust bate de UM único IP
(container/processo), então o limite é COMPARTILHADO por todos os usuários
virtuais desta run, não por-usuário. Este script NUNCA tenta contornar isso
(múltiplos IPs falsos, header spoofing, etc.) — respeitar o rate limit
existente é requisito (RNF-70 §16 "não fabricar cache hit artificial").
PREDICT_WAIT_SECONDS controla o intervalo entre requisições de cada usuário
virtual para manter o agregado abaixo de 100/min; o padrão (0) é para testes
de CONCORRÊNCIA/contenção onde estourar 429 é o comportamento esperado e
observável (ver RELATORIO-RNF-70-RNF-71.md).

Configuração (env vars, todas opcionais)
------------------------------------------
    PREDICT_MODE          "random" (default) ou "cache_hit"
    PREDICT_PATH          path do endpoint (default /predict/)
    PREDICT_WAIT_SECONDS  espera fixa (s) entre requisições de cada
                           usuário virtual (default 0)
"""

from __future__ import annotations

import os
import random
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import httpx
from locust import HttpUser, between, events
from locust import task as locust_task

_MODE: str = os.environ.get("PREDICT_MODE", "random").strip().lower()
_PATH: str = os.environ.get("PREDICT_PATH", "/predict/")
_WAIT_SECONDS: float = float(os.environ.get("PREDICT_WAIT_SECONDS", "0"))
_P95_SLA_MS: float = 50.0  # RNF-70 — cache-hit p95 < 50ms

# Payload fixo — usado em PREDICT_MODE=cache_hit (todo mundo bate na MESMA
# chave de cache) e como valor-base do modo "random" abaixo.
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


def _random_payload() -> dict[str, float]:
    """Payload com sensores analógicos perturbados — nunca repete, força MISS."""
    p = dict(_BASE_PAYLOAD)
    for key in (
        "TP2",
        "TP3",
        "H1",
        "DV_pressure",
        "Reservoirs",
        "Motor_current",
        "Oil_temperature",
    ):
        p[key] = round(p[key] * random.uniform(0.9, 1.1), 4)
    return p


class PredictUser(HttpUser):
    wait_time = between(_WAIT_SECONDS, _WAIT_SECONDS)

    @locust_task
    def predict(self) -> None:
        payload = _BASE_PAYLOAD if _MODE == "cache_hit" else _random_payload()
        name = f"{_PATH} (cache_hit)" if _MODE == "cache_hit" else _PATH
        self.client.post(_PATH, json=payload, name=name)


# ---------------------------------------------------------------------------
# Priming — só em PREDICT_MODE=cache_hit, FORA das estatísticas do Locust.
# ---------------------------------------------------------------------------


@events.test_start.add_listener
def _prime_cache(environment, **kwargs) -> None:  # type: ignore[no-untyped-def]
    if _MODE != "cache_hit":
        return
    host = environment.host or "http://localhost:8000"
    url = f"{host.rstrip('/')}{_PATH}"
    try:
        resp = httpx.post(url, json=_BASE_PAYLOAD, timeout=30.0)
        print(f"\n[priming] MISS esperado — POST {url} -> HTTP {resp.status_code}")
    except (
        Exception
    ) as exc:  # noqa: BLE001 — priming falho deve ser visível, não silencioso
        print(f"\n[priming] FALHOU: {exc}")


# ---------------------------------------------------------------------------
# Veredito RNF-70 — impresso automaticamente ao final de uma run headless
# em PREDICT_MODE=cache_hit
# ---------------------------------------------------------------------------


@events.test_stop.add_listener
def _print_rnf70_verdict(environment, **kwargs) -> None:  # type: ignore[no-untyped-def]
    if _MODE != "cache_hit":
        return
    name = f"{_PATH} (cache_hit)"
    stats = environment.runner.stats.get(name, "POST")
    if stats is None or stats.num_requests == 0:
        print(f"\nRNF-70: sem amostras de '{name}' — verifique o host/endpoint.")
        return

    p50 = stats.get_response_time_percentile(0.50)
    p95 = stats.get_response_time_percentile(0.95)
    p99 = stats.get_response_time_percentile(0.99)
    passed = stats.num_failures == 0 and p95 < _P95_SLA_MS

    print(f"\n{'=' * 60}")
    print("RNF-70 — POST /predict (cache HIT): p95 < 50ms")
    print(f"{'=' * 60}")
    print(f"  Requests               : {stats.num_requests}")
    print(f"  Failures               : {stats.num_failures}")
    print(f"  p50                    : {p50:.1f} ms")
    print(f"  p95                    : {p95:.1f} ms")
    print(f"  p99                    : {p99:.1f} ms")
    print(f"  Resultado              : {'PASS' if passed else 'FAIL'}")
    print(f"{'=' * 60}\n")
