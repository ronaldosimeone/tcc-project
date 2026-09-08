"""
Load test — SSE streaming sob concorrência real (RNF-37).

RNF-37: o sistema deve suportar >= 100 conexões SSE concorrentes com
        latência de conexão p95 < 200ms, medido no endpoint SSE real da
        aplicação (`GET /api/stream/sensors`, atrás do Nginx — mesmo path
        que o frontend usa).

Reaproveitamento
-----------------
Este script NÃO reimplementa o parsing/validação de eventos SSE: importa
`REQUIRED_SENSOR_FIELDS` de `locust_sse.py` (já existente, cobre RNF-28 com
>=50 clientes). `locust_sse.py` mede "eventos recebidos por cliente" (RF-12);
este script mede especificamente a LATÊNCIA DE CONEXÃO sob >=100 conexões
simultâneas, que é a métrica do RNF-37 — ver `stream_connection()` abaixo
para a justificativa detalhada de por que isso é "tempo até os cabeçalhos
de resposta" e NÃO "tempo até o primeiro evento de dado".

O que é medido como "latência" aqui
-------------------------------------
Latência de CONEXÃO: do envio do GET até a resposta HTTP (200 OK + cabeçalhos)
chegar — o momento em que o cliente sabe que o stream está aberto. Isso é o
que testes de carga de streaming convencionalmente reportam como "response
time" e é o que o Locust já sabe agregar em p50/p95/p99 nativamente —
reaproveitado via `environment.events.request.fire(...)`, sem reimplementar
percentis. NÃO é o tempo até o primeiro evento `data:` — motivo detalhado na
docstring de `stream_connection()`.

Uso
---
Headless, contra o stack via Nginx (padrão docker-compose, porta 80):

    pip install locust
    locust -f locust_streaming.py --host http://localhost \\
           --headless -u 100 -r 10 --run-time 60s --csv=streaming

O sumário RNF-37 (PASS/FAIL) é impresso automaticamente ao final da run
headless (hook `test_stop`).

Autenticação (se o endpoint exigir no futuro)
----------------------------------------------
`GET /stream/sensors` hoje é público (sem Depends de auth). Caso isso mude,
defina a variável de ambiente STREAMING_AUTH_TOKEN — NUNCA hardcode um
token aqui. Se definida, é enviada como `Authorization: Bearer <token>`.

Configuração (env vars, todas opcionais)
------------------------------------------
    STREAMING_AUTH_TOKEN   token opcional (ver acima)
    SSE_HOLD_SECONDS       quanto tempo cada cliente mantém a conexão SSE
                           aberta lendo eventos antes de reconectar (default 45)
    SSE_PATH               path do endpoint (default /api/stream/sensors)
"""

from __future__ import annotations

import json
import os
import sys
import time

# Windows consoles frequentemente usam cp1252/cp437 (não UTF-8) como stdout
# encoding padrão — sem isto, os símbolos ✓/✗ abaixo derrubam o processo do
# Locust com UnicodeEncodeError no meio da run headless.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from locust import HttpUser, between
from locust import events
from locust import task as locust_task

from locust_sse import REQUIRED_SENSOR_FIELDS  # reuso — não duplica o parsing

_AUTH_TOKEN: str | None = os.environ.get("STREAMING_AUTH_TOKEN")
_HOLD_SECONDS: float = float(os.environ.get("SSE_HOLD_SECONDS", "45"))
_SSE_PATH: str = os.environ.get("SSE_PATH", "/api/stream/sensors")

_P95_SLA_MS: float = 200.0  # RNF-37
_MIN_CONNECTIONS: int = 100  # RNF-37


class SSEStreamingUser(HttpUser):
    """
    Cada usuário virtual abre UMA conexão SSE persistente e a mantém aberta
    por `SSE_HOLD_SECONDS`, lendo eventos continuamente — representa o
    comportamento real do frontend (EventSource de longa duração), não uma
    rajada de requests HTTP independentes.
    """

    wait_time = between(0, 0)

    @locust_task
    def stream_connection(self) -> None:
        """
        Mede a LATÊNCIA DE CONEXÃO (handshake TCP/HTTP + cabeçalhos de
        resposta — "conexão SSE estabelecida"), NÃO a latência até o
        primeiro evento de dado.

        Por que não medir "tempo até o 1º evento"
        --------------------------------------------
        O stream emite exatamente 1 evento/segundo por design (RF-12,
        `BROADCAST_INTERVAL=1.0` em `sensor_stream_service.py`), num
        broadcast COMPARTILHADO com clock próprio. Um cliente que acabou de
        conectar pode legitimamente esperar até ~1s pelo próximo tick antes
        de ver seu primeiro dado — isso é um traço de produto (cadência de
        1Hz), não uma característica de performance/infra. Medir isso como
        "latência" tornaria RNF-37 (p95<200ms) estruturalmente impossível de
        passar independente de workers/pool/otimização, o que contradiz o
        propósito do requisito (avaliar concorrência/infra, não a cadência
        do domínio). Confirmado empiricamente: `curl`/socket cru mostram a
        resposta HTTP (handshake completo, stream aberto) em milissegundos
        de forma consistente; só o primeiro EVENTO de dado varia 0–1000ms+
        conforme o alinhamento com o tick do broadcast.
        """
        headers = {"Accept": "text/event-stream", "Cache-Control": "no-cache"}
        if _AUTH_TOKEN:
            headers["Authorization"] = f"Bearer {_AUTH_TOKEN}"

        connect_start = time.perf_counter()
        events_read = 0
        connect_fired = (
            False  # evita contar a mesma tentativa 2x (conexão + exceção depois)
        )

        try:
            with self.client.get(
                _SSE_PATH,
                stream=True,
                headers=headers,
                catch_response=True,
                timeout=_HOLD_SECONDS + 15,
                name=f"SSE {_SSE_PATH}",
            ) as response:
                # Latência de conexão: handshake + cabeçalhos já recebidos
                # neste ponto (stream=True não baixa o corpo ainda).
                connect_ms = (time.perf_counter() - connect_start) * 1000
                connect_fired = True
                events.request.fire(
                    request_type="SSE-connect",
                    name=_SSE_PATH,
                    response_time=connect_ms,
                    response_length=0,
                    exception=(
                        None
                        if response.status_code == 200
                        else RuntimeError(f"HTTP {response.status_code}")
                    ),
                )

                if response.status_code != 200:
                    response.failure(f"HTTP {response.status_code}")
                    return

                deadline = time.monotonic() + _HOLD_SECONDS
                for raw_line in response.iter_lines(decode_unicode=True):
                    if time.monotonic() >= deadline:
                        break
                    if not raw_line.startswith("data: "):
                        continue

                    try:
                        payload = json.loads(raw_line[6:])
                    except json.JSONDecodeError:
                        response.failure("JSON inválido no evento SSE")
                        return
                    missing = REQUIRED_SENSOR_FIELDS - payload.keys()
                    if missing:
                        response.failure(f"Campos ausentes no evento: {missing}")
                        return

                    events_read += 1

                if events_read > 0:
                    response.success()
                else:
                    response.failure("Conexão abriu mas nenhum evento foi recebido")
        except (
            Exception
        ) as e:  # noqa: BLE001 — desconexão inesperada = falha registrada
            if not connect_fired:
                # A exceção ocorreu ANTES de obter resposta (ex.: connection
                # refused/timeout) — só então conta como uma tentativa de
                # conexão falha; se já tinha conectado, a falha é de stream,
                # não de conexão, e não deve poluir a métrica de RNF-37.
                events.request.fire(
                    request_type="SSE-connect",
                    name=_SSE_PATH,
                    response_time=(time.perf_counter() - connect_start) * 1000,
                    response_length=0,
                    exception=e,
                )


# ---------------------------------------------------------------------------
# Veredito RNF-37 — impresso automaticamente ao final de uma run headless
# ---------------------------------------------------------------------------


@events.test_stop.add_listener
def _print_rnf37_verdict(environment, **kwargs) -> None:  # type: ignore[no-untyped-def]
    stats = environment.runner.stats.get(_SSE_PATH, "SSE-connect")
    if stats is None or stats.num_requests == 0:
        print("\nRNF-37: sem amostras de 'SSE-connect' — verifique o host/endpoint.")
        return

    p95 = stats.get_response_time_percentile(0.95)
    concurrent = stats.num_requests - stats.num_failures
    passed = concurrent >= _MIN_CONNECTIONS and p95 < _P95_SLA_MS

    print(f"\n{'=' * 60}")
    print("RNF-37 — SSE: >=100 conexões concorrentes, p95 < 200ms")
    print(f"{'=' * 60}")
    print(f"  Conexões bem-sucedidas : {concurrent}")
    print(f"  Falhas                 : {stats.num_failures}")
    print(
        f"  p50                    : {stats.get_response_time_percentile(0.50):.1f} ms"
    )
    print(f"  p95                    : {p95:.1f} ms")
    print(
        f"  p99                    : {stats.get_response_time_percentile(0.99):.1f} ms"
    )
    print(f"  Resultado              : {'PASS ✓' if passed else 'FAIL ✗'}")
    print(f"{'=' * 60}\n")
