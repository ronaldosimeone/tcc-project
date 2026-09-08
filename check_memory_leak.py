"""
Verificação de memory leak sob carga (RNF-38).

RNF-38: crescimento de memória > 50MB durante o teste definido = FAIL.

Métrica usada e por quê
-------------------------
RSS (Resident Set Size) do PROCESSO REAL do serviço `api`, não
`tracemalloc`. `tracemalloc` só enxerga alocações Python; o container roda
sob asyncpg (driver C), ONNX Runtime (C++) e SQLAlchemy — boa parte da
memória real do serviço nunca passaria pelo `tracemalloc`, então ele
sub-mediria um leak real. RSS é o que efetivamente aparece em `docker stats`
/ monitoramento de produção, então é o que RNF-38 pretende proteger.

Duas fontes de RSS, nesta ordem de preferência
-------------------------------------------------
1. `docker stats <container>` — mede o container `api` real do
   docker-compose de fora (cgroup), sem precisar instalar nada dentro da
   imagem. É o caminho padrão (RNF-38 se refere ao processo real de
   produção, que roda em container).
2. `psutil` (fallback) — mede um PID local, para quando a API roda fora do
   Docker (`uvicorn src.main:app` direto). Documentado explicitamente no
   relatório de saída porque RSS local pode não representar o ambiente de
   produção (cgroup limits, overhead do container).

Geração de carga
------------------
Reaproveita `_run_sse_client` de `locust_sse.py` (já valida os campos do
evento e mede erros) — não reimplementa um cliente SSE paralelo.

Ciclos
------
Mede RSS antes de QUALQUER carga (baseline frio), roda um ciclo de warmup
(descartado — aquece pools, caches, JIT de import etc.) e então N ciclos de
carga+assentamento, registrando RSS após cada um. Crescimento sustentado
ciclo-a-ciclo (não só entre baseline e ciclo 1) é o sinal de leak real;
crescimento que estabiliza após o warmup é alocação normal.

Uso
---
    # Via Docker (padrão — mede o container real do docker-compose)
    python check_memory_leak.py --container api

    # Via PID local (uvicorn rodando fora do Docker)
    python check_memory_leak.py --pid 12345

    # Parâmetros de carga (mesmos do locust_sse standalone)
    python check_memory_leak.py --container api --clients 100 \\
        --cycles 3 --load-duration 30 --settle-seconds 10 \\
        --url http://localhost/api
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from dataclasses import dataclass, field

# Windows consoles frequentemente usam cp1252/cp437 (não UTF-8) — sem isto,
# o print do PASS/FAIL abaixo pode derrubar o processo com UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from locust_sse import (
    ClientResult,
    _run_sse_client,
)  # reuso — não duplica o cliente SSE

import httpx

_GROWTH_LIMIT_MB: float = 50.0  # RNF-38


# ---------------------------------------------------------------------------
# Amostragem de RSS
# ---------------------------------------------------------------------------


def _rss_via_docker_stats(container: str) -> float:
    """
    Retorna o RSS (MB) do container via `docker stats --no-stream`.

    Usa `MemUsage` (formato "123.4MiB / 3GiB") em vez de `MemPerc` — dá o
    valor absoluto, que é o que RNF-38 compara (delta em MB).
    """
    out = subprocess.run(
        [
            "docker",
            "stats",
            container,
            "--no-stream",
            "--format",
            "{{.MemUsage}}",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    raw = out.stdout.strip().split("/")[0].strip()  # ex.: "123.4MiB"
    return _parse_mem_to_mb(raw)


def _parse_mem_to_mb(raw: str) -> float:
    raw = raw.strip()
    for suffix, factor in (
        ("GiB", 1024.0),
        ("MiB", 1.0),
        ("KiB", 1.0 / 1024.0),
        ("GB", 1000.0),
        ("MB", 1.0),
        ("KB", 1.0 / 1000.0),
        ("B", 1.0 / (1024.0 * 1024.0)),
    ):
        if raw.endswith(suffix):
            return float(raw[: -len(suffix)]) * factor
    raise ValueError(f"Formato de memória não reconhecido: {raw!r}")


def _rss_via_psutil(pid: int) -> float:
    import psutil  # import local — só é necessário neste caminho

    return psutil.Process(pid).memory_info().rss / (1024.0 * 1024.0)


def sample_rss_mb(container: str | None, pid: int | None) -> float:
    if container is not None:
        return _rss_via_docker_stats(container)
    if pid is not None:
        return _rss_via_psutil(pid)
    raise ValueError("Informe --container ou --pid para medir RSS.")


# ---------------------------------------------------------------------------
# Geração de carga (reaproveita locust_sse._run_sse_client)
# ---------------------------------------------------------------------------


async def _generate_load(
    url: str, n_clients: int, duration: float
) -> list[ClientResult]:
    limits = httpx.Limits(
        max_connections=n_clients + 10, max_keepalive_connections=n_clients + 10
    )
    async with httpx.AsyncClient(timeout=duration + 15, limits=limits) as client:
        tasks = [_run_sse_client(client, url, duration, i) for i in range(n_clients)]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Ciclos de medição
# ---------------------------------------------------------------------------


@dataclass
class CycleSample:
    label: str
    rss_mb: float
    load_errors: int = 0


@dataclass
class LeakCheckResult:
    samples: list[CycleSample] = field(default_factory=list)

    @property
    def baseline_mb(self) -> float:
        return self.samples[0].rss_mb

    @property
    def final_mb(self) -> float:
        return self.samples[-1].rss_mb

    @property
    def growth_mb(self) -> float:
        return self.final_mb - self.baseline_mb

    @property
    def passed(self) -> bool:
        return self.growth_mb <= _GROWTH_LIMIT_MB


async def run_leak_check(
    url: str,
    container: str | None,
    pid: int | None,
    n_clients: int,
    cycles: int,
    load_duration: float,
    settle_seconds: float,
) -> LeakCheckResult:
    result = LeakCheckResult()

    baseline = sample_rss_mb(container, pid)
    result.samples.append(CycleSample(label="baseline (frio)", rss_mb=baseline))
    print(f"[baseline]  RSS = {baseline:.1f} MB")

    # Warmup — descartado do cálculo de crescimento. Aquece pools de conexão,
    # caches de feature engineering, buffers do event loop etc. Sem isso, o
    # "ciclo 1" sempre pareceria a maior alocação e mascararia se há
    # crescimento *sustentado* nos ciclos seguintes.
    print("[warmup] rodando ciclo de aquecimento (descartado)…")
    await _generate_load(url, n_clients, load_duration)
    await asyncio.sleep(settle_seconds)
    warm = sample_rss_mb(container, pid)
    result.samples.append(CycleSample(label="pós-warmup", rss_mb=warm))
    print(f"[warmup]    RSS = {warm:.1f} MB (descartado do cálculo de growth)")

    for i in range(1, cycles + 1):
        print(
            f"[cycle {i}] gerando carga: {n_clients} clientes SSE por {load_duration}s…"
        )
        results = await _generate_load(url, n_clients, load_duration)
        errors = sum(1 for r in results if r.error)

        await asyncio.sleep(
            settle_seconds
        )  # assentamento — GC natural do processo real
        rss = sample_rss_mb(container, pid)
        result.samples.append(
            CycleSample(label=f"ciclo {i}", rss_mb=rss, load_errors=errors)
        )
        print(f"[cycle {i}] RSS = {rss:.1f} MB (erros de carga: {errors})")

    return result


def print_report(result: LeakCheckResult, metric_source: str) -> None:
    print(f"\n{'=' * 60}")
    print("RNF-38 — Memory leak check")
    print(f"{'=' * 60}")
    print(f"  Fonte da métrica     : {metric_source}")
    for s in result.samples:
        print(
            f"  {s.label:<14}: {s.rss_mb:8.1f} MB"
            + (f"  (erros de carga: {s.load_errors})" if s.load_errors else "")
        )
    print(f"  Baseline (pós-warmup): {result.samples[1].rss_mb:.1f} MB")
    print(f"  Final                : {result.final_mb:.1f} MB")
    growth_from_warm = result.final_mb - result.samples[1].rss_mb
    print(f"  Crescimento (warmup->final): {growth_from_warm:+.1f} MB")
    print(f"  Limite RNF-38        : {_GROWTH_LIMIT_MB:.0f} MB")
    passed = growth_from_warm <= _GROWTH_LIMIT_MB
    print(f"  Resultado            : {'PASS ✓' if passed else 'FAIL ✗'}")
    print(f"{'=' * 60}\n")

    if not passed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RNF-38 memory leak check (RSS real)")
    parser.add_argument(
        "--container",
        default="api",
        help="Nome do container docker-compose a medir via `docker stats` (default: api). "
        "Use --pid para medir um processo local em vez de um container.",
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="PID local a medir via psutil, em vez de --container.",
    )
    parser.add_argument(
        "--url", default="http://localhost/api", help="Base URL da API."
    )
    parser.add_argument(
        "--clients", type=int, default=100, help="Clientes SSE por ciclo."
    )
    parser.add_argument(
        "--cycles", type=int, default=3, help="Ciclos de carga medidos."
    )
    parser.add_argument(
        "--load-duration", type=float, default=30.0, help="Duração de cada ciclo (s)."
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=10.0,
        help="Espera após cada ciclo antes de medir RSS (assentamento/GC).",
    )
    args = parser.parse_args()

    container = None if args.pid is not None else args.container
    metric_source = (
        f"psutil (PID {args.pid}, processo local — pode não representar produção)"
        if args.pid is not None
        else f"docker stats (container '{container}')"
    )

    outcome = asyncio.run(
        run_leak_check(
            url=args.url,
            container=container,
            pid=args.pid,
            n_clients=args.clients,
            cycles=args.cycles,
            load_duration=args.load_duration,
            settle_seconds=args.settle_seconds,
        )
    )
    print_report(outcome, metric_source)
