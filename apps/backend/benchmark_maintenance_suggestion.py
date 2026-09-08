"""
Benchmark de latência de MaintenanceSuggestionService (RF-22 / RNF-46).

Mede o fluxo real, ponta a ponta, contra os serviços REAIS (não mocks):

    MaintenanceSuggestionService.suggest(request)
        -> MCPSearchClient.search_maintenance_manual  (mcp-server real, Docker)
        -> Ollama /api/chat                            (Llama 3.2 3B real, local)

Separa o tempo de cada etapa (MCP / Ollama / total) — RF-22 pede
explicitamente essa quebra, não só o tempo agregado.

RNF-46 não define um SLA de latência explícito nesta especificação — este
benchmark documenta a latência real medida, sem inventar um limite de
PASS/FAIL (diferente do benchmark de RNF-45 em apps/mcp-server, que tem
uma meta explícita de p95 < 500ms).

Uso
---
    docker compose exec api python benchmark_maintenance_suggestion.py
"""

from __future__ import annotations

import asyncio
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.core.config import settings
from src.schemas.maintenance import MaintenanceSuggestionRequest
from src.services.maintenance_suggestion_service import (
    SYSTEM_PROMPT,
    MaintenanceSuggestionService,
)
from src.services.mcp_client import MCPSearchClient
from src.services.ollama_client import OllamaClient

WARMUP_ITERATIONS = 1
ITERATIONS = 10

# Consultas variadas, relacionadas ao corpus real indexado (RF-20/RF-21) —
# todas acima do threshold de RF-22 (0.7) para garantir que MCP e Ollama
# rodem em toda execução medida.
SCENARIOS: list[MaintenanceSuggestionRequest] = [
    MaintenanceSuggestionRequest(
        failure_probability=0.9,
        equipment_name="Bomba centrifuga BC-200",
        symptom_description="vazamento na bomba centrifuga",
    ),
    MaintenanceSuggestionRequest(
        failure_probability=0.85,
        equipment_name="Motor eletrico ME-750",
        symptom_description="temperatura elevada no motor eletrico",
    ),
    MaintenanceSuggestionRequest(
        failure_probability=0.95,
        equipment_name="Compressor CX-500",
        symptom_description="vazamento de oleo no compressor",
    ),
]


def _percentiles(values_ms: list[float]) -> dict[str, float]:
    arr = np.array(values_ms)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


async def _run_once(
    mcp_client: MCPSearchClient,
    ollama_client: OllamaClient,
    request: MaintenanceSuggestionRequest,
) -> dict[str, float]:
    query = MaintenanceSuggestionService._build_query(request)

    t0 = time.perf_counter()
    raw_results = await mcp_client.search_maintenance_manual(query)
    t_mcp_ms = (time.perf_counter() - t0) * 1000

    contexts = MaintenanceSuggestionService._extract_contexts(raw_results)
    prompt = MaintenanceSuggestionService._build_prompt(request, query, contexts)

    t1 = time.perf_counter()
    await ollama_client.generate(SYSTEM_PROMPT, prompt)
    t_ollama_ms = (time.perf_counter() - t1) * 1000

    return {
        "mcp_ms": t_mcp_ms,
        "ollama_ms": t_ollama_ms,
        "total_ms": t_mcp_ms + t_ollama_ms,
    }


async def run_benchmark() -> dict[str, Any]:
    mcp_client = MCPSearchClient(
        base_url=settings.mcp_server_url, timeout=settings.mcp_client_timeout_seconds
    )
    ollama_client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout=settings.ollama_client_timeout_seconds,
    )

    print(f"Ollama model: {settings.ollama_model} @ {settings.ollama_base_url}")
    print(f"MCP server: {settings.mcp_server_url}")

    print(f"Warm-up ({WARMUP_ITERATIONS} execução, não medida)...")
    for _ in range(WARMUP_ITERATIONS):
        await _run_once(mcp_client, ollama_client, SCENARIOS[0])

    print(f"Medindo {ITERATIONS} execuções...")
    mcp_times: list[float] = []
    ollama_times: list[float] = []
    total_times: list[float] = []
    for i in range(ITERATIONS):
        scenario = SCENARIOS[i % len(SCENARIOS)]
        timings = await _run_once(mcp_client, ollama_client, scenario)
        mcp_times.append(timings["mcp_ms"])
        ollama_times.append(timings["ollama_ms"])
        total_times.append(timings["total_ms"])
        print(
            f"  [{i + 1}/{ITERATIONS}] mcp={timings['mcp_ms']:.1f}ms "
            f"ollama={timings['ollama_ms']:.1f}ms total={timings['total_ms']:.1f}ms"
        )

    return {
        "model": settings.ollama_model,
        "iterations": ITERATIONS,
        "warmup_iterations": WARMUP_ITERATIONS,
        "python_version": platform.python_version(),
        "mcp_ms": {k: round(v, 2) for k, v in _percentiles(mcp_times).items()},
        "ollama_ms": {k: round(v, 2) for k, v in _percentiles(ollama_times).items()},
        "total_ms": {k: round(v, 2) for k, v in _percentiles(total_times).items()},
        "raw_total_ms": [round(v, 2) for v in total_times],
    }


def main() -> None:
    report = asyncio.run(run_benchmark())

    print()
    print("Maintenance Suggestion Benchmark")
    print()
    print(f"Model: {report['model']}")
    print(
        f"Iterations: {report['iterations']} (+ {report['warmup_iterations']} warm-up, não medido)"
    )
    print()
    for stage in ("mcp_ms", "ollama_ms", "total_ms"):
        s = report[stage]
        print(
            f"{stage:10s} p50={s['p50']:.1f}ms p95={s['p95']:.1f}ms p99={s['p99']:.1f}ms "
            f"mean={s['mean']:.1f}ms min={s['min']:.1f}ms max={s['max']:.1f}ms"
        )
    print()
    print(
        "RNF-46 não define SLA de latência explícito nesta especificação — números acima "
        "documentam a latência real medida (sem PASS/FAIL inventado)."
    )

    out_path = Path(__file__).resolve().parent / "maintenance_suggestion_benchmark.json"
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nRelatório salvo em {out_path}")


if __name__ == "__main__":
    main()
