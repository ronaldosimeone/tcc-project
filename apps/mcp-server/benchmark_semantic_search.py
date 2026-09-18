"""
Benchmark de latência da busca semântica (RF-21 / RNF-45).

Mede a operação REAL, ponta a ponta, exatamente como a tool MCP a executa —
não uma função isolada:

    search_maintenance_manual(query)   # server.py, RF-19/RF-21
        -> SemanticSearchService.search(query)   # semantic_search.py
            -> model.encode(query)               # embedding da query
            -> collection.query(...)             # vector store real
            -> cosine_similarity + filtro + top 5

Roda contra o vector store REAL (`CHROMA_DB_PATH`) populado por
`apps/mcp-server/index_manuals.py` (RF-20) — não usa mocks. O carregamento
do modelo de embeddings (rede/disco, ~segundos a dezenas de segundos na
primeira vez) acontece uma única vez, ANTES do cronômetro começar, e não
entra na medição — RNF-45 mede a latência da BUSCA, não o boot do processo.

Uso
---
    docker compose exec mcp-server python benchmark_semantic_search.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from index_manuals import CHROMA_DB_PATH, EMBEDDING_MODEL, get_collection  # noqa: E402
from server import search_maintenance_manual  # noqa: E402

# ---------------------------------------------------------------------------
# Configuração do benchmark
# ---------------------------------------------------------------------------

RNF_45_TARGET_P95_MS = 500.0
WARMUP_ITERATIONS = 3

# >= 20 consultas variadas, relacionadas ao conteúdo REAL indexado nesta
# task (manuais de compressor, bomba centrífuga e motor elétrico — ver
# apps/mcp-server/README/`index_manuals.py`). Não é uma lista de queries
# garantidamente "relevantes o suficiente" para cruzar o threshold de 0.6 —
# isso é uma medição de LATÊNCIA, não de qualidade de recuperação (ver
# `avg_results_per_query` no relatório).
QUERIES: list[str] = [
    "troca de oleo do compressor",
    "vazamento de oleo no compressor industrial",
    "manutencao preventiva do compressor CX-500",
    "revisao tecnica do compressor",
    "filtro de ar do compressor entupido",
    "vazamento na bomba centrifuga",
    "cavitacao na bomba centrifuga BC-200",
    "ruido excessivo na sucao da bomba",
    "troca da vedacao mecanica da bomba",
    "alinhamento do eixo da bomba centrifuga",
    "rolamento da bomba com desgaste",
    "temperatura do motor eletrico trifasico",
    "vibracao anormal no motor eletrico",
    "balanceamento do rotor do motor",
    "limpeza das aletas de ventilacao do motor",
    "superaquecimento do enrolamento do motor ME-750",
    "quando trocar o oleo lubrificante do equipamento",
    "procedimento de parada de emergencia",
    "inspecao periodica de equipamento industrial",
    "manutencao preditiva de maquinas rotativas",
    "checklist de manutencao mensal",
    "equipamento com ruido fora do padrao",
    "substituicao de peca desgastada",
    "manual de manutencao industrial",
]


def _percentiles(latencies_ms: list[float]) -> dict[str, float]:
    arr = np.array(latencies_ms)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def run_benchmark() -> dict[str, Any]:
    collection = get_collection(CHROMA_DB_PATH)
    chunk_count = collection.count()
    if chunk_count == 0:
        raise RuntimeError(
            "Vector store vazio (0 chunks) em "
            f"{CHROMA_DB_PATH} — rode `python index_manuals.py` antes do benchmark "
            "(RF-20). Não faz sentido medir latência de busca sem dados reais."
        )

    print(f"Vector store real: {chunk_count} chunks em {CHROMA_DB_PATH}")
    print(f"Modelo de embeddings: {EMBEDDING_MODEL}")

    # Warm-up — carrega o modelo (lazy singleton em server._get_service) e
    # aquece o índice ANN/torch. NÃO entra na medição.
    print(f"Warm-up ({WARMUP_ITERATIONS} consultas, não medidas)...")
    warmup_start = time.perf_counter()
    for query in QUERIES[:WARMUP_ITERATIONS]:
        search_maintenance_manual(query)
    print(f"Warm-up concluído em {time.perf_counter() - warmup_start:.2f}s")

    print(f"Medindo {len(QUERIES)} consultas...")
    latencies_ms: list[float] = []
    result_counts: list[int] = []
    for query in QUERIES:
        start = time.perf_counter()
        response = search_maintenance_manual(query)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)
        result_counts.append(len(response["results"]))

    stats = _percentiles(latencies_ms)
    report = {
        "model": EMBEDDING_MODEL,
        "documents_chunks": chunk_count,
        "queries": len(QUERIES),
        "warmup_iterations": WARMUP_ITERATIONS,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "latencies_ms": [round(v, 3) for v in latencies_ms],
        "avg_results_per_query": round(sum(result_counts) / len(result_counts), 2),
        "p50_ms": round(stats["p50"], 2),
        "p95_ms": round(stats["p95"], 2),
        "p99_ms": round(stats["p99"], 2),
        "rnf_45_target_p95_ms": RNF_45_TARGET_P95_MS,
        "rnf_45_status": "PASS" if stats["p95"] < RNF_45_TARGET_P95_MS else "FAIL",
    }
    return report


def main() -> None:
    report = run_benchmark()

    print()
    print("Semantic Search Benchmark")
    print()
    print(f"Model: {report['model']}")
    print(f"Documents: {report['documents_chunks']} chunks")
    print(f"Queries: {report['queries']}")
    print(f"Warm-up: {report['warmup_iterations']} iterations (not measured)")
    print(f"Avg results/query: {report['avg_results_per_query']}")
    print()
    print(f"p50: {report['p50_ms']} ms")
    print(f"p95: {report['p95_ms']} ms")
    print(f"p99: {report['p99_ms']} ms")
    print()
    print(
        f"RNF-45 (p95 < {report['rnf_45_target_p95_ms']} ms): {report['rnf_45_status']}"
    )

    # Relativo ao próprio script (apps/mcp-server/), não ao repo root: o
    # container só enxerga apps/mcp-server via bind mount (ver
    # docker-compose.yml::mcp-server) — não há ".." além disso lá dentro.
    out_path = Path(__file__).resolve().parent / "semantic_search_benchmark.json"
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nRelatório salvo em {out_path}")


if __name__ == "__main__":
    main()
