"""
Stage 2 of the RAG benchmark (RF-36/RF-37) — real semantic retrieval.

Indexes the manuals of ``--manuals`` (sidecars honoured) into a TEMPORARY
vector store and, for every query produced by stage 1
(apps/backend/benchmarks/benchmark_rag.py queries), stores the top-20
chunks with their cosine score — unfiltered, so the threshold policies
(mcp cut 0.6, RN-04 τ sweep) can be applied offline.

Usage (inside the mcp-server image):
    python benchmark_rag_retrieval.py --queries /bench/out/rag_queries.json \
        --manuals /src/data/manuals --out /bench/out/rag_retrieval.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import index_manuals as im
from semantic_search import SemanticSearchService

TOP_N: int = 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--manuals", type=Path, default=im.MANUALS_DIR)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    queries: dict[str, dict[str, str]] = json.loads(
        args.queries.read_text(encoding="utf-8")
    )
    model = im.load_embedding_model(im.EMBEDDING_MODEL)

    with tempfile.TemporaryDirectory() as tmp:
        stats = im.run_indexing(
            args.manuals, Path(tmp), im.EMBEDDING_MODEL, im.CHUNK_SIZE, im.CHUNK_OVERLAP
        )
        collection = im.get_collection(Path(tmp))
        service = SemanticSearchService(
            collection,
            model,
            max_results=TOP_N,
            score_threshold=-1.0,
            candidate_pool=max(TOP_N, 50),
        )
        results: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for case_id, variants in queries.items():
            results[case_id] = {}
            for config, query in variants.items():
                results[case_id][config] = [
                    {"text": r.text, "score": round(r.score, 6), "metadata": r.metadata}
                    for r in service.search(query)
                ]
        out = {
            "embedding_model": im.EMBEDDING_MODEL,
            "chunk_size": im.CHUNK_SIZE,
            "chunk_overlap": im.CHUNK_OVERLAP,
            "n_chunks": collection.count(),
            "indexing": stats.summary(),
            "results": results,
        }
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.out} ({len(results)} cases)")


if __name__ == "__main__":
    main()
