"""
RNF-45 re-measured per embedder — decision input for swapping MiniLM → bge-m3.

Same operation as ``benchmark_semantic_search.py`` (query embedding + vector
store + cosine + top 5, warm-up excluded), but for several embedders over the
SAME corpus (the real manuals, sidecars honoured), each indexed into a
temporary store. Also records what the swap costs outside latency: model load
time, resident memory and embedding dimension.

Usage (inside the mcp-server image, CPU only — the production setting):
    docker run --rm -v <manuals>:/app/data/manuals:ro -v <hf-cache>:/hf \\
        -e HF_HOME=/hf -v <out>:/out projeto-tcc-mcp-server:latest \\
        python benchmark_embedder_latency.py --out /out/embedder_latency.json
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

import index_manuals as im
from benchmark_semantic_search import QUERIES, RNF_45_TARGET_P95_MS, WARMUP_ITERATIONS
from semantic_search import SemanticSearchService

MODELS: tuple[str, ...] = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "BAAI/bge-m3",
)
REPEATS: int = 5  # every query is timed REPEATS times


def _rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def measure(model_name: str, manuals: Path) -> dict[str, Any]:
    rss0 = _rss_mb()
    t0 = time.perf_counter()
    model = im.load_embedding_model(model_name)
    load_s = time.perf_counter() - t0
    rss_model = _rss_mb() - rss0  # before indexing (which loads its own copy)
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        stats = im.run_indexing(
            manuals, Path(tmp), model_name, im.CHUNK_SIZE, im.CHUNK_OVERLAP
        )
        index_s = time.perf_counter() - t0
        service = SemanticSearchService(im.get_collection(Path(tmp)), model)
        for q in QUERIES[:WARMUP_ITERATIONS]:
            service.search(q)
        times: list[float] = []
        for _ in range(REPEATS):
            for q in QUERIES:
                t = time.perf_counter()
                service.search(q)
                times.append((time.perf_counter() - t) * 1000)
    arr = np.array(times)
    p95 = float(np.percentile(arr, 95))
    out = {
        "model": model_name,
        "embedding_dim": int(model.get_sentence_embedding_dimension() or 0),
        "load_s": round(load_s, 2),
        "index_s": round(index_s, 2),
        "rss_model_mb": round(rss_model, 1),
        "index_stats": {k: v for k, v in vars(stats).items() if isinstance(v, int)},
        "n_timed_queries": int(arr.size),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": p95,
        "p99_ms": float(np.percentile(arr, 99)),
        "rnf45_pass": p95 < RNF_45_TARGET_P95_MS,
    }
    del model, service
    gc.collect()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--manuals", type=Path, default=im.MANUALS_DIR)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    args = parser.parse_args()
    import torch

    result: dict[str, Any] = {
        "requirement": "RNF-45 (p95 < 500 ms)",
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "queries": len(QUERIES),
        "repeats": REPEATS,
        "models": [],
    }
    for name in args.models:
        row = measure(name, args.manuals)
        print(json.dumps(row), flush=True)
        result["models"].append(row)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
