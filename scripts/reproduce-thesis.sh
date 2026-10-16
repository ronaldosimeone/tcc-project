#!/usr/bin/env bash
# Regenerates every experimental result, table and figure of the thesis — RNF-78.
#
# Usage (from the repository root, Git Bash / Linux / macOS):
#   bash scripts/reproduce-thesis.sh            # everything
#   bash scripts/reproduce-thesis.sh ml rag     # only some stages
#
# Stages
#   ml     temporal evaluation, explanations, RUL, drift, robustness, edge   (apps/ml/.venv)
#   rag    RAG retrieval (mcp-server image) + generation (backend venv + Ollama)
#   docs   renders docs/*.md from the JSON results (no training)
#   manifest  writes docs/reproducibility_manifest.json (commit, versions, hashes)
#
# Requirements
#   apps/ml/.venv and apps/backend/.venv installed from their requirements files,
#   Docker with the `projeto-tcc-mcp-server` image built (docker compose build mcp-server),
#   Ollama running locally with llama3.2:3b and qwen2.5:3b-instruct pulled,
#   the WAP-5 manual + sidecar in apps/mcp-server/data/manuals (see CORPUS-CURATION.md).
#
# Every stage is deterministic given the seeds fixed in the scripts; the LLM
# stage uses temperature 0 and seed 42 (Ollama), which is deterministic per
# model build but not across Ollama versions — the manifest records them.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML_PY="$ROOT/apps/ml/.venv/Scripts/python.exe"
BE_PY="$ROOT/apps/backend/.venv/Scripts/python.exe"
[ -x "$ML_PY" ] || ML_PY="$ROOT/apps/ml/.venv/bin/python"
[ -x "$BE_PY" ] || BE_PY="$ROOT/apps/backend/.venv/bin/python"

STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(ml rag docs manifest)

run_ml() {
  cd "$ROOT/apps/ml"
  "$ML_PY" -m pytest -q tests/test_temporal_validation.py tests/test_uncertainty.py \
    tests/test_explainability.py tests/test_rul.py tests/test_drift_detectors.py tests/test_promotion_gate.py
  "$ML_PY" -m src.evaluate_temporal --seeds 5 --neg-stride 5   # RF-28/29/31, RNF-79
  "$ML_PY" -m src.explain_events --max-rows 600                 # RF-32
  "$ML_PY" -m src.evaluate_rul                                  # RF-30
  "$ML_PY" -m src.evaluate_drift --trials 30 --calib-trials 30  # RF-34
  "$ML_PY" -m src.evaluate_robustness --search 40               # RNF-81
  "$ML_PY" -m src.edge_quantization                             # RNF-80
  "$ML_PY" -m pytest -q tests/test_digital_twin.py
  "$ML_PY" -m src.evaluate_twin --seeds 3                       # digital twin + fusão
}

mcp_docker() {
  # $1 = embedding model, rest = command run inside the mcp-server image
  local model="$1"; shift
  MSYS_NO_PATHCONV=1 docker run --rm -e EMBEDDING_MODEL="$model" \
    -v "$ROOT/apps/mcp-server:/src" -v "$ROOT/apps/backend/benchmarks:/bench" -w /src \
    projeto-tcc-mcp-server:latest sh -c "$*"
}

run_rag() {
  cd "$ROOT/apps/backend"
  "$BE_PY" benchmarks/build_rag_cases.py
  "$BE_PY" benchmarks/benchmark_rag.py queries
  mcp_docker "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" \
    python benchmark_rag_retrieval.py --queries /bench/out/rag_queries.json --manuals /src/data/manuals --out /bench/out/rag_retrieval.json
  mcp_docker "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" \
    python benchmark_rag_retrieval.py --queries /bench/out/rag_queries.json --manuals /src/data/manuals --out /bench/out/rag_retrieval_mpnet.json
  mcp_docker "BAAI/bge-m3" \
    python benchmark_rag_retrieval.py --queries /bench/out/rag_queries.json --manuals /src/data/manuals --out /bench/out/rag_retrieval_bge-m3.json
  rm -f benchmarks/out/rag_generation.json
  "$BE_PY" benchmarks/benchmark_rag.py generate --models llama3.2:3b qwen2.5:3b-instruct
  "$BE_PY" benchmarks/benchmark_rag.py generate --models llama3.2:3b qwen2.5:3b-instruct \
    --configs rag rag_xai --retrieval-file rag_retrieval_bge-m3.json --mcp-cut -1 --tau 0.6 --tag bge-m3_tau0.60
  # RNF-84 red-team (the "no guard" baseline is kept from the original run).
  "$BE_PY" benchmarks/benchmark_redteam.py queries
  mcp_docker "BAAI/bge-m3" \
    "python redteam_build_corpus.py --base /src/data/manuals --out /bench/out/redteam_manuals && \
     python benchmark_rag_retrieval.py --queries /bench/out/redteam_queries.json --manuals /bench/out/redteam_manuals --out /bench/out/redteam_retrieval.json"
  "$BE_PY" benchmarks/benchmark_redteam.py run --models llama3.2:3b qwen2.5:3b-instruct --tau 0.6
}

run_docs() {
  cd "$ROOT/apps/ml"
  "$ML_PY" -m src.render_temporal_report
  cd "$ROOT/apps/backend"
  "$BE_PY" benchmarks/benchmark_rag.py report
  "$BE_PY" benchmarks/benchmark_redteam.py report
}

run_manifest() {
  cd "$ROOT"
  "$ML_PY" scripts/write_manifest.py
}

for stage in "${STAGES[@]}"; do
  echo "==> stage: $stage"
  "run_$stage"
done
echo "Done. Results: apps/ml/models/*.json, apps/backend/benchmarks/out/, docs/*.md, docs/figures-ml/"
