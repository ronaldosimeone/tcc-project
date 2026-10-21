#!/usr/bin/env bash
# Regenerates every experimental result, table and figure of the thesis — RNF-78.
#
# Usage (from the repository root, Git Bash / Linux / macOS):
#   bash scripts/reproduce-thesis.sh            # everything
#   bash scripts/reproduce-thesis.sh ml rag     # only some stages
#
# Stages
#   ml     temporal evaluation, explanations, RUL, drift, robustness, edge,  (apps/ml/.venv)
#          digital twin, cross-dataset (RF-33), energy (RNF-83)
#   infra  TimescaleDB vs Postgres benchmark (Docker) + RNF-45 per embedder
#   rag    RAG retrieval (mcp-server image) + generation (backend venv + Ollama)
#   docs   renders docs/*.md from the JSON results (no training)
#   manifest  writes docs/reproducibility_manifest.json (commit, versions, hashes)
#
# Requirements
#   apps/ml/.venv and apps/backend/.venv installed from their requirements files,
#   Docker with the `projeto-tcc-mcp-server` image built (docker compose build mcp-server),
#   Ollama running locally with llama3.2:3b and qwen2.5:3b-instruct pulled,
#   the WAP-5 manual + sidecar in apps/mcp-server/data/manuals (see CORPUS-CURATION.md),
#   MetroPT 2022 dataset_train.csv in apps/ml/data/raw/metropt2022/ (Zenodo
#   10.5281/zenodo.6854240, md5 checked by the ingestion).
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
GPU_PY="$ROOT/apps/ml/.venv-gpu/Scripts/python.exe"
[ -x "$GPU_PY" ] || GPU_PY="$ML_PY"

STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(ml rag infra docs manifest)

run_ml() {
  cd "$ROOT/apps/ml"
  "$ML_PY" -m pytest -q tests/test_temporal_validation.py tests/test_uncertainty.py \
    tests/test_explainability.py tests/test_rul.py tests/test_drift_detectors.py tests/test_promotion_gate.py
  "$ML_PY" -m src.evaluate_temporal --seeds 5 --neg-stride 5   # RF-28/29/31, RNF-79
  "$ML_PY" -m src.explain_events --max-rows 600                 # RF-32
  "$ML_PY" -m src.evaluate_rul                                  # RF-30
  "$ML_PY" -m src.evaluate_rul --pooled                         # RF-30 com 7 falhas
  "$ML_PY" -m src.evaluate_drift --trials 30 --calib-trials 30  # RF-34
  "$ML_PY" -m src.evaluate_robustness --search 40               # RNF-81
  "$ML_PY" -m src.edge_quantization                             # RNF-80
  "$ML_PY" -m pytest -q tests/test_digital_twin.py
  "$ML_PY" -m src.evaluate_twin --seeds 3                       # digital twin + fusão
  "$ML_PY" -m pytest -q tests/test_ingest_metropt2022.py
  "$ML_PY" -m src.ingest_metropt2022                            # RF-33 — 2º dataset
  "$ML_PY" -m src.evaluate_cross_dataset --seeds 3              # RF-33, RN-05 com 6 folds
  "$ML_PY" -m src.evaluate_cross_dataset --seeds 3 --drop-sensors Reservoirs Oil_level \
    --tag harmonized --only shift cross pooled                  # harmonização semântica
  "$ML_PY" -m src.evaluate_twin_cross --seeds 3                 # twin + fusão entre períodos
  "$ML_PY" -m src.evaluate_production_on_2022                   # modelo implantado no 2022
  # Retreino sem vazamento de TODOS os modelos servíveis + matrizes de confusão
  # Protocolo aninhado (tuning + calibração + limiar sem vazamento), 3 seeds.
  # Árvores na CPU; redes no venv de GPU (apps/ml/.venv-gpu, torch CUDA) se existir.
  "$ML_PY" -m src.evaluate_nested --models random_forest logistic_regression
  "$GPU_PY" -m src.evaluate_nested --models xgboost autoencoder patchtst bilstm tcn mlp
  "$GPU_PY" -m src.train_production                             # os 7 artefatos *_v3 servidos
  "$ML_PY" -m src.evaluate_sensitivity                          # k × horizonte × razão de custo
  "$ML_PY" -m src.evaluate_conformal_aci                        # conformal estático × ACI
  # Auditoria da validação e das features (revisão técnica, rodada 6)
  "$ML_PY" -m src.fold_chronology                               # datas de cada fold externo/interno
  "$ML_PY" -m src.event_correlation                             # associação sensor × falha por evento
  for v in "--drop-sensors DV_pressure --tag no_dv_pressure" "--only-sensors DV_pressure --tag dv_pressure_only"; do
    "$ML_PY" -m src.evaluate_nested --models random_forest logistic_regression --seeds 1 $v
    "$GPU_PY" -m src.evaluate_nested --models xgboost --seeds 1 $v
  done
  "$ML_PY" -m src.render_ablation_report                        # ablação DV_pressure
  "$GPU_PY" -m src.evaluate_anticipation                        # alvo "falha nas próximas H"
  # MetroPT2 (Zenodo 7766691) em data/raw/metropt2/MetroPT2.csv — falhas inéditas.
  "$ML_PY" -m src.ingest_metropt2
  "$GPU_PY" -m src.evaluate_metropt2 --no-oil-models            # servidos + twin + sem óleo
  "$ML_PY" -m src.build_diagnosis_cases                         # evidência XAI das 9 falhas
  # RNF-83 — rode com a máquina ociosa (mede energia da máquina inteira).
  "$ML_PY" -m src.evaluate_energy
}

run_infra() {
  cd "$ROOT/apps/backend"
  docker rm -f bench-pg bench-ts >/dev/null 2>&1 || true
  docker run -d --name bench-pg -e POSTGRES_PASSWORD=bench -p 55432:5432 postgres:15 >/dev/null
  docker run -d --name bench-ts -e POSTGRES_PASSWORD=bench -p 55433:5432     timescale/timescaledb:2.17.2-pg15 postgres -c shared_preload_libraries=timescaledb >/dev/null
  until docker exec bench-pg pg_isready -U postgres >/dev/null 2>&1 &&         docker exec bench-ts pg_isready -U postgres >/dev/null 2>&1; do sleep 1; done
  sleep 3
  "$BE_PY" benchmarks/benchmark_timescale.py     --plain postgresql://postgres:bench@localhost:55432/postgres     --timescale postgresql://postgres:bench@localhost:55433/postgres
  docker rm -f bench-pg bench-ts >/dev/null
  mcp_docker "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"     python benchmark_embedder_latency.py --manuals /src/data/manuals --out /src/embedder_latency_benchmark.json
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
  # Diagnóstico RAG nas falhas reais (casos de apps/ml/src/build_diagnosis_cases.py).
  "$BE_PY" benchmarks/benchmark_rag_diagnosis.py queries
  mcp_docker "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"     python benchmark_rag_retrieval.py --queries /bench/out/rag_diag_queries.json --manuals /src/data/manuals --out /bench/out/rag_diag_retrieval.json
  mcp_docker "BAAI/bge-m3"     python benchmark_rag_retrieval.py --queries /bench/out/rag_diag_queries.json --manuals /src/data/manuals --out /bench/out/rag_diag_retrieval_bge-m3.json
  rm -f benchmarks/out/rag_diag_generation.json
  "$BE_PY" benchmarks/benchmark_rag_diagnosis.py generate --model llama3.2:3b
  # Paridade offline × ao vivo: cenários reais pelo caminho de serving exato.
  MSYS_NO_PATHCONV=1 docker exec -w /app api python benchmarks/benchmark_live_replay.py
}

run_docs() {
  cd "$ROOT/apps/ml"
  "$ML_PY" -m src.render_temporal_report
  "$ML_PY" -m src.render_cross_dataset_report
  "$ML_PY" -m src.render_nested_report
  cd "$ROOT/apps/backend"
  "$BE_PY" benchmarks/benchmark_rag.py report
  "$BE_PY" benchmarks/benchmark_redteam.py report
  "$BE_PY" benchmarks/benchmark_rag_diagnosis.py report
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
