"""
Reproducibility manifest — RNF-78.

Writes docs/reproducibility_manifest.json with everything needed to tell
whether two result sets are comparable: git commit + dirty flag, Python and
key library versions, SHA-256 of every result JSON and of the data/model
artefacts they were computed from, the Docker image digest of the
mcp-server, and the Ollama model digests.

Usage:  apps/ml/.venv/Scripts/python.exe scripts/write_manifest.py
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen

ROOT: Path = Path(__file__).resolve().parents[1]
OUT: Path = ROOT / "docs" / "reproducibility_manifest.json"

RESULT_FILES: list[str] = [
    "apps/ml/models/eval_temporal.json",
    "apps/ml/models/explain_events.json",
    "apps/ml/models/eval_rul.json",
    "apps/ml/models/eval_drift.json",
    "apps/ml/models/eval_robustness.json",
    "apps/ml/models/edge_quantization.json",
    "apps/ml/models/eval_twin.json",
    "apps/ml/models/metropt2022_ingest.json",
    "apps/ml/models/eval_cross_dataset.json",
    "apps/ml/models/eval_cross_dataset_harmonized.json",
    "apps/ml/models/eval_energy.json",
    "apps/backend/benchmarks/timescale_benchmark.json",
    "apps/mcp-server/embedder_latency_benchmark.json",
    "apps/backend/benchmarks/rag_cases_v1.json",
    "apps/backend/benchmarks/out/rag_retrieval.json",
    "apps/backend/benchmarks/out/rag_retrieval_mpnet.json",
    "apps/backend/benchmarks/out/rag_retrieval_bge-m3.json",
    "apps/backend/benchmarks/out/rag_generation.json",
    "apps/backend/benchmarks/out/redteam_results_no_guard.json",
    "apps/backend/benchmarks/out/redteam_results.json",
]
INPUT_FILES: list[str] = [
    "apps/ml/data/processed/metropt3.parquet",
    "apps/ml/data/processed/metropt2022.parquet",
    "apps/ml/models/random_forest_v2.onnx",
    "apps/ml/models/mlp_v1.onnx",
    "apps/ml/models/tcn_v1.onnx",
    "apps/mcp-server/data/manuals/wap5-scheduled-maintenance-vol-b.pdf",
    "apps/mcp-server/data/manuals/wap5-scheduled-maintenance-vol-b.meta.json",
]
PACKAGES: list[str] = [
    "numpy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "scipy",
    "shap",
    "onnxruntime",
    "torch",
    "lightning",
]


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _cmd(*args: str) -> str | None:
    try:
        return subprocess.run(
            args, capture_output=True, text=True, check=True, cwd=ROOT
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def _ollama_models() -> list[dict[str, Any]] | None:
    try:
        with urlopen(
            "http://localhost:11434/api/tags", timeout=3
        ) as resp:  # noqa: S310 — local only
            data = json.loads(resp.read())
        return [
            {"name": m["name"], "digest": m["digest"]} for m in data.get("models", [])
        ]
    except OSError:
        return None


def main() -> None:
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": _cmd("git", "rev-parse", "HEAD"),
            "dirty": bool(_cmd("git", "status", "--porcelain")),
        },
        "platform": {
            "python": platform.python_version(),
            "system": platform.platform(),
            "machine": platform.machine(),
        },
        "packages_ml_env": {p: _version(p) for p in PACKAGES},
        "docker_mcp_server_image": _cmd(
            "docker",
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            "projeto-tcc-mcp-server:latest",
        ),
        "ollama_models": _ollama_models(),
        "inputs_sha256": {f: _sha256(ROOT / f) for f in INPUT_FILES},
        "results_sha256": {f: _sha256(ROOT / f) for f in RESULT_FILES},
    }
    OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
