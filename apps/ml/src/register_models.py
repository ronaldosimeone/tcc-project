"""
MLflow Model Registry — champion / challenger with the RN-05 gate.

Entry point:
    MLFLOW_TRACKING_URI=http://localhost:5000 python -m src.register_models \
        --champion random_forest_v2 --challenger xgboost_v2

For each model: log the ONNX artefact (+ model card, + the leakage-free
temporal metrics of ``models/eval_temporal.json``) in a run, register it as
a new version of ``predictiq-failure-detector`` and set the alias
(``champion`` / ``challenger``). The RN-05 promotion-gate decision (paired
one-sided Wilcoxon over temporal folds) is attached to the challenger
version as tags — so the registry itself records WHY the challenger is not
(yet) champion.

The backend consumes the same names: ``ACTIVE_MODEL`` = champion,
``CHALLENGER_MODEL`` = challenger (shadow on the main asset + canary share of
the fleet, see apps/backend/src/services/shadow_service.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

_ML_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ML_ROOT))

from src.promotion_gate import decide  # noqa: E402

MODELS_DIR: Path = _ML_ROOT / "models"
REGISTERED_NAME = "predictiq-failure-detector"
EVAL_JSON: Path = MODELS_DIR / "eval_temporal.json"
# RF-33: evidence for models trained on both recording periods.
CROSS_JSON: Path = MODELS_DIR / "eval_cross_dataset_harmonized.json"
# backend model name -> evaluate_temporal model name (temporal metrics)
_EVAL_NAME: dict[str, str] = {
    "random_forest_v2": "random_forest",
    "random_forest": "random_forest",
    "xgboost_v2": "xgboost",
    "xgboost_v3": "xgboost",
    "xgboost": "xgboost",
}


def _artefacts(name: str) -> list[Path]:
    files = [
        MODELS_DIR / f"{name}.onnx",
        MODELS_DIR / f"{name}.onnx.data",
        MODELS_DIR / f"{name}_card.json",
    ]
    found = [f for f in files if f.exists()]
    if not any(f.suffix == ".onnx" for f in found):
        raise FileNotFoundError(f"No ONNX artefact for {name!r} in {MODELS_DIR}")
    return found


def _temporal_metrics(name: str, eval_data: dict[str, Any] | None) -> dict[str, float]:
    if eval_data is None or name not in _EVAL_NAME:
        return {}
    per_model = eval_data["protocols"]["walk_forward"]["summary"]["per_model"].get(
        _EVAL_NAME[name], {}
    )
    return {f"wf_{k}": float(v["mean"]) for k, v in per_model.items() if v.get("n")}


def register(
    name: str, alias: str, client: MlflowClient, eval_data: dict[str, Any] | None
) -> str:
    with mlflow.start_run(run_name=f"register-{name}") as run:
        for path in _artefacts(name):
            mlflow.log_artifact(str(path), artifact_path="model")
        mlflow.set_tags(
            {"backend_model_name": name, "format": "onnx", "requirement": "RF-10/RN-05"}
        )
        metrics = _temporal_metrics(name, eval_data)
        if metrics:
            mlflow.log_metrics(metrics)
        version = mlflow.register_model(
            f"runs:/{run.info.run_id}/model", REGISTERED_NAME
        )
    client.set_registered_model_alias(REGISTERED_NAME, alias, version.version)
    client.set_model_version_tag(
        REGISTERED_NAME, version.version, "backend_model_name", name
    )
    print(f"{name} -> {REGISTERED_NAME} v{version.version} @{alias}")
    return str(version.version)


def _load_evidence(kind: str) -> dict[str, Any] | None:
    """Both evidence files are exposed under the ``walk_forward`` protocol key
    that ``_temporal_metrics`` and ``decide`` read."""
    if kind == "temporal":
        return (
            json.loads(EVAL_JSON.read_text(encoding="utf-8"))
            if EVAL_JSON.exists()
            else None
        )
    if not CROSS_JSON.exists():
        return None
    pooled = json.loads(CROSS_JSON.read_text(encoding="utf-8"))["pooled_walk_forward"]
    return {"protocols": {"walk_forward": pooled}}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--champion", required=True)
    parser.add_argument(
        "--evidence",
        choices=["temporal", "cross_harmonized"],
        default="temporal",
        help="temporal = MetroPT-3 walk-forward; cross_harmonized = pooled "
        "MetroPT-3 + MetroPT 2022 walk-forward (RF-33)",
    )
    parser.add_argument(
        "--challenger-only",
        action="store_true",
        help="register only the challenger; keep the current champion version",
    )
    parser.add_argument("--challenger", required=True)
    parser.add_argument("--experiment", default="predictiq-registry")
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment)
    client = MlflowClient()
    eval_data = _load_evidence(args.evidence)

    if not args.challenger_only:
        register(args.champion, "champion", client, eval_data)
    challenger_version = register(args.challenger, "challenger", client, eval_data)

    if (
        eval_data is not None
        and args.champion in _EVAL_NAME
        and args.challenger in _EVAL_NAME
    ):
        gate = decide(eval_data, _EVAL_NAME[args.challenger], _EVAL_NAME[args.champion])
        for key, value in {
            "rn05_promote": str(gate.promote).lower(),
            "rn05_p_value": f"{gate.p_value:.4f}",
            "rn05_median_improvement": f"{gate.median_improvement:+.4f}",
            "rn05_reason": gate.reason[:250],
            "rn05_evidence": args.evidence,
        }.items():
            client.set_model_version_tag(
                REGISTERED_NAME, challenger_version, key, value
            )
        print(f"RN-05 gate: promote={gate.promote} — {gate.reason}")


if __name__ == "__main__":
    main()
