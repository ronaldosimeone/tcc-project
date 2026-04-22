"""
Model promotion script (RNF-26, RNF-27).

Connects to an MLflow tracking server, finds the best run in a given
experiment by a configurable metric, downloads its artefacts, injects
the winning run_id into the model card, and copies everything to the
destination directory consumed by the backend.

Usage
-----
    # With MLflow running locally (docker compose up mlflow)
    python src/promote_model.py

    # Override defaults
    python src/promote_model.py \\
        --experiment mlp_metropt3 \\
        --metric test_f1_class1 \\
        --tracking-uri http://localhost:5000 \\
        --dest-dir models/

Environment variables
---------------------
    MLFLOW_TRACKING_URI   overrides --tracking-uri when set
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlflow.tracking import MlflowClient

_HERE: Path = Path(__file__).resolve().parent  # apps/ml/src/
_ML_ROOT: Path = _HERE.parent  # apps/ml/
_DEFAULT_DEST: Path = _ML_ROOT / "models"
_DEFAULT_EXPERIMENT: str = "mlp_metropt3"
_DEFAULT_METRIC: str = "test_f1_class1"
_DEFAULT_TRACKING_URI: str = "http://localhost:5000"
_DEFAULT_ARTIFACT_PATH: str = "model"
_DEFAULT_CARD_FILENAME: str = "mlp_v1_card.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def find_best_run(
    client: MlflowClient,
    experiment_name: str,
    metric: str,
) -> tuple[str, float]:
    """Return (run_id, metric_value) for the run with the highest metric.

    Raises:
        ValueError: if the experiment does not exist or has no runs.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in MLflow.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'.")

    best = runs[0]
    metric_value: float = best.data.metrics.get(metric, float("nan"))
    log.info(
        "Best run: run_id=%s | %s=%.4f",
        best.info.run_id,
        metric,
        metric_value,
    )
    return best.info.run_id, metric_value


def download_artefacts(
    client: MlflowClient,
    run_id: str,
    artifact_path: str,
    dst_path: Path,
) -> Path:
    """Download artefacts for a run to dst_path and return the local directory."""
    local = client.download_artifacts(
        run_id=run_id,
        path=artifact_path,
        dst_path=str(dst_path),
    )
    return Path(local)


def inject_run_id(card: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Return a new model card dict with run_id and promoted_at injected (RNF-26).

    This is a pure function — the original dict is not mutated.
    """
    return {
        **card,
        "promoted_run_id": run_id,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }


def promote(
    tracking_uri: str,
    experiment_name: str,
    metric: str,
    artifact_path: str,
    dest_dir: Path,
    card_filename: str,
) -> None:
    """Full promotion pipeline: find → download → inject run_id → deploy."""
    client = MlflowClient(tracking_uri=tracking_uri)

    run_id, metric_value = find_best_run(client, experiment_name, metric)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        log.info("Downloading artefacts for run %s …", run_id)
        local_artefacts = download_artefacts(client, run_id, artifact_path, tmp_path)

        card_path = local_artefacts / card_filename
        if not card_path.exists():
            raise FileNotFoundError(
                f"'{card_filename}' not found in downloaded artefacts at {local_artefacts}. "
                "Re-run train_mlp.py to register the model card as an MLflow artefact."
            )

        with card_path.open("r", encoding="utf-8") as fh:
            card: dict[str, Any] = json.load(fh)

        card = inject_run_id(card, run_id)

        dest_dir.mkdir(parents=True, exist_ok=True)

        for src_file in local_artefacts.iterdir():
            if not src_file.is_file():
                continue  # ignora subdiretórios como checkpoints/
            if src_file.name == card_filename:
                continue  # written separately after run_id injection
            dest_file = dest_dir / src_file.name
            shutil.copy2(src_file, dest_file)
            log.info("  %s → %s", src_file.name, dest_file)

        dest_card = dest_dir / card_filename
        dest_card.write_text(json.dumps(card, indent=2), encoding="utf-8")
        log.info("Model card saved with promoted_run_id=%s → %s", run_id, dest_card)

    log.info(
        "Promotion complete: %s=%.4f | dest=%s",
        metric,
        metric_value,
        dest_dir,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Promote the best MLflow run to the production models directory."
    )
    p.add_argument(
        "--tracking-uri",
        default=_DEFAULT_TRACKING_URI,
        help=f"MLflow tracking server URI (default: {_DEFAULT_TRACKING_URI}).",
    )
    p.add_argument(
        "--experiment",
        default=_DEFAULT_EXPERIMENT,
        help=f"Experiment name (default: {_DEFAULT_EXPERIMENT}).",
    )
    p.add_argument(
        "--metric",
        default=_DEFAULT_METRIC,
        help=f"Metric to maximise when selecting the best run (default: {_DEFAULT_METRIC}).",
    )
    p.add_argument(
        "--artifact-path",
        default=_DEFAULT_ARTIFACT_PATH,
        help=f"Artefact sub-path inside the run (default: '{_DEFAULT_ARTIFACT_PATH}').",
    )
    p.add_argument(
        "--dest-dir",
        type=Path,
        default=_DEFAULT_DEST,
        help=f"Destination directory for promoted artefacts (default: {_DEFAULT_DEST}).",
    )
    p.add_argument(
        "--card-filename",
        default=_DEFAULT_CARD_FILENAME,
        help=f"Model card filename inside the artefact path (default: {_DEFAULT_CARD_FILENAME}).",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    promote(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment,
        metric=args.metric,
        artifact_path=args.artifact_path,
        dest_dir=args.dest_dir,
        card_filename=args.card_filename,
    )
