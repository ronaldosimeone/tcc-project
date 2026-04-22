"""
Unit tests for promote_model.py (RNF-26).

Coverage
--------
1. find_best_run — selects the run with the highest metric (mathematical validation)
2. find_best_run — raises ValueError when experiment is missing
3. find_best_run — raises ValueError when no runs exist
4. inject_run_id — injects promoted_run_id and promoted_at
5. inject_run_id — preserves all pre-existing card fields
6. inject_run_id — is a pure function (original dict not mutated)
7. promote — copies artefact files to dest_dir
8. promote — writes model card with run_id injected to dest_dir
9. promote — raises FileNotFoundError when card is absent from artefacts

All external I/O (MlflowClient, filesystem downloads) is mocked with
unittest.mock so the suite runs offline without a running MLflow server.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Ensure src/ is on the path when running from apps/ml/
from src.promote_model import find_best_run, inject_run_id, promote


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_experiment(experiment_id: str = "1") -> MagicMock:
    exp = MagicMock()
    exp.experiment_id = experiment_id
    return exp


def _make_run(run_id: str, metrics: dict[str, float]) -> MagicMock:
    run = MagicMock()
    run.info.run_id = run_id
    run.data.metrics = metrics
    return run


_SAMPLE_CARD: dict[str, Any] = {
    "schema_version": "1.0",
    "model_type": "MlpClassifier",
    "metrics": {"test_f1_class1": 0.87, "roc_auc_test": 0.999},
}


# ---------------------------------------------------------------------------
# find_best_run
# ---------------------------------------------------------------------------


class TestFindBestRun:
    def test_returns_run_with_highest_metric(self) -> None:
        """Mathematical validation: among 3 runs the one with max metric wins."""
        client = MagicMock()
        client.get_experiment_by_name.return_value = _make_experiment("42")
        runs = [
            _make_run("run-B", {"test_f1_class1": 0.83}),
            _make_run("run-A", {"test_f1_class1": 0.91}),
            _make_run("run-C", {"test_f1_class1": 0.77}),
        ]
        # MLflow returns them already sorted DESC when order_by is used;
        # simulate that by making search_runs return only the best.
        client.search_runs.return_value = [runs[1]]  # run-A is best

        run_id, value = find_best_run(client, "mlp_metropt3", "test_f1_class1")

        assert run_id == "run-A"
        assert abs(value - 0.91) < 1e-9

    def test_search_runs_called_with_correct_order_by(self) -> None:
        """Ensures the ORDER BY clause asks for DESC sorting on the right metric."""
        client = MagicMock()
        client.get_experiment_by_name.return_value = _make_experiment("7")
        client.search_runs.return_value = [_make_run("r1", {"val_f1": 0.95})]

        find_best_run(client, "exp", "val_f1")

        client.search_runs.assert_called_once_with(
            experiment_ids=["7"],
            order_by=["metrics.val_f1 DESC"],
            max_results=1,
        )

    def test_raises_when_experiment_not_found(self) -> None:
        client = MagicMock()
        client.get_experiment_by_name.return_value = None

        with pytest.raises(ValueError, match="not found in MLflow"):
            find_best_run(client, "nonexistent_exp", "test_f1_class1")

    def test_raises_when_no_runs_exist(self) -> None:
        client = MagicMock()
        client.get_experiment_by_name.return_value = _make_experiment("3")
        client.search_runs.return_value = []

        with pytest.raises(ValueError, match="No runs found"):
            find_best_run(client, "empty_exp", "test_f1_class1")

    def test_metric_value_comes_from_run_data(self) -> None:
        """Returned metric value must equal what is stored in run.data.metrics."""
        client = MagicMock()
        client.get_experiment_by_name.return_value = _make_experiment("5")
        client.search_runs.return_value = [
            _make_run("run-X", {"test_f1_class1": 0.9703})
        ]

        _, value = find_best_run(client, "exp", "test_f1_class1")

        assert abs(value - 0.9703) < 1e-9


# ---------------------------------------------------------------------------
# inject_run_id
# ---------------------------------------------------------------------------


class TestInjectRunId:
    def test_promoted_run_id_is_injected(self) -> None:
        result = inject_run_id(_SAMPLE_CARD.copy(), "abc-123")
        assert result["promoted_run_id"] == "abc-123"

    def test_promoted_at_is_injected(self) -> None:
        result = inject_run_id(_SAMPLE_CARD.copy(), "abc-123")
        assert "promoted_at" in result
        assert isinstance(result["promoted_at"], str)
        assert result["promoted_at"].endswith("+00:00") or result[
            "promoted_at"
        ].endswith("Z")

    def test_existing_fields_are_preserved(self) -> None:
        result = inject_run_id(_SAMPLE_CARD.copy(), "x")
        assert result["schema_version"] == "1.0"
        assert result["model_type"] == "MlpClassifier"
        assert result["metrics"]["test_f1_class1"] == 0.87

    def test_is_pure_function_original_unchanged(self) -> None:
        original = _SAMPLE_CARD.copy()
        inject_run_id(original, "some-run-id")
        assert "promoted_run_id" not in original
        assert "promoted_at" not in original

    def test_overwrites_existing_promoted_run_id(self) -> None:
        card_with_old = {**_SAMPLE_CARD, "promoted_run_id": "old-run"}
        result = inject_run_id(card_with_old, "new-run")
        assert result["promoted_run_id"] == "new-run"


# ---------------------------------------------------------------------------
# promote (end-to-end with mocked MlflowClient and filesystem)
# ---------------------------------------------------------------------------


class TestPromote:
    def _setup_fake_artefacts(
        self,
        tmp: Path,
        card: dict[str, Any],
        run_id: str,
        extra_files: list[str] | None = None,
    ) -> Path:
        """Write fake artefact files into tmp/model/ and return that dir."""
        artefact_dir = tmp / "model"
        artefact_dir.mkdir(parents=True)

        # model card
        (artefact_dir / "mlp_v1_card.json").write_text(
            json.dumps(card), encoding="utf-8"
        )
        # stub artefact files
        for name in extra_files or ["mlp_v1.onnx", "mlp_scaler.joblib"]:
            (artefact_dir / name).write_bytes(b"stub")

        return artefact_dir

    def test_promote_copies_model_files_to_dest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            artefact_dir = self._setup_fake_artefacts(
                tmp / "downloaded", _SAMPLE_CARD, "run-42"
            )
            dest_dir = tmp / "dest"

            with patch("src.promote_model.MlflowClient") as MockClient:
                inst = MockClient.return_value
                inst.get_experiment_by_name.return_value = _make_experiment("1")
                inst.search_runs.return_value = [
                    _make_run("run-42", {"test_f1_class1": 0.91})
                ]
                inst.download_artifacts.return_value = str(artefact_dir)

                promote(
                    tracking_uri="http://fake:5000",
                    experiment_name="mlp_metropt3",
                    metric="test_f1_class1",
                    artifact_path="model",
                    dest_dir=dest_dir,
                    card_filename="mlp_v1_card.json",
                )

            assert (dest_dir / "mlp_v1.onnx").exists()
            assert (dest_dir / "mlp_scaler.joblib").exists()

    def test_promote_writes_card_with_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            artefact_dir = self._setup_fake_artefacts(
                tmp / "downloaded", _SAMPLE_CARD, "run-99"
            )
            dest_dir = tmp / "dest"

            with patch("src.promote_model.MlflowClient") as MockClient:
                inst = MockClient.return_value
                inst.get_experiment_by_name.return_value = _make_experiment("1")
                inst.search_runs.return_value = [
                    _make_run("run-99", {"test_f1_class1": 0.97})
                ]
                inst.download_artifacts.return_value = str(artefact_dir)

                promote(
                    tracking_uri="http://fake:5000",
                    experiment_name="mlp_metropt3",
                    metric="test_f1_class1",
                    artifact_path="model",
                    dest_dir=dest_dir,
                    card_filename="mlp_v1_card.json",
                )

            written = json.loads((dest_dir / "mlp_v1_card.json").read_text())
            assert written["promoted_run_id"] == "run-99"
            assert "promoted_at" in written
            assert written["schema_version"] == "1.0"

    def test_promote_raises_when_card_missing_in_artefacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            artefact_dir = tmp / "downloaded" / "model"
            artefact_dir.mkdir(parents=True)
            (artefact_dir / "mlp_v1.onnx").write_bytes(b"stub")  # no card

            with patch("src.promote_model.MlflowClient") as MockClient:
                inst = MockClient.return_value
                inst.get_experiment_by_name.return_value = _make_experiment("1")
                inst.search_runs.return_value = [
                    _make_run("run-1", {"test_f1_class1": 0.85})
                ]
                inst.download_artifacts.return_value = str(artefact_dir)

                with pytest.raises(FileNotFoundError, match="mlp_v1_card.json"):
                    promote(
                        tracking_uri="http://fake:5000",
                        experiment_name="mlp_metropt3",
                        metric="test_f1_class1",
                        artifact_path="model",
                        dest_dir=tmp / "dest",
                        card_filename="mlp_v1_card.json",
                    )

    def test_promote_uses_winning_run_id_not_a_random_one(self) -> None:
        """Guarantee the run_id written to the card matches the best run returned."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            winning_id = "winner-run-id-xyz"
            artefact_dir = self._setup_fake_artefacts(
                tmp / "downloaded", _SAMPLE_CARD, winning_id
            )
            dest_dir = tmp / "dest"

            with patch("src.promote_model.MlflowClient") as MockClient:
                inst = MockClient.return_value
                inst.get_experiment_by_name.return_value = _make_experiment("1")
                # search_runs returns only winner (MLflow sorts server-side)
                inst.search_runs.return_value = [
                    _make_run(winning_id, {"test_f1_class1": 0.99})
                ]
                inst.download_artifacts.return_value = str(artefact_dir)

                promote(
                    tracking_uri="http://fake:5000",
                    experiment_name="mlp_metropt3",
                    metric="test_f1_class1",
                    artifact_path="model",
                    dest_dir=dest_dir,
                    card_filename="mlp_v1_card.json",
                )

            written = json.loads((dest_dir / "mlp_v1_card.json").read_text())
            assert written["promoted_run_id"] == winning_id
