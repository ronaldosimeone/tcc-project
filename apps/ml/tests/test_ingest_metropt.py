"""
Unit tests for apps/ml/src/ingest_metropt.py.

All tests that touch the filesystem use pytest's `tmp_path` fixture so no
artefacts are written to the real data/ directory during the test run.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# We need to redirect the path constants before import or patch them after.
# Patching is done per-test so we can control paths safely.
# ---------------------------------------------------------------------------
import apps.ml.src.ingest_metropt as ingest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_df() -> pd.DataFrame:
    """Return a minimal DataFrame that satisfies the expected MetroPT schema."""
    data: dict[str, list] = {col: [0.0] * 5 for col in ingest.EXPECTED_COLUMNS}
    data["timestamp"] = pd.date_range("2020-01-01", periods=5, freq="1s")
    return pd.DataFrame(data)


def _make_zip_bytes(csv_content: str) -> bytes:
    """Pack *csv_content* into an in-memory ZIP archive and return its bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("MetroPT3(AirCompressor).csv", csv_content)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def patched_paths(tmp_path: Path) -> Generator[dict[str, Path], None, None]:
    """
    Redirect all module-level path constants to a temporary directory so tests
    are fully isolated from the real data/ folder.
    """
    raw_dir: Path = tmp_path / "data" / "raw"
    processed_dir: Path = tmp_path / "data" / "processed"

    with (
        patch.object(ingest, "DATA_RAW_DIR", raw_dir),
        patch.object(ingest, "DATA_PROCESSED_DIR", processed_dir),
        patch.object(ingest, "RAW_ZIP_PATH", raw_dir / "metropt3.zip"),
        patch.object(ingest, "RAW_CSV_PATH", raw_dir / "MetroPT3(AirCompressor).csv"),
        patch.object(
            ingest, "PROCESSED_PARQUET_PATH", processed_dir / "metropt3.parquet"
        ),
    ):
        yield {
            "raw_dir": raw_dir,
            "processed_dir": processed_dir,
            "zip_path": raw_dir / "metropt3.zip",
            "csv_path": raw_dir / "MetroPT3(AirCompressor).csv",
            "parquet_path": processed_dir / "metropt3.parquet",
        }


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------


def test_ensure_directories_creates_raw_dir(patched_paths: dict[str, Path]) -> None:
    """ensure_directories() must create the raw data directory."""
    ingest.ensure_directories()
    assert patched_paths["raw_dir"].exists(), "Raw directory was not created."


def test_ensure_directories_creates_processed_dir(
    patched_paths: dict[str, Path]
) -> None:
    """ensure_directories() must create the processed data directory."""
    ingest.ensure_directories()
    assert patched_paths[
        "processed_dir"
    ].exists(), "Processed directory was not created."


def test_ensure_directories_is_idempotent(patched_paths: dict[str, Path]) -> None:
    """Calling ensure_directories() twice must not raise."""
    ingest.ensure_directories()
    ingest.ensure_directories()  # second call – should not raise


# ---------------------------------------------------------------------------
# Idempotency guard (RNF-05)
# ---------------------------------------------------------------------------


def test_is_already_processed_returns_false_when_parquet_missing(
    patched_paths: dict[str, Path],
) -> None:
    """is_already_processed() must return False when the Parquet file does not exist."""
    assert ingest.is_already_processed() is False


def test_is_already_processed_returns_true_when_parquet_exists(
    patched_paths: dict[str, Path],
) -> None:
    """is_already_processed() must return True when the Parquet file already exists."""
    parquet_path: Path = patched_paths["parquet_path"]
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.touch()

    assert ingest.is_already_processed() is True


def test_run_ingestion_skips_when_parquet_exists(
    patched_paths: dict[str, Path]
) -> None:
    """
    If the processed Parquet already exists, run_ingestion() must return early
    without calling download_dataset().
    """
    parquet_path: Path = patched_paths["parquet_path"]
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.touch()

    with patch.object(ingest, "download_dataset") as mock_download:
        ingest.run_ingestion()
        mock_download.assert_not_called()


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_expected_columns_contains_critical_sensors() -> None:
    """EXPECTED_COLUMNS must contain the critical MetroPT sensor columns."""
    critical: list[str] = ["TP2", "TP3", "H1", "DV_pressure", "Motor_current"]
    for col in critical:
        assert (
            col in ingest.EXPECTED_COLUMNS
        ), f"Critical column '{col}' missing from EXPECTED_COLUMNS."


def test_validate_schema_passes_for_valid_df() -> None:
    """validate_schema() must not raise when all expected columns are present."""
    df: pd.DataFrame = _make_valid_df()
    ingest.validate_schema(df)  # should not raise


def test_validate_schema_raises_for_missing_column() -> None:
    """validate_schema() must raise ValueError when a required column is absent."""
    df: pd.DataFrame = _make_valid_df().drop(columns=["TP2"])
    with pytest.raises(ValueError, match="TP2"):
        ingest.validate_schema(df)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def test_preprocess_sorts_by_timestamp() -> None:
    """preprocess() must return rows sorted in ascending timestamp order."""
    df: pd.DataFrame = _make_valid_df()
    df["timestamp"] = pd.date_range("2020-01-01", periods=5, freq="1s")[
        ::-1
    ]  # reverse order
    result: pd.DataFrame = ingest.preprocess(df)
    assert list(result["timestamp"]) == sorted(result["timestamp"].tolist())


def test_preprocess_sensor_columns_are_float32() -> None:
    """Sensor columns must be cast to float32 by preprocess()."""
    df: pd.DataFrame = _make_valid_df()
    result: pd.DataFrame = ingest.preprocess(df)
    sensor_cols: list[str] = [c for c in ingest.EXPECTED_COLUMNS if c != "timestamp"]
    for col in sensor_cols:
        assert result[col].dtype == "float32", f"Column '{col}' is not float32."


# ---------------------------------------------------------------------------
# Save Parquet
# ---------------------------------------------------------------------------


def test_save_parquet_creates_file(patched_paths: dict[str, Path]) -> None:
    """save_parquet() must persist the DataFrame as a .parquet file."""
    patched_paths["processed_dir"].mkdir(parents=True, exist_ok=True)
    df: pd.DataFrame = ingest.preprocess(_make_valid_df())
    ingest.save_parquet(df)
    assert patched_paths["parquet_path"].exists()


def test_save_parquet_is_readable(patched_paths: dict[str, Path]) -> None:
    """The saved Parquet file must be readable back into an equivalent DataFrame."""
    patched_paths["processed_dir"].mkdir(parents=True, exist_ok=True)
    df: pd.DataFrame = ingest.preprocess(_make_valid_df())
    ingest.save_parquet(df)

    loaded: pd.DataFrame = pd.read_parquet(patched_paths["parquet_path"])
    assert list(loaded.columns) == list(df.columns)
    assert len(loaded) == len(df)


# ---------------------------------------------------------------------------
# Full pipeline (mocked network)
# ---------------------------------------------------------------------------


def test_run_ingestion_full_pipeline(patched_paths: dict[str, Path]) -> None:
    """
    run_ingestion() must complete without errors when the network call is mocked
    to return a valid ZIP archive containing the expected CSV.
    """
    valid_df: pd.DataFrame = _make_valid_df()
    csv_bytes: str = valid_df.to_csv(index=False)
    zip_bytes: bytes = _make_zip_bytes(csv_bytes)

    mock_response: MagicMock = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_content = MagicMock(return_value=[zip_bytes])

    with patch("requests.get", return_value=mock_response):
        ingest.run_ingestion()

    assert patched_paths[
        "parquet_path"
    ].exists(), "Parquet not created after full pipeline run."
