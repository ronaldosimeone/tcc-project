"""
Unit tests for apps/ml/src/ingest_metropt.py.

All tests that touch the filesystem use pytest's `tmp_path` fixture so no
artefacts are written to the real data/ directory during the test run.

Design note
-----------
`validate_schema`, `preprocess` and `save_parquet` were merged into
`run_ingestion()` in the current implementation.  Tests that previously
targeted those helpers now exercise the same behaviour through
`run_ingestion()` with a mocked network call and a controlled CSV payload.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import src.ingest_metropt as ingest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_df() -> pd.DataFrame:
    """Return a minimal DataFrame that satisfies the expected MetroPT schema."""
    data: dict[str, list] = {col: [0.0] * 5 for col in ingest.EXPECTED_COLUMNS}
    data["timestamp"] = pd.date_range("2020-01-01", periods=5, freq="1s")
    return pd.DataFrame(data)


def _make_zip_bytes(
    csv_content: str, filename: str = "MetroPT3(AirCompressor).csv"
) -> bytes:
    """Pack *csv_content* into an in-memory ZIP archive and return its bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(filename, csv_content)
    return buf.getvalue()


def _mock_response(zip_bytes: bytes) -> MagicMock:
    """Build a requests.Response mock that streams *zip_bytes*."""
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.iter_content = MagicMock(return_value=[zip_bytes])
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def patched_paths(tmp_path: Path) -> Generator[dict[str, Path], None, None]:
    """
    Redirect all module-level path constants to a temporary directory so tests
    are fully isolated from the real data/ folder.

    The ZIP filename mirrors the constant defined in the module
    (RAW_ZIP_PATH = DATA_RAW_DIR / "metropt-3+dataset.zip").
    """
    raw_dir: Path = tmp_path / "data" / "raw"
    processed_dir: Path = tmp_path / "data" / "processed"
    zip_path: Path = raw_dir / "metropt-3+dataset.zip"
    csv_path: Path = raw_dir / "MetroPT3(AirCompressor).csv"
    parquet_path: Path = processed_dir / "metropt3.parquet"

    with (
        patch.object(ingest, "DATA_RAW_DIR", raw_dir),
        patch.object(ingest, "DATA_PROCESSED_DIR", processed_dir),
        patch.object(ingest, "RAW_ZIP_PATH", zip_path),
        patch.object(ingest, "RAW_CSV_PATH", csv_path),
        patch.object(ingest, "PROCESSED_PARQUET_PATH", parquet_path),
    ):
        yield {
            "raw_dir": raw_dir,
            "processed_dir": processed_dir,
            "zip_path": zip_path,
            "csv_path": csv_path,
            "parquet_path": parquet_path,
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
    ingest.ensure_directories()


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
    patched_paths["parquet_path"].parent.mkdir(parents=True, exist_ok=True)
    patched_paths["parquet_path"].touch()
    assert ingest.is_already_processed() is True


def test_run_ingestion_skips_when_parquet_exists(
    patched_paths: dict[str, Path]
) -> None:
    """
    If the processed Parquet already exists, run_ingestion() must return early
    without calling download_dataset().
    """
    patched_paths["parquet_path"].parent.mkdir(parents=True, exist_ok=True)
    patched_paths["parquet_path"].touch()

    with patch.object(ingest, "download_dataset") as mock_download:
        ingest.run_ingestion()
        mock_download.assert_not_called()


# ---------------------------------------------------------------------------
# EXPECTED_COLUMNS contract
# ---------------------------------------------------------------------------


def test_expected_columns_contains_critical_sensors() -> None:
    """EXPECTED_COLUMNS must contain the critical MetroPT sensor columns."""
    critical: list[str] = ["TP2", "TP3", "H1", "DV_pressure", "Motor_current"]
    for col in critical:
        assert (
            col in ingest.EXPECTED_COLUMNS
        ), f"Critical column '{col}' missing from EXPECTED_COLUMNS."


def test_expected_columns_includes_timestamp() -> None:
    """timestamp must be present as the first column."""
    assert ingest.EXPECTED_COLUMNS[0] == "timestamp"


# ---------------------------------------------------------------------------
# Schema validation — tested via run_ingestion (logic inlined there)
# ---------------------------------------------------------------------------


def test_run_ingestion_raises_on_missing_column(patched_paths: dict[str, Path]) -> None:
    """
    run_ingestion() must raise ValueError when the CSV is missing a required column.
    The missing-column check (formerly validate_schema) is embedded in run_ingestion.
    """
    # Build a CSV that lacks the 'TP2' column
    df_bad = _make_valid_df().drop(columns=["TP2"])
    zip_bytes = _make_zip_bytes(df_bad.to_csv(index=False))

    with patch("requests.get", return_value=_mock_response(zip_bytes)):
        with pytest.raises(ValueError, match="TP2"):
            ingest.run_ingestion()


def test_run_ingestion_succeeds_with_valid_schema(
    patched_paths: dict[str, Path]
) -> None:
    """run_ingestion() must complete without errors given a CSV with all required columns."""
    zip_bytes = _make_zip_bytes(_make_valid_df().to_csv(index=False))
    with patch("requests.get", return_value=_mock_response(zip_bytes)):
        ingest.run_ingestion()  # must not raise
    assert patched_paths["parquet_path"].exists()


# ---------------------------------------------------------------------------
# Preprocessing behaviour — tested via run_ingestion output
# ---------------------------------------------------------------------------


def test_run_ingestion_output_sorted_by_timestamp(
    patched_paths: dict[str, Path]
) -> None:
    """
    The saved Parquet must be sorted in ascending timestamp order.
    Injects a CSV with reversed timestamps to confirm the sort is applied.
    """
    df = _make_valid_df()
    df["timestamp"] = pd.date_range("2020-01-01", periods=5, freq="1s")[::-1]
    zip_bytes = _make_zip_bytes(df.to_csv(index=False))

    with patch("requests.get", return_value=_mock_response(zip_bytes)):
        ingest.run_ingestion()

    result = pd.read_parquet(patched_paths["parquet_path"])
    timestamps = result["timestamp"].tolist()
    assert timestamps == sorted(timestamps), "Parquet rows are not sorted by timestamp."


def test_run_ingestion_sensor_columns_are_float32(
    patched_paths: dict[str, Path]
) -> None:
    """Sensor columns in the saved Parquet must be dtype float32."""
    zip_bytes = _make_zip_bytes(_make_valid_df().to_csv(index=False))

    with patch("requests.get", return_value=_mock_response(zip_bytes)):
        ingest.run_ingestion()

    result = pd.read_parquet(patched_paths["parquet_path"])
    sensor_cols: list[str] = [c for c in ingest.EXPECTED_COLUMNS if c != "timestamp"]
    for col in sensor_cols:
        assert result[col].dtype == "float32", f"Column '{col}' is not float32."


# ---------------------------------------------------------------------------
# Output integrity — Parquet file tests
# ---------------------------------------------------------------------------


def test_run_ingestion_parquet_is_readable(patched_paths: dict[str, Path]) -> None:
    """The saved Parquet file must be loadable and have the correct shape."""
    df_input = _make_valid_df()
    zip_bytes = _make_zip_bytes(df_input.to_csv(index=False))

    with patch("requests.get", return_value=_mock_response(zip_bytes)):
        ingest.run_ingestion()

    loaded = pd.read_parquet(patched_paths["parquet_path"])
    assert len(loaded) == len(df_input)
    assert set(ingest.EXPECTED_COLUMNS).issubset(set(loaded.columns))


def test_run_ingestion_parquet_preserves_row_count(
    patched_paths: dict[str, Path]
) -> None:
    """Row count in the Parquet must equal the number of rows in the source CSV."""
    df_input = _make_valid_df()
    zip_bytes = _make_zip_bytes(df_input.to_csv(index=False))

    with patch("requests.get", return_value=_mock_response(zip_bytes)):
        ingest.run_ingestion()

    loaded = pd.read_parquet(patched_paths["parquet_path"])
    assert len(loaded) == len(
        df_input
    ), f"Expected {len(df_input)} rows, got {len(loaded)}."


# ---------------------------------------------------------------------------
# Full pipeline (mocked network) — end-to-end smoke test
# ---------------------------------------------------------------------------


def test_run_ingestion_full_pipeline(patched_paths: dict[str, Path]) -> None:
    """
    run_ingestion() must complete without errors when the network call is mocked
    to return a valid ZIP archive containing the expected CSV.
    """
    zip_bytes = _make_zip_bytes(_make_valid_df().to_csv(index=False))

    with patch("requests.get", return_value=_mock_response(zip_bytes)):
        ingest.run_ingestion()

    assert patched_paths[
        "parquet_path"
    ].exists(), "Parquet not created after full pipeline run."
