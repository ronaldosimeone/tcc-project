"""
MetroPT Dataset Ingestion Script.

Downloads the MetroPT-3 dataset from the UCI Machine Learning Repository,
validates the schema, and persists it as Parquet for downstream processing.

Usage:
    python -m apps.ml.src.ingest_metropt
    # or from apps/ml/
    python src/ingest_metropt.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import requests  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# UCI ML Repository – MetroPT-3 Dataset (Air Compressor of a Metro Train)
# https://archive.ics.uci.edu/dataset/791/metropt-3+dataset
DATASET_URL: str = "https://archive.ics.uci.edu/static/public/791/metropt-3+dataset.zip"

# Columns defined by the MetroPT-3 paper / UCI description.
EXPECTED_COLUMNS: list[str] = [
    "timestamp",
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
]

# Paths (relative to this file's location so the script is location-agnostic)
_MODULE_DIR: Path = Path(__file__).resolve().parent
_ML_ROOT: Path = _MODULE_DIR.parent
DATA_RAW_DIR: Path = _ML_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = _ML_ROOT / "data" / "processed"

RAW_ZIP_PATH: Path = DATA_RAW_DIR / "metropt3.zip"
RAW_CSV_PATH: Path = DATA_RAW_DIR / "MetroPT3(AirCompressor).csv"
PROCESSED_PARQUET_PATH: Path = DATA_PROCESSED_DIR / "metropt3.parquet"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    stream=sys.stdout,
)
logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ensure_directories() -> None:
    """Create raw and processed data directories if they do not exist."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Data directories are ready: %s | %s", DATA_RAW_DIR, DATA_PROCESSED_DIR)


def is_already_processed() -> bool:
    """Return True when the processed Parquet file already exists (idempotency guard)."""
    exists: bool = PROCESSED_PARQUET_PATH.exists()
    if exists:
        logger.info(
            "Processed file already exists at %s – skipping ingestion.",
            PROCESSED_PARQUET_PATH,
        )
    return exists


def download_dataset() -> None:
    """Download the MetroPT-3 ZIP archive if not already present on disk."""
    if RAW_ZIP_PATH.exists():
        logger.info("Raw ZIP already cached at %s – skipping download.", RAW_ZIP_PATH)
        return

    logger.info("Downloading MetroPT-3 dataset from %s …", DATASET_URL)
    response: requests.Response = requests.get(DATASET_URL, stream=True, timeout=120)
    response.raise_for_status()

    with RAW_ZIP_PATH.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=8_192):
            fh.write(chunk)

    logger.info(
        "Download complete: %s (%.2f MB)",
        RAW_ZIP_PATH,
        RAW_ZIP_PATH.stat().st_size / 1e6,
    )


def extract_csv() -> None:
    """Extract the CSV from the downloaded ZIP archive."""
    if RAW_CSV_PATH.exists():
        logger.info(
            "Raw CSV already extracted at %s – skipping extraction.", RAW_CSV_PATH
        )
        return

    import zipfile

    logger.info("Extracting CSV from %s …", RAW_ZIP_PATH)
    with zipfile.ZipFile(RAW_ZIP_PATH, "r") as zf:
        csv_members: list[str] = [
            name for name in zf.namelist() if name.endswith(".csv")
        ]
        if not csv_members:
            raise FileNotFoundError("No CSV file found inside the ZIP archive.")
        # Extract the first (and only) CSV to raw dir
        zf.extract(csv_members[0], DATA_RAW_DIR)
        extracted_path: Path = DATA_RAW_DIR / csv_members[0]
        if extracted_path != RAW_CSV_PATH:
            extracted_path.rename(RAW_CSV_PATH)

    logger.info("Extraction complete: %s", RAW_CSV_PATH)


def load_raw_csv() -> pd.DataFrame:
    """Load the raw CSV into a DataFrame, parsing the timestamp column."""
    logger.info("Loading raw CSV from %s …", RAW_CSV_PATH)
    df: pd.DataFrame = pd.read_csv(
        RAW_CSV_PATH,
        parse_dates=["timestamp"],
        low_memory=False,
    )
    logger.info("Loaded %d rows × %d columns.", len(df), len(df.columns))
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if any expected column is missing from the DataFrame."""
    missing: list[str] = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Schema validation failed. Missing columns: {missing}\n"
            f"Present columns: {list(df.columns)}"
        )
    logger.info(
        "Schema validation passed – all %d expected columns present.",
        len(EXPECTED_COLUMNS),
    )


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply lightweight preprocessing steps:
    - Sort by timestamp.
    - Reset index after sort.
    - Coerce numeric sensor columns to float32 to reduce memory footprint.
    """
    sensor_cols: list[str] = [c for c in EXPECTED_COLUMNS if c != "timestamp"]

    df = df.sort_values("timestamp").reset_index(drop=True)
    df[sensor_cols] = df[sensor_cols].astype("float32")

    logger.info(
        "Preprocessing complete. Final shape: %d rows × %d columns.",
        len(df),
        len(df.columns),
    )
    return df


def save_parquet(df: pd.DataFrame) -> None:
    """Persist the processed DataFrame as Parquet (snappy compression)."""
    df.to_parquet(PROCESSED_PARQUET_PATH, index=False, compression="snappy")
    logger.info(
        "Saved Parquet to %s (%.2f MB).",
        PROCESSED_PARQUET_PATH,
        PROCESSED_PARQUET_PATH.stat().st_size / 1e6,
    )


def run_ingestion() -> None:
    """
    Orchestrate the full ingestion pipeline.

    Steps:
        1. Ensure output directories exist.
        2. Short-circuit if processed file already present (idempotency – RNF-05).
        3. Download raw ZIP.
        4. Extract CSV.
        5. Load → validate schema → preprocess → save Parquet.
    """
    ensure_directories()

    if is_already_processed():
        return

    download_dataset()
    extract_csv()

    df: pd.DataFrame = load_raw_csv()
    validate_schema(df)
    df = preprocess(df)
    save_parquet(df)

    logger.info("Ingestion pipeline finished successfully.")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_ingestion()
