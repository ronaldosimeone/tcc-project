"""
MetroPT Dataset Ingestion Script.

Downloads or uses local MetroPT-3 dataset, validates schema,
creates the binary 'anomaly' target column from real failure intervals,
and persists the result as Parquet.
"""

from __future__ import annotations

import logging
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Link oficial que costuma oscilar na UCI
URL = "https://archive.ics.uci.edu/static/public/791/metropt-3+dataset.zip"

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

# Real failure intervals from the MetroPT-3 incident report.
# Each tuple is (start_inclusive, end_inclusive) in "YYYY-MM-DD HH:MM" format.
FAILURE_INTERVALS: list[tuple[str, str]] = [
    ("2020-04-18 00:00", "2020-04-18 23:59"),
    ("2020-05-29 23:30", "2020-05-30 06:00"),
    ("2020-06-05 10:00", "2020-06-07 14:30"),
    ("2020-07-15 14:30", "2020-07-15 19:00"),
]

# Paths
_MODULE_DIR: Path = Path(__file__).resolve().parent
_ML_ROOT: Path = _MODULE_DIR.parent
DATA_RAW_DIR: Path = _ML_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = _ML_ROOT / "data" / "processed"

# Ajustado para o nome real do download manual da UCI
RAW_ZIP_PATH: Path = DATA_RAW_DIR / "metropt-3+dataset.zip"
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
# Functions
# ---------------------------------------------------------------------------


def ensure_directories() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Diretórios prontos: %s", DATA_RAW_DIR)


def is_already_processed() -> bool:
    """Return True only when the Parquet already exists AND contains the 'anomaly' column."""
    if not PROCESSED_PARQUET_PATH.exists():
        return False
    try:
        pd.read_parquet(PROCESSED_PARQUET_PATH, columns=["anomaly"])
        logger.info(
            "Parquet com coluna 'anomaly' já existe em %s. Pulando...",
            PROCESSED_PARQUET_PATH,
        )
        return True
    except Exception:
        logger.info(
            "Parquet existe mas sem coluna 'anomaly'. Regenerando...",
        )
        return False


def label_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary 'anomaly' column by crossing timestamps with the real
    failure intervals from the MetroPT-3 incident report.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'timestamp' column of datetime dtype.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with a new 'anomaly' column (int8: 0 = normal, 1 = fault).
    """
    df = df.copy()
    df["anomaly"] = 0
    for start_str, end_str in FAILURE_INTERVALS:
        start: pd.Timestamp = pd.Timestamp(start_str)
        end: pd.Timestamp = pd.Timestamp(end_str)
        mask: pd.Series = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        df.loc[mask, "anomaly"] = 1

    n_anomalies: int = int(df["anomaly"].sum())
    logger.info(
        "Anomaly labels criados: %d anomalias / %d total (%.4f%%)",
        n_anomalies,
        len(df),
        n_anomalies / len(df) * 100,
    )
    df["anomaly"] = df["anomaly"].astype("int8")
    return df


def download_dataset() -> None:
    """Tenta baixar, mas pula se o arquivo manual já estiver lá."""
    if RAW_ZIP_PATH.exists():
        logger.info(
            "Arquivo ZIP encontrado localmente em %s. Pulando download.", RAW_ZIP_PATH
        )
        return

    logger.info("Tentando baixar dataset da UCI: %s", URL)
    try:
        response: requests.Response = requests.get(URL, stream=True, timeout=120)
        response.raise_for_status()
        with RAW_ZIP_PATH.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=8192):
                fh.write(chunk)
        logger.info("Download concluído com sucesso.")
    except Exception as e:
        logger.error("Falha no download automático: %s", e)
        logger.error(
            "POR FAVOR: Baixe o arquivo manualmente no site da UCI e coloque em: %s",
            RAW_ZIP_PATH,
        )
        sys.exit(1)


def extract_csv() -> None:
    if RAW_CSV_PATH.exists():
        logger.info("CSV já extraído em %s.", RAW_CSV_PATH)
        return

    if not RAW_ZIP_PATH.exists():
        raise FileNotFoundError(f"Arquivo ZIP não encontrado em {RAW_ZIP_PATH}")

    logger.info("Extraindo CSV do arquivo ZIP...")
    with zipfile.ZipFile(RAW_ZIP_PATH, "r") as zf:
        csv_members = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_members:
            raise FileNotFoundError("Nenhum CSV encontrado dentro do ZIP.")

        # Extrai o arquivo
        zf.extract(csv_members[0], DATA_RAW_DIR)
        extracted_path = DATA_RAW_DIR / csv_members[0]

        # Renomeia para o padrão esperado se necessário
        if extracted_path != RAW_CSV_PATH:
            if RAW_CSV_PATH.exists():
                RAW_CSV_PATH.unlink()
            extracted_path.rename(RAW_CSV_PATH)

    logger.info("Extração concluída: %s", RAW_CSV_PATH)


def run_ingestion() -> None:
    ensure_directories()

    if is_already_processed():
        return

    # Se você colocou o arquivo manualmente, o download_dataset vai apenas ignorar o erro 404
    download_dataset()

    extract_csv()

    logger.info("Lendo CSV e convertendo para Parquet (isso pode demorar um pouco)...")
    df = pd.read_csv(RAW_CSV_PATH, parse_dates=["timestamp"], low_memory=False)

    # Validação Básica
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas faltando no CSV: {missing}")

    # Otimização de Memória
    sensor_cols = [c for c in EXPECTED_COLUMNS if c != "timestamp"]
    df = df.sort_values("timestamp").reset_index(drop=True)
    df[sensor_cols] = df[sensor_cols].astype("float32")

    # Rotulagem de anomalias com base nos intervalos de falha reais
    df = label_anomalies(df)

    # Salva Parquet
    df.to_parquet(PROCESSED_PARQUET_PATH, index=False, compression="snappy")
    logger.info("Sucesso! Arquivo processado criado em: %s", PROCESSED_PARQUET_PATH)


if __name__ == "__main__":
    run_ingestion()
