"""
MetroPT Dataset Ingestion Script (Fixed Version).

Downloads or uses local MetroPT-3 dataset, validates schema, 
and persists it as Parquet.
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
    if PROCESSED_PARQUET_PATH.exists():
        logger.info(
            "Arquivo Parquet já existe em %s. Pulando...", PROCESSED_PARQUET_PATH
        )
        return True
    return False


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

    # Salva Parquet
    df.to_parquet(PROCESSED_PARQUET_PATH, index=False, compression="snappy")
    logger.info("Sucesso! Arquivo processado criado em: %s", PROCESSED_PARQUET_PATH)


if __name__ == "__main__":
    run_ingestion()
