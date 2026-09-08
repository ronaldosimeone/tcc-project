"""
Application settings managed via pydantic-settings.

All values can be overridden through environment variables or a .env file
placed at the project root.  Variable names are case-insensitive.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolved at import time: apps/backend/src/core/config.py → parents[3] = apps/
_ML_MODELS: Path = Path(__file__).resolve().parents[3] / "ml" / "models"
_DEFAULT_MODEL_PATH: Path = _ML_MODELS / "random_forest_final.joblib"
_DEFAULT_XGBOOST_PATH: Path = _ML_MODELS / "xgboost_v1.joblib"
_DEFAULT_MLP_ONNX_PATH: Path = _ML_MODELS / "mlp_v1.onnx"
_DEFAULT_MLP_SCALER_PATH: Path = _ML_MODELS / "mlp_scaler.joblib"
_DEFAULT_RF_V2_ONNX_PATH: Path = _ML_MODELS / "random_forest_v2.onnx"
_DEFAULT_XGB_V2_ONNX_PATH: Path = _ML_MODELS / "xgboost_v2.onnx"

# Sequential DL artefact paths — RNF-24 extension (TCN, BiLSTM, PatchTST).
# Override via TCN_ONNX_PATH / BILSTM_ONNX_PATH / PATCHTST_ONNX_PATH env-vars.
_DEFAULT_TCN_ONNX_PATH: Path = _ML_MODELS / "tcn_v1.onnx"
_DEFAULT_TCN_SCALER_PATH: Path = _ML_MODELS / "tcn_scaler.joblib"
_DEFAULT_BILSTM_ONNX_PATH: Path = _ML_MODELS / "bilstm_v1.onnx"
_DEFAULT_BILSTM_SCALER_PATH: Path = _ML_MODELS / "bilstm_scaler.joblib"
_DEFAULT_PATCHTST_ONNX_PATH: Path = _ML_MODELS / "patchtst_v1.onnx"
_DEFAULT_PATCHTST_SCALER_PATH: Path = _ML_MODELS / "patchtst_scaler.joblib"

# Unsupervised autoencoder artefact paths (Conv1D reconstruction-error model).
# Override via AUTOENCODER_ONNX_PATH / AUTOENCODER_SCALER_PATH env-vars.
_DEFAULT_AUTOENCODER_ONNX_PATH: Path = _ML_MODELS / "autoencoder_v1.onnx"
_DEFAULT_AUTOENCODER_SCALER_PATH: Path = _ML_MODELS / "autoencoder_scaler.joblib"


class Settings(BaseSettings):
    """Typed application configuration backed by environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Project metadata ──────────────────────────────────────────────────
    project_name: str = Field(default="Predictive Maintenance API")
    version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)

    # ── Database ──────────────────────────────────────────────────────────
    # Expected format: postgresql+asyncpg://user:password@host:port/dbname
    postgres_url: PostgresDsn = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/tcc_db",
        alias="DATABASE_URL",
    )

    # RNF-37 — pool tunável via .env para o load test (locust_streaming.py /
    # README §14.7). Defaults preservam o comportamento anterior (antes
    # hardcoded 10/20 diretamente em database.py).
    db_pool_size: int = Field(
        default=10,
        alias="DB_POOL_SIZE",
        description="SQLAlchemy async engine pool_size (conexões persistentes por worker).",
    )
    db_max_overflow: int = Field(
        default=20,
        alias="DB_MAX_OVERFLOW",
        description="SQLAlchemy async engine max_overflow (conexões extras sob pico).",
    )

    # ── Ollama (local LLM) — RF-22 / RNF-46 ─────────────────────────────────
    ollama_base_url: str = Field(
        default="http://host.docker.internal:11434",
        alias="OLLAMA_BASE_URL",
    )
    # Modelo usado por MaintenanceSuggestionService (src/services/ollama_client.py).
    # Default = Llama 3.2 3B (CLAUDE.md §4) — configurável sem redeploy.
    ollama_model: str = Field(
        default="llama3.2:3b",
        alias="OLLAMA_MODEL",
    )
    # Timeout do cliente HTTP do Ollama — generoso porque um Llama 3.2 3B
    # rodando em CPU pode levar dezenas de segundos para um plano completo.
    ollama_client_timeout_seconds: float = Field(
        default=120.0,
        alias="OLLAMA_CLIENT_TIMEOUT_SECONDS",
    )

    # RF-19 / RNF-43 — endereço do serviço MCP standalone (apps/mcp-server),
    # container e processo próprios (ver docker-compose.yml::mcp-server).
    # Consumida desde a RF-22 por MaintenanceSuggestionService
    # (src/services/mcp_client.py) — chama a tool `search_maintenance_manual`
    # via streamable-http (SDK oficial `mcp`, ver requirements.txt).
    mcp_server_url: str = Field(
        default="http://mcp-server:8100",
        alias="MCP_SERVER_URL",
    )
    # Timeout do cliente MCP — generoso porque a primeira chamada após o
    # mcp-server subir aciona o carregamento do SentenceTransformer
    # (~segundos a ~30s a frio, ver apps/mcp-server/semantic_search.py).
    mcp_client_timeout_seconds: float = Field(
        default=60.0,
        alias="MCP_CLIENT_TIMEOUT_SECONDS",
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        alias="ALLOWED_ORIGINS",
    )

    # ── Machine Learning ──────────────────────────────────────────────────
    # Override via MODEL_PATH env-var when deploying in Docker or CI.
    # Default resolves to apps/ml/models/random_forest_final.joblib inside the monorepo.
    model_path: Path = Field(
        default=_DEFAULT_MODEL_PATH,
        alias="MODEL_PATH",
        description="Absolute path to the trained Random Forest .joblib artefact.",
    )

    # RF-10 — multi-model selection without redeploy.
    # Set ACTIVE_MODEL=xgboost to swap the inference engine at startup.
    active_model: str = Field(
        default="random_forest",
        alias="ACTIVE_MODEL",
        description=(
            "Active inference model: 'random_forest', 'xgboost', 'mlp', "
            "'random_forest_v2', 'xgboost_v2', 'tcn', 'bilstm', 'patchtst', "
            "or 'autoencoder'."
        ),
    )

    # RF-11 — admin token for /models management endpoints.
    # MUST be overridden in production via the ADMIN_API_TOKEN environment variable.
    admin_api_token: str = Field(
        default="change-me-in-production",
        alias="ADMIN_API_TOKEN",
        description="Bearer token required for all /models admin endpoints.",
    )

    xgboost_model_path: Path = Field(
        default=_DEFAULT_XGBOOST_PATH,
        alias="XGBOOST_MODEL_PATH",
        description="Absolute path to the trained XGBoost .joblib artefact.",
    )

    # RNF-24 — MLP via ONNX Runtime.
    # Set ACTIVE_MODEL=mlp to use the neural network inference engine.
    mlp_onnx_path: Path = Field(
        default=_DEFAULT_MLP_ONNX_PATH,
        alias="MLP_ONNX_PATH",
        description="Absolute path to the trained MLP .onnx artefact.",
    )

    mlp_scaler_path: Path = Field(
        default=_DEFAULT_MLP_SCALER_PATH,
        alias="MLP_SCALER_PATH",
        description="Absolute path to the StandardScaler .joblib artefact for the MLP.",
    )

    # RF-10 V2 — tree-based ONNX artefacts (skl2onnx / onnxmltools exports).
    # Loaded by OnnxTreeAdapter; no scaler needed (tree ensembles).
    rf_v2_onnx_path: Path = Field(
        default=_DEFAULT_RF_V2_ONNX_PATH,
        alias="RF_V2_ONNX_PATH",
        description="Absolute path to the Random Forest V2 .onnx artefact.",
    )

    xgboost_v2_onnx_path: Path = Field(
        default=_DEFAULT_XGB_V2_ONNX_PATH,
        alias="XGBOOST_V2_ONNX_PATH",
        description="Absolute path to the XGBoost V2 .onnx artefact.",
    )

    # ── Sequential DL models (TCN / BiLSTM / PatchTST) ───────────────────
    tcn_onnx_path: Path = Field(
        default=_DEFAULT_TCN_ONNX_PATH,
        alias="TCN_ONNX_PATH",
        description="Absolute path to the TCN .onnx artefact.",
    )
    tcn_scaler_path: Path = Field(
        default=_DEFAULT_TCN_SCALER_PATH,
        alias="TCN_SCALER_PATH",
        description="Absolute path to the TCN per-channel StandardScaler.",
    )
    bilstm_onnx_path: Path = Field(
        default=_DEFAULT_BILSTM_ONNX_PATH,
        alias="BILSTM_ONNX_PATH",
        description="Absolute path to the BiLSTM .onnx artefact.",
    )
    bilstm_scaler_path: Path = Field(
        default=_DEFAULT_BILSTM_SCALER_PATH,
        alias="BILSTM_SCALER_PATH",
        description="Absolute path to the BiLSTM per-channel StandardScaler.",
    )
    patchtst_onnx_path: Path = Field(
        default=_DEFAULT_PATCHTST_ONNX_PATH,
        alias="PATCHTST_ONNX_PATH",
        description="Absolute path to the PatchTST .onnx artefact.",
    )
    patchtst_scaler_path: Path = Field(
        default=_DEFAULT_PATCHTST_SCALER_PATH,
        alias="PATCHTST_SCALER_PATH",
        description="Absolute path to the PatchTST per-channel StandardScaler.",
    )

    # ── Simulator data source ─────────────────────────────────────────────
    # Points to the processed MetroPT-3 parquet streamed by SensorSimulator.
    # Override via SIMULATOR_PARQUET_PATH env-var (e.g., in Docker Compose).
    simulator_parquet_path: Path = Field(
        default=Path(__file__).resolve().parents[3]
        / "ml"
        / "data"
        / "processed"
        / "metropt3.parquet",
        alias="SIMULATOR_PARQUET_PATH",
        description="Absolute path to the processed MetroPT-3 parquet file.",
    )

    # ── Unsupervised Conv1D Autoencoder ───────────────────────────────────
    autoencoder_onnx_path: Path = Field(
        default=_DEFAULT_AUTOENCODER_ONNX_PATH,
        alias="AUTOENCODER_ONNX_PATH",
        description="Absolute path to the Conv1D Autoencoder .onnx artefact.",
    )
    autoencoder_scaler_path: Path = Field(
        default=_DEFAULT_AUTOENCODER_SCALER_PATH,
        alias="AUTOENCODER_SCALER_PATH",
        description="Absolute path to the Autoencoder per-channel StandardScaler.",
    )


# Module-level singleton – import this directly instead of instantiating Settings elsewhere.
settings: Settings = Settings()
