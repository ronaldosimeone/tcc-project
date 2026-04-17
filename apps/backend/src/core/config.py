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

    # ── Ollama (local LLM) ────────────────────────────────────────────────
    ollama_base_url: str = Field(
        default="http://host.docker.internal:11434",
        alias="OLLAMA_BASE_URL",
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
        description="Active inference model: 'random_forest' or 'xgboost'.",
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


# Module-level singleton – import this directly instead of instantiating Settings elsewhere.
settings: Settings = Settings()
