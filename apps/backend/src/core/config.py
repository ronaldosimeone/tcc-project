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
_DEFAULT_MODEL_PATH: Path = (
    Path(__file__).resolve().parents[3] / "ml" / "models" / "rf_model.joblib"
)


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
    # Default resolves to apps/ml/models/rf_model.joblib inside the monorepo.
    model_path: Path = Field(
        default=_DEFAULT_MODEL_PATH,
        alias="MODEL_PATH",
        description="Absolute path to the trained Random Forest .joblib artefact.",
    )


# Module-level singleton – import this directly instead of instantiating Settings elsewhere.
settings: Settings = Settings()
