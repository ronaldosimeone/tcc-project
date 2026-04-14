"""
Pydantic v2 schemas (DTOs) for the health-check endpoint.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DatabaseStatus(BaseModel):
    """Result of the live database connectivity probe."""

    connected: bool
    latency_ms: float | None = Field(
        default=None, description="Round-trip time in milliseconds"
    )
    error: str | None = Field(
        default=None, description="Error message when connected=False"
    )


class HealthResponse(BaseModel):
    """Response body returned by GET /health."""

    status: str = Field(description="'ok' when the service is fully operational")
    version: str
    database: DatabaseStatus
