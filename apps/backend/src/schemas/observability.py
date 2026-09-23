"""
Pydantic v2 schemas (DTOs) — RNF-77, taxa de erro consultada do Prometheus.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ErrorRateStatus = Literal["NORMAL", "WARNING", "CRITICAL"]


class ErrorRateResponse(BaseModel):
    """Resposta de `GET /observability/error-rate` — consumida pelo Dashboard."""

    status: ErrorRateStatus = Field(
        description="NORMAL/WARNING/CRITICAL conforme os limiares configurados."
    )
    error_rate: float = Field(
        description="Fração de requisições HTTP 5xx sobre o total, na janela (0.0–1.0)."
    )
    window: str = Field(description="Janela de tempo do rate() PromQL, ex. '5m'.")
    threshold_warning: float
    threshold_critical: float
    prometheus_reachable: bool = Field(
        description=(
            "False quando o Prometheus não respondeu — nesse caso `status` é "
            "sempre NORMAL por padrão seguro (nunca dispara alerta a partir de "
            "um dado ausente), mas este campo permite distinguir "
            "'confirmado saudável' de 'não foi possível medir'."
        )
    )
