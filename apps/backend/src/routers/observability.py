"""
Observability router — RNF-77.

Responsibilities (Clean Arch §3):
- I/O only: nenhuma lógica de negócio aqui — toda a query PromQL e a
  classificação NORMAL/WARNING/CRITICAL vivem em `observability_service.py`.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.schemas.observability import ErrorRateResponse
from src.services.observability_service import get_error_rate_status

router: APIRouter = APIRouter(prefix="/observability", tags=["Observability"])


@router.get(
    "/error-rate",
    response_model=ErrorRateResponse,
    summary="Taxa de erro HTTP 5xx atual (RNF-77)",
    description=(
        "Consulta o Prometheus interno e devolve o veredito NORMAL/WARNING/"
        "CRITICAL usado pelo alerta visual do Dashboard. Nunca expõe o "
        "Prometheus diretamente ao cliente — sempre 200, mesmo se o "
        "Prometheus estiver indisponível (ver `prometheus_reachable`)."
    ),
)
async def get_error_rate() -> ErrorRateResponse:
    return await get_error_rate_status()
