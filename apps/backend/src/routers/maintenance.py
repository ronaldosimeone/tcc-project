"""
Maintenance router — RF-22 / RNF-46.

Responsibilities (Clean Arch §3):
- I/O only: receive request, inject MaintenanceSuggestionService, return response.
- Zero business logic — threshold/MCP/Ollama orchestration lives in
  MaintenanceSuggestionService (src/services/maintenance_suggestion_service.py).

Path: `POST /v1/maintenance/suggest` — via Nginx (`location /api/`) e
`root_path="/api"` do FastAPI, alcançável em produção como
`POST /api/v1/maintenance/suggest` (path exato pedido pela especificação).

Nota de convenção: dos routers existentes, só `predictions.py` usa prefixo
`/v1` (`APIRouter(prefix="/v1", ...)`); `predict.py`/`models.py`/
`simulator.py` não versionam. A convenção do projeto é inconsistente, não
inexistente — como a especificação desta task pede o path exato com `/v1/`,
seguimos o precedente de `predictions.py` aqui, sem alterar os outros
routers. Ver README "RF-22 — Divergências".
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from src.core.config import settings
from src.schemas.maintenance import (
    MaintenanceSuggestionRequest,
    MaintenanceSuggestionResponse,
)
from src.services.maintenance_suggestion_service import MaintenanceSuggestionService
from src.services.mcp_client import MCPSearchClient
from src.services.ollama_client import OllamaClient

router: APIRouter = APIRouter(prefix="/v1/maintenance", tags=["Maintenance"])


def get_maintenance_suggestion_service() -> MaintenanceSuggestionService:
    """FastAPI Depends factory — instancia clientes leves (sem estado de
    conexão persistente) a cada requisição, igual ao padrão de
    `get_alert_service`. Nenhum singleton de processo necessário aqui: o
    custo real (modelo de embeddings, ChromaDB) já é amortizado do lado do
    mcp-server (RF-21, `server._get_service`)."""
    mcp_client = MCPSearchClient(
        base_url=settings.mcp_server_url,
        timeout=settings.mcp_client_timeout_seconds,
    )
    ollama_client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout=settings.ollama_client_timeout_seconds,
    )
    return MaintenanceSuggestionService(
        mcp_client=mcp_client,
        ollama_client=ollama_client,
        model=settings.ollama_model,
    )


@router.post(
    "/suggest",
    response_model=MaintenanceSuggestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Automatic maintenance plan suggestion (RAG via MCP + Ollama)",
    description=(
        "**RF-22** — Quando `failure_probability > 0.7` (estrito), consulta os "
        "manuais técnicos via MCP (`search_maintenance_manual`, RF-21) e usa o "
        "Llama 3.2 3B local (Ollama, RNF-46) para gerar um plano de manutenção "
        "em Markdown fundamentado exclusivamente nos trechos recuperados. "
        "Quando a probabilidade não excede o limiar, `triggered=False` e nem "
        "MCP nem Ollama são chamados."
    ),
    responses={
        503: {"description": "MCP ou Ollama indisponível — ver `detail` na resposta."},
        502: {
            "description": "Ollama respondeu, mas com conteúdo inválido (não-Markdown)."
        },
    },
)
async def suggest_maintenance(
    payload: MaintenanceSuggestionRequest,
    service: MaintenanceSuggestionService = Depends(get_maintenance_suggestion_service),
) -> MaintenanceSuggestionResponse:
    return await service.suggest(payload)
