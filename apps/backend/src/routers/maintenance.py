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

RNF-56: a factory `get_maintenance_suggestion_service` e os imports de
`MCPSearchClient`/`OllamaClient` (infraestrutura) foram movidos para
`src/services/maintenance_suggestion_service.py` — este router só conhece
o Protocol (`MaintenanceSuggestionServiceProtocol`), nunca os clientes
concretos por trás dele.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import StreamingResponse

from src.schemas.maintenance import (
    MaintenanceSuggestionRequest,
    MaintenanceSuggestionResponse,
)
from src.services.maintenance_suggestion_service import (
    SuggestionStreamEvent,
    get_maintenance_suggestion_service,
)
from src.services.protocols import MaintenanceSuggestionServiceProtocol

log = structlog.get_logger(__name__)

# Mesmo padrão de headers de src/routers/stream.py (SSE de sensores, RF-12) —
# desabilita buffering do Nginx via header de resposta (funciona em qualquer
# location, não só /api/stream/ — ver README §4.9).
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

router: APIRouter = APIRouter(prefix="/v1/maintenance", tags=["Maintenance"])


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
    service: MaintenanceSuggestionServiceProtocol = Depends(
        get_maintenance_suggestion_service
    ),
) -> MaintenanceSuggestionResponse:
    return await service.suggest(payload)


def _format_sse_event(event: SuggestionStreamEvent) -> str:
    """Serializa um `SuggestionStreamEvent` no protocolo SSE desta task
    (RF-23 §2): ``event: <type>\\ndata: <json>\\n\\n``. Cada `data:` é sempre
    JSON válido — nunca texto solto."""
    data: dict[str, Any]
    if event.type == "searching":
        data = {}
    elif event.type == "token":
        data = {"token": event.token}
    elif event.type == "done":
        data = {
            "markdown": event.markdown,
            "references": [ref.model_dump() for ref in event.references],
        }
    else:  # "skipped" | "error"
        data = {"message": event.message}
    return f"event: {event.type}\ndata: {json.dumps(data)}\n\n"


@router.post(
    "/suggest/stream",
    summary="Streaming (SSE) do plano de manutenção — tokens do Llama 3.2 3B em tempo real",
    description=(
        "**RF-23 / RNF-47** — mesma regra de negócio de `POST /suggest` "
        "(threshold `> 0.7`, MCP, RAG), mas transmite os tokens do Ollama "
        "via Server-Sent Events conforme são gerados, em vez de esperar a "
        "resposta completa. Protocolo: eventos `searching` (MCP em andamento), "
        "`token` (incremental), `done` (markdown completo + referências), "
        "`skipped` (threshold não ultrapassado) ou `error`. Canal "
        "independente do SSE de sensores "
        "(`/api/stream/sensors`, RF-12) — não reaproveita nem altera esse "
        "contrato."
    ),
    response_description="text/event-stream — eventos token/done/skipped/error.",
)
async def suggest_maintenance_stream(
    request: Request,
    payload: MaintenanceSuggestionRequest,
    service: MaintenanceSuggestionServiceProtocol = Depends(
        get_maintenance_suggestion_service
    ),
) -> StreamingResponse:
    async def event_generator() -> AsyncGenerator[str, None]:
        log.info("maintenance_stream_opened", probability=payload.failure_probability)
        try:
            async for event in service.suggest_stream(payload):
                if await request.is_disconnected():
                    log.info("maintenance_stream_client_disconnected")
                    break
                yield _format_sse_event(event)
        except Exception:
            # Rede de segurança — qualquer exceção não prevista pelo service
            # ainda fecha o stream com um evento estruturado, nunca uma
            # conexão pendurada ou um traceback vazando pro cliente.
            log.exception("maintenance_stream_unexpected_error")
            yield _format_sse_event(
                SuggestionStreamEvent(
                    type="error",
                    message="Erro inesperado ao gerar a sugestão de manutenção.",
                )
            )
        finally:
            log.info("maintenance_stream_closed")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
