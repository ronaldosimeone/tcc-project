"""
Cliente MCP — RF-22 / RNF-46.

Wrapper fino sobre o SDK oficial `mcp` (mesmo pacote usado pelo servidor em
apps/mcp-server, `mcp==2.2.0`), falando o protocolo REAL confirmado por
auditoria: transporte `streamable-http`, endpoint `POST /mcp`, handshake
`initialize` -> `call_tool`. Não reimplementa a busca semântica — apenas
chama a tool `search_maintenance_manual` já exposta pelo mcp-server (RF-21).

Este backend NUNCA fala com o ChromaDB diretamente nem duplica o
SemanticSearchService — toda a lógica de busca continua exclusivamente em
apps/mcp-server (RF-20/RF-21).
"""

from __future__ import annotations

from typing import Any

import structlog
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from src.core.exceptions import MCPUnavailableError

log = structlog.get_logger(__name__)

# Nome real da tool, confirmado em apps/mcp-server/server.py (RF-19/RF-21).
# CLAUDE.md/AGENTS.md mencionam "search_manuals" como nome conceitual — o
# nome registrado no servidor real é este; não renomear sem necessidade.
SEARCH_TOOL_NAME = "search_maintenance_manual"


class MCPSearchClient:
    """Cliente para a ferramenta MCP `search_maintenance_manual`."""

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        # base_url = MCP_SERVER_URL (ex.: http://mcp-server:8100) — o SDK
        # espera a URL completa do endpoint MCP (.../mcp).
        self._url = base_url.rstrip("/") + "/mcp"
        self._timeout = timeout

    async def search_maintenance_manual(self, query: str) -> dict[str, Any]:
        """
        Abre uma sessão MCP, chama a tool e devolve `structured_content`
        (``{"query": ..., "results": [...]}`` — ver apps/mcp-server/server.py).

        Levanta `MCPUnavailableError` (503) para qualquer falha de rede,
        timeout, ou resposta em formato inesperado — nunca propaga o
        traceback/URL interna ao cliente HTTP (RF-22 §13).
        """
        try:
            async with streamable_http_client(self._url) as (read_stream, write_stream):
                async with ClientSession(
                    read_stream, write_stream, read_timeout_seconds=self._timeout
                ) as session:
                    await session.initialize()
                    result = await session.call_tool(SEARCH_TOOL_NAME, {"query": query})
        except MCPUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 — fronteira de serviço externo
            # anyio agrupa falhas de conexão/timeout em ExceptionGroup; captura
            # ampla e intencional aqui, convertida num erro de domínio único.
            log.warning("mcp_client_connection_error", url=self._url, error=str(exc))
            raise MCPUnavailableError(
                "Não foi possível conectar ao servidor MCP de busca em manuais."
            ) from exc

        if result.is_error:
            log.warning(
                "mcp_tool_returned_error", tool=SEARCH_TOOL_NAME, content=result.content
            )
            raise MCPUnavailableError(
                "A ferramenta de busca em manuais retornou um erro."
            )

        structured = result.structured_content
        if not isinstance(structured, dict) or "results" not in structured:
            log.warning("mcp_tool_unexpected_shape", structured=structured)
            raise MCPUnavailableError("Resposta do servidor MCP em formato inesperado.")

        return structured
