"""
PredictIQ MCP Server — RF-19.

Servidor MCP independente, isolado do backend FastAPI (RNF-43). Expõe uma
única ferramenta stub, `search_maintenance_manual`, que futuramente será
substituída por busca real em manuais técnicos (RAG via Ollama + ChromaDB,
descrito em CLAUDE.md §4 como "search_manuals") — sem alterar a interface
pública da ferramenta.

SDK
---
`mcp` (Model Context Protocol SDK oficial, mcp>=2.0 — ver requirements.txt).
Nesta versão, a classe de alto nível para construir um servidor com
ferramentas decoradas chama-se `MCPServer` (renomeada de `FastMCP` na v1.x —
`from mcp.server.mcpserver import MCPServer`, não `mcp.server.fastmcp`).

Transporte
----------
`streamable-http` — o transporte HTTP estável e oficialmente suportado por
esta versão do SDK (`MCPServer.run(transport="streamable-http")` /
`run_streamable_http_async(host, port, ...)`). Substitui o antigo transporte
HTTP+SSE de dois endpoints da spec anterior do MCP; é o único transporte
HTTP direto (por URL) — os outros dois suportados pelo SDK (`stdio`, `sse`)
não servem para um serviço standalone acessível por URL na rede Docker.

Uso
---
    python server.py                 # lê MCP_SERVER_PORT do ambiente (default 8100)
    MCP_SERVER_PORT=9000 python server.py
"""

from __future__ import annotations

import logging
import os
from typing import Any

from mcp.server.mcpserver import MCPServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("predictiq.mcp_server")

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

# Porta configurável por variável de ambiente — necessária porque o serviço
# roda em container próprio (RNF-43) e a porta precisa ser previsível para
# MCP_SERVER_URL (ver .env.example / README).
MCP_SERVER_PORT: int = int(os.environ.get("MCP_SERVER_PORT", "8100"))
MCP_SERVER_HOST: str = "0.0.0.0"  # escuta em todas as interfaces — requisito da task

mcp = MCPServer(
    name="predictiq-mcp",
    version="0.1.0",
    instructions=(
        "Servidor MCP do PredictIQ. Expõe ferramentas de suporte à manutenção "
        "preditiva. `search_maintenance_manual` está atualmente em modo stub "
        "(RF-19) — não realiza busca real."
    ),
)


# ---------------------------------------------------------------------------
# Ferramenta — RF-19
# ---------------------------------------------------------------------------


@mcp.tool()
def search_maintenance_manual(query: str) -> dict[str, Any]:
    """
    Busca em manuais técnicos de manutenção (STUB — RF-19).

    Interface pública estável: recebe `query` (string) e devolve uma
    estrutura de resultados. A implementação atual NÃO realiza busca real —
    é um stub funcional e determinístico, preparado para ser substituído por
    RAG real (embeddings + ChromaDB, orquestrado pelo Ollama — ver CLAUDE.md
    §4) sem alterar esta assinatura nem o formato da resposta.

    Parameters
    ----------
    query:
        Texto da consulta (ex.: "vazamento de óleo no compressor").

    Returns
    -------
    dict
        Estrutura determinística e explicitamente marcada como stub:
        ``{"status": "stub", "query": ..., "results": [], "message": ...}``.
    """
    log.info("search_maintenance_manual chamada (stub) | query=%r", query)
    return {
        "status": "stub",
        "query": query,
        "results": [],
        "message": (
            "Busca real em manuais de manutenção ainda não implementada "
            "(RF-19). Esta é uma resposta stub determinística — nenhum "
            "acesso a banco de dados, arquivo ou serviço externo foi feito."
        ),
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info(
        "Iniciando PredictIQ MCP Server | host=%s port=%d transport=streamable-http",
        MCP_SERVER_HOST,
        MCP_SERVER_PORT,
    )
    mcp.run(transport="streamable-http", host=MCP_SERVER_HOST, port=MCP_SERVER_PORT)
