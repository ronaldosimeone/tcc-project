"""
PredictIQ MCP Server — RF-19 / RF-21.

Servidor MCP independente, isolado do backend FastAPI (RNF-43). Expõe uma
única ferramenta, `search_maintenance_manual`, que faz busca semântica real
sobre os manuais indexados por `index_manuals.py` (RF-20) via
`SemanticSearchService` (RF-21) — ver `semantic_search.py`. Interface da
ferramenta (nome, `query: string`) preservada desde o stub original (RF-19).

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

from index_manuals import (
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    get_collection,
    load_embedding_model,
)
from semantic_search import SemanticSearchService

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
    version="0.2.0",
    instructions=(
        "Servidor MCP do PredictIQ. Expõe ferramentas de suporte à manutenção "
        "preditiva. `search_maintenance_manual` faz busca semântica real "
        "sobre os manuais indexados (RF-20/RF-21) e devolve, no máximo, os 5 "
        "trechos mais relevantes com score de similaridade acima de 0.6."
    ),
)


# ---------------------------------------------------------------------------
# Singleton de processo — RNF-45. Modelo de embeddings e collection do
# ChromaDB carregados UMA vez (no primeiro uso da ferramenta), reaproveitados
# em todas as chamadas seguintes. Nunca recarregar por query.
# ---------------------------------------------------------------------------

_service: SemanticSearchService | None = None


def _get_service() -> SemanticSearchService:
    global _service
    if _service is None:
        log.info("Inicializando SemanticSearchService (primeira chamada da tool)")
        collection = get_collection(CHROMA_DB_PATH)
        model = load_embedding_model(EMBEDDING_MODEL)
        _service = SemanticSearchService(collection=collection, model=model)
    return _service


# ---------------------------------------------------------------------------
# Ferramenta — RF-19 / RF-21
# ---------------------------------------------------------------------------


@mcp.tool()
def search_maintenance_manual(query: str) -> dict[str, Any]:
    """
    Busca semântica em manuais técnicos de manutenção (RF-21).

    Interface pública estável desde o stub original (RF-19): recebe `query`
    (string) e devolve uma estrutura de resultados. Delega a busca real para
    `SemanticSearchService` (semantic_search.py), que consulta a collection
    ChromaDB `maintenance_manuals` preparada por `index_manuals.py` (RF-20).

    Parameters
    ----------
    query:
        Texto da consulta (ex.: "vazamento de óleo no compressor").

    Returns
    -------
    dict
        ``{"query": ..., "results": [{"text", "score", "metadata"}, ...]}``.
        No máximo 5 itens, cada um com score de similaridade > 0.6. Uma
        consulta sem nenhum trecho relevante o suficiente é um resultado
        normal — devolve ``"results": []``, sem lançar exceção.
    """
    log.info("search_maintenance_manual chamada | query=%r", query)
    service = _get_service()
    results = service.search(query)
    return {
        "query": query,
        "results": [
            {
                "text": result.text,
                "score": round(result.score, 4),
                "metadata": result.metadata,
            }
            for result in results
        ],
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
