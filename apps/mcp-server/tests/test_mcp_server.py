"""
Testes do MCP Server standalone (RF-19).

Cobertura:
  1. Inicialização — o módulo importa e o servidor é construído sem erro.
  2. Registro — `search_maintenance_manual` está em `mcp.list_tools()`.
  3. Schema — aceita `query: string` (obrigatório); rejeita entrada inválida
     (campo ausente) via validação pydantic do próprio SDK.
  4. Execução — resposta existe, é determinística, identifica-se como stub,
     e não faz nenhum acesso externo (não há nada para acessar — a
     implementação é 100% local/síncrona).
  5. Transporte — smoke test do app ASGI real (`streamable_http_app()`) via
     cliente HTTP em processo (sem bind de porta/rede — não depende de
     serviços externos nem de internet).

Reaproveita a MESMA instância `mcp`/ferramenta de `server.py` — não duplica
a lógica do stub aqui.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import MCP_SERVER_HOST, mcp, search_maintenance_manual  # noqa: E402

TOOL_NAME = "search_maintenance_manual"


def _run(coro):
    """Executa uma coroutine em um novo event loop — evita depender do
    plugin pytest-asyncio (dependência a menos, ver requirements.txt)."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 1. Inicialização
# ---------------------------------------------------------------------------


def test_server_module_imports_without_error() -> None:
    """O módulo server.py já foi importado no topo deste arquivo — se algo
    tivesse falhado na construção do MCPServer ou no registro da
    ferramenta, a coleta deste arquivo de teste já teria falhado antes."""
    assert mcp is not None
    assert mcp.name == "predictiq-mcp"


def test_server_host_binds_all_interfaces() -> None:
    """Requisito explícito da task: o servidor deve escutar em 0.0.0.0."""
    assert MCP_SERVER_HOST == "0.0.0.0"


# ---------------------------------------------------------------------------
# 2. Registro da ferramenta
# ---------------------------------------------------------------------------


def test_search_maintenance_manual_is_registered() -> None:
    tools = _run(mcp.list_tools())
    names = [t.name for t in tools]
    assert TOOL_NAME in names
    assert len(tools) == 1, "Task pede exatamente UMA ferramenta inicial"


# ---------------------------------------------------------------------------
# 3. Schema
# ---------------------------------------------------------------------------


def test_tool_schema_requires_query_string() -> None:
    tools = _run(mcp.list_tools())
    tool = next(t for t in tools if t.name == TOOL_NAME)
    schema = tool.input_schema

    assert schema["type"] == "object"
    assert "query" in schema["properties"]
    assert schema["properties"]["query"]["type"] == "string"
    assert schema["required"] == ["query"]


def test_tool_rejects_missing_query() -> None:
    from mcp.server.mcpserver.exceptions import ToolError

    with pytest.raises(ToolError):
        _run(mcp.call_tool(TOOL_NAME, {}))


def test_tool_rejects_wrong_type_for_query() -> None:
    from mcp.server.mcpserver.exceptions import ToolError

    with pytest.raises(ToolError):
        _run(mcp.call_tool(TOOL_NAME, {"query": 12345}))


# ---------------------------------------------------------------------------
# 4. Execução
# ---------------------------------------------------------------------------


def test_direct_call_returns_deterministic_stub_response() -> None:
    """Chama a função Python diretamente (sem passar pelo transporte MCP)."""
    result = search_maintenance_manual("vazamento de óleo")

    assert result["status"] == "stub"
    assert result["query"] == "vazamento de óleo"
    assert result["results"] == []
    assert "message" in result and isinstance(result["message"], str)


def test_direct_call_is_deterministic_across_calls() -> None:
    r1 = search_maintenance_manual("mesma consulta")
    r2 = search_maintenance_manual("mesma consulta")
    assert r1 == r2


def test_call_tool_via_mcp_returns_stub_and_no_error() -> None:
    """Executa através do mecanismo do SDK (`call_tool`), não da função
    Python crua — cobre a camada de despacho/validação do MCPServer."""
    result = _run(mcp.call_tool(TOOL_NAME, {"query": "correia do compressor"}))

    assert result.is_error is False
    assert result.structured_content is not None
    assert result.structured_content["status"] == "stub"
    assert result.structured_content["query"] == "correia do compressor"
    assert result.structured_content["results"] == []
    # Explicitamente identifica o comportamento como stub (não é busca real).
    assert "stub" in result.structured_content["message"].lower() or (
        "não implementada" in result.structured_content["message"].lower()
    )


def test_stub_never_touches_network_or_filesystem(monkeypatch) -> None:
    """Garante que a execução do stub não tenta nenhum I/O externo — se
    algum código futuro acidentalmente adicionar uma chamada de rede/disco
    ao stub, este teste falha alto e claro."""
    import socket

    def _blocked(*args, **kwargs):
        raise AssertionError("search_maintenance_manual não deve abrir sockets")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr("builtins.open", _blocked)

    result = search_maintenance_manual("qualquer consulta")
    assert result["status"] == "stub"


# ---------------------------------------------------------------------------
# 5. Transporte — smoke test do app ASGI real (streamable-http)
# ---------------------------------------------------------------------------


def test_streamable_http_app_responds_to_initialize() -> None:
    """
    Smoke test do transporte escolhido (streamable-http) — usa
    `starlette.testclient.TestClient` (mesma lib usada pelos testes do
    backend), que abre e fecha o lifespan ASGI do app automaticamente. Sem
    porta de rede real, sem dependência de serviços externos/internet.

    `TransportSecuritySettings(allowed_hosts=["testserver"])` é passado
    explicitamente só para este app de teste — protege contra DNS rebinding
    por padrão (`allowed_hosts=[]` rejeita tudo); em produção `server.py`
    NÃO passa esse override, então o comportamento real do serviço não é
    afetado por esta configuração de teste.
    """
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.testclient import TestClient

    app = mcp.streamable_http_app(
        transport_security=TransportSecuritySettings(allowed_hosts=["testserver"])
    )

    with TestClient(app) as client:
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "pytest", "version": "0"},
                },
            },
            headers={"Accept": "application/json, text/event-stream"},
        )

    assert response.status_code == 200
    assert "predictiq-mcp" in response.text
    assert "protocolVersion" in response.text
