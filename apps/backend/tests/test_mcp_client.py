"""
Testes de `MCPSearchClient` (RF-22 / RNF-46, RNF-64/RNF-65).

Auditoria desta task: 31% de cobertura — `MCPSearchClient` é sempre mockado
por inteiro nos testes de `MaintenanceSuggestionService` (a fronteira de
integração deliberada do projeto, ver `test_full_pipeline.py`), então sua
PRÓPRIA lógica (mapeamento de exceção, validação de `is_error`/
`structured_content`) nunca era exercitada diretamente.

Aqui o SDK `mcp` (`streamable_http_client`/`ClientSession`, dependência
externa já testada pelo próprio pacote) é o único ponto substituído — dublês
mínimos que só replicam a forma do protocolo assíncrono real (dois context
managers aninhados) — para exercitar de verdade o código PRÓPRIO deste
módulo: construção da URL, `try/except` de conexão, checagem de
`result.is_error`, validação de `structured_content`.

A integração REAL (backend -> mcp-server de verdade) já foi validada
manualmente contra o container real nas tasks RNF-62 (ver
RNF-62-mcp-remediacao.md) — não repetida aqui como teste automatizado
porque exigiria a stack `docker compose` de pé, fora do escopo de uma
suíte `pytest` unitária/de CI.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.core.exceptions import MCPUnavailableError
from src.services.mcp_client import SEARCH_TOOL_NAME, MCPSearchClient

MCP_URL = "http://mcp-server-fake:8100"


class _FakeStream:
    """Placeholder para (read_stream, write_stream) — nunca lido/escrito de
    verdade nestes testes; só precisa existir para o unpacking funcionar."""


class _FakeStreamableHttpClient:
    """Substitui `mcp.client.streamable_http.streamable_http_client` — async
    context manager que produz o par (read_stream, write_stream)."""

    def __init__(self, url: str) -> None:
        self.url = url

    async def __aenter__(self) -> tuple[_FakeStream, _FakeStream]:
        return (_FakeStream(), _FakeStream())

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _FakeResult:
    def __init__(
        self, *, is_error: bool, structured_content: Any, content: Any = None
    ) -> None:
        self.is_error = is_error
        self.structured_content = structured_content
        self.content = content


class _FakeClientSession:
    """Substitui `mcp.client.session.ClientSession` — async context manager
    com `.initialize()`/`.call_tool()`. `call_tool_result`/`raise_on_enter`
    controlam o comportamento simulado por teste."""

    def __init__(
        self,
        *args: Any,
        call_tool_result: _FakeResult | None = None,
        raise_on_call: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        self._result = call_tool_result
        self._raise_on_call = raise_on_call
        self.initialize = AsyncMock()

    async def __aenter__(self) -> "_FakeClientSession":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> _FakeResult:
        self.last_call = (name, arguments)
        if self._raise_on_call is not None:
            raise self._raise_on_call
        assert self._result is not None
        return self._result


def _patch_mcp_sdk(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: _FakeResult | None = None,
    raise_on_call: Exception | None = None,
    raise_on_transport: Exception | None = None,
) -> None:
    def fake_streamable_http_client(url: str):
        if raise_on_transport is not None:
            raise raise_on_transport
        return _FakeStreamableHttpClient(url)

    def fake_session(*args: Any, **kwargs: Any) -> _FakeClientSession:
        return _FakeClientSession(
            *args, call_tool_result=result, raise_on_call=raise_on_call, **kwargs
        )

    monkeypatch.setattr(
        "src.services.mcp_client.streamable_http_client", fake_streamable_http_client
    )
    monkeypatch.setattr("src.services.mcp_client.ClientSession", fake_session)


# ---------------------------------------------------------------------------
# Caminho feliz
# ---------------------------------------------------------------------------


async def test_search_success_returns_structured_content(monkeypatch) -> None:
    expected = {"query": "vazamento", "results": [{"text": "x", "score": 0.8}]}
    _patch_mcp_sdk(
        monkeypatch, result=_FakeResult(is_error=False, structured_content=expected)
    )

    client = MCPSearchClient(base_url=MCP_URL)
    out = await client.search_maintenance_manual("vazamento")

    assert out == expected


def test_url_construction_appends_mcp_endpoint_and_strips_trailing_slash() -> None:
    assert MCPSearchClient(base_url=f"{MCP_URL}/")._url == f"{MCP_URL}/mcp"
    assert MCPSearchClient(base_url=MCP_URL)._url == f"{MCP_URL}/mcp"


async def test_search_calls_the_documented_tool_name_with_the_query(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    class _Recording(_FakeClientSession):
        async def call_tool(self, name, arguments):  # type: ignore[override]
            captured["name"] = name
            captured["arguments"] = arguments
            return _FakeResult(
                is_error=False, structured_content={"query": "q", "results": []}
            )

    def fake_session(*a, **kw):
        return _Recording(*a, **kw)

    monkeypatch.setattr(
        "src.services.mcp_client.streamable_http_client",
        lambda url: _FakeStreamableHttpClient(url),
    )
    monkeypatch.setattr("src.services.mcp_client.ClientSession", fake_session)

    client = MCPSearchClient(base_url=MCP_URL)
    await client.search_maintenance_manual("vazamento de oleo")

    assert captured["name"] == SEARCH_TOOL_NAME == "search_maintenance_manual"
    assert captured["arguments"] == {"query": "vazamento de oleo"}


# ---------------------------------------------------------------------------
# Erros mapeados para MCPUnavailableError (RF-22 §13 — nunca vaza a URL/traceback)
# ---------------------------------------------------------------------------


async def test_transport_connection_failure_raises_mcp_unavailable(monkeypatch) -> None:
    _patch_mcp_sdk(monkeypatch, raise_on_transport=ConnectionRefusedError("refused"))
    client = MCPSearchClient(base_url=MCP_URL)
    with pytest.raises(MCPUnavailableError) as exc_info:
        await client.search_maintenance_manual("q")
    # Nunca propaga a URL interna do MCP na mensagem de erro (RF-22 §13).
    assert MCP_URL not in str(exc_info.value)


async def test_generic_exception_during_call_is_wrapped_as_mcp_unavailable(
    monkeypatch,
) -> None:
    _patch_mcp_sdk(monkeypatch, raise_on_call=RuntimeError("anyio ExceptionGroup boom"))
    client = MCPSearchClient(base_url=MCP_URL)
    with pytest.raises(MCPUnavailableError, match="Não foi possível conectar"):
        await client.search_maintenance_manual("q")


async def test_mcp_unavailable_error_from_inner_call_is_not_double_wrapped(
    monkeypatch,
) -> None:
    """Se o próprio SDK levantasse um `MCPUnavailableError` (não acontece
    hoje, mas o `except MCPUnavailableError: raise` existe para isso), ele
    deve propagar intacto — nunca reembrulhado numa mensagem genérica."""
    original = MCPUnavailableError("mensagem original específica")
    _patch_mcp_sdk(monkeypatch, raise_on_call=original)
    client = MCPSearchClient(base_url=MCP_URL)
    with pytest.raises(MCPUnavailableError, match="mensagem original específica"):
        await client.search_maintenance_manual("q")


async def test_tool_is_error_true_raises_mcp_unavailable(monkeypatch) -> None:
    _patch_mcp_sdk(
        monkeypatch,
        result=_FakeResult(is_error=True, structured_content=None, content="deu erro"),
    )
    client = MCPSearchClient(base_url=MCP_URL)
    with pytest.raises(MCPUnavailableError, match="erro"):
        await client.search_maintenance_manual("q")


@pytest.mark.parametrize(
    "structured_content",
    [
        None,
        "uma string, nao um dict",
        {"query": "q"},  # dict, mas sem a chave "results"
        [],  # lista, nao dict
    ],
)
async def test_malformed_structured_content_raises_mcp_unavailable(
    monkeypatch, structured_content: Any
) -> None:
    _patch_mcp_sdk(
        monkeypatch,
        result=_FakeResult(is_error=False, structured_content=structured_content),
    )
    client = MCPSearchClient(base_url=MCP_URL)
    with pytest.raises(MCPUnavailableError, match="formato inesperado"):
        await client.search_maintenance_manual("q")


async def test_structured_content_with_results_key_and_extra_keys_is_accepted(
    monkeypatch,
) -> None:
    """O contrato exige só que "results" esteja presente — chaves extras
    (ex.: metadados futuros do mcp-server) não devem quebrar o parsing."""
    payload = {"query": "q", "results": [], "server_version": "0.2.0"}
    _patch_mcp_sdk(
        monkeypatch, result=_FakeResult(is_error=False, structured_content=payload)
    )
    client = MCPSearchClient(base_url=MCP_URL)
    out = await client.search_maintenance_manual("q")
    assert out == payload
