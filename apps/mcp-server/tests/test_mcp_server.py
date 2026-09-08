"""
Testes do MCP Server standalone (RF-19 / RF-21).

Cobertura:
  1. Inicialização — o módulo importa e o servidor é construído sem erro.
  2. Registro — `search_maintenance_manual` está em `mcp.list_tools()`.
  3. Schema — aceita `query: string` (obrigatório); rejeita entrada inválida
     (campo ausente) via validação pydantic do próprio SDK.
  4. Execução/integração — a tool efetivamente delega para
     `SemanticSearchService` (RF-21) e devolve a estrutura
     `{"query", "results": [{"text","score","metadata"}]}`. Os testes de
     "plumbing" (schema, determinismo, erro) usam um `_FakeService` simples;
     um teste dedicado usa um `SemanticSearchService` REAL sobre uma
     collection ChromaDB REAL (temporária) — só o `SentenceTransformer` é
     fake — pra provar a integração de verdade, não só que a tool devolve o
     que um mock qualquer mandar.
  5. Transporte — smoke test do app ASGI real (`streamable_http_app()`) via
     cliente HTTP em processo (sem bind de porta/rede — não depende de
     serviços externos nem de internet).

Nenhum teste baixa modelo real nem toca `data/chroma` real — ver
`_reset_service_singleton` (autouse) e `_FixedVectorModel`/`_FakeService`
abaixo. Reaproveita a MESMA instância `mcp`/ferramenta de `server.py` — não
duplica a lógica de busca aqui (isso é coberto em `test_semantic_search.py`).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server  # noqa: E402
from index_manuals import get_collection  # noqa: E402
from semantic_search import SearchResult, SemanticSearchService  # noqa: E402
from server import MCP_SERVER_HOST, mcp, search_maintenance_manual  # noqa: E402

TOOL_NAME = "search_maintenance_manual"


def _run(coro):
    """Executa uma coroutine em um novo event loop — evita depender do
    plugin pytest-asyncio (dependência a menos, ver requirements.txt)."""
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _reset_service_singleton():
    """Todo teste começa (e termina) sem singleton — nenhum teste herda o
    `_service` que outro tenha injetado, e nenhum acidentalmente carrega o
    modelo/collection reais (rede/disco pesado) por causa de estado
    vazado."""
    server._service = None
    yield
    server._service = None


class _FakeService:
    """Dublê simples pra testar o "plumbing" da tool (schema de saída,
    determinismo, propagação de erro) sem depender do ChromaDB/modelo
    reais. A integração de verdade com `SemanticSearchService` é coberta
    à parte, em `test_search_maintenance_manual_uses_real_service_end_to_end`."""

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.queries: list[str] = []

    def search(self, query: str) -> list[SearchResult]:
        self.queries.append(query)
        return self._results


class _FixedVectorModel:
    """Mesmo papel do `_FixedVectorModel` de `test_semantic_search.py`:
    ignora o texto, sempre devolve `vector` — sem rede, sem download."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector

    def encode(
        self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
    ):
        return np.array([self.vector for _ in texts])


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
# 4. Execução / integração — RF-21
# ---------------------------------------------------------------------------


def test_direct_call_returns_structured_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chama a função Python diretamente (sem passar pelo transporte MCP).
    `_service` é um dublê — este teste cobre o formato da resposta da tool,
    não a busca real (ver teste end-to-end abaixo)."""
    fake = _FakeService(
        [
            SearchResult(
                text="trecho relevante do manual",
                score=0.87,
                metadata={"file_name": "manual.pdf", "page": 2},
            )
        ]
    )
    monkeypatch.setattr(server, "_service", fake)

    result = search_maintenance_manual("vazamento de óleo")

    assert result["query"] == "vazamento de óleo"
    assert result["results"] == [
        {
            "text": "trecho relevante do manual",
            "score": 0.87,
            "metadata": {"file_name": "manual.pdf", "page": 2},
        }
    ]
    assert fake.queries == [
        "vazamento de óleo"
    ]  # a tool repassa a query recebida, não inventa outra


def test_direct_call_with_no_relevant_results_returns_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nenhum chunk acima do threshold é um resultado normal — sem exceção."""
    monkeypatch.setattr(server, "_service", _FakeService([]))

    result = search_maintenance_manual("consulta sem nada relacionado")

    assert result == {"query": "consulta sem nada relacionado", "results": []}


def test_direct_call_is_deterministic_across_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server,
        "_service",
        _FakeService([SearchResult(text="x", score=0.9, metadata={})]),
    )

    r1 = search_maintenance_manual("mesma consulta")
    r2 = search_maintenance_manual("mesma consulta")
    assert r1 == r2


def test_call_tool_via_mcp_returns_structured_results_and_no_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executa através do mecanismo do SDK (`call_tool`), não da função
    Python crua — cobre a camada de despacho/validação do MCPServer."""
    fake = _FakeService(
        [
            SearchResult(
                text="correia gasta",
                score=0.72,
                metadata={"file_name": "compressor.pdf", "page": 5},
            )
        ]
    )
    monkeypatch.setattr(server, "_service", fake)

    result = _run(mcp.call_tool(TOOL_NAME, {"query": "correia do compressor"}))

    assert result.is_error is False
    assert result.structured_content is not None
    assert result.structured_content["query"] == "correia do compressor"
    assert result.structured_content["results"][0]["text"] == "correia gasta"
    assert result.structured_content["results"][0]["score"] == 0.72
    assert (
        result.structured_content["results"][0]["metadata"]["file_name"]
        == "compressor.pdf"
    )


def test_search_maintenance_manual_uses_real_service_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Prova a integração de verdade (RF-21): NÃO usa `_FakeService` — monta
    um `SemanticSearchService` real sobre uma collection ChromaDB real
    (temporária, populada com um chunk conhecido). Só o `SentenceTransformer`
    é substituído (sem rede/download). Bloqueia `socket.connect` também, pra
    provar que o caminho de busca em si não abre conexão nenhuma uma vez que
    modelo/collection já estão disponíveis (o download do modelo é uma
    preocupação de infra à parte, não do algoritmo de busca)."""
    import socket

    collection = get_collection(tmp_path / "chroma")
    collection.add(
        ids=["c1"],
        embeddings=[[1.0, 0.0]],
        documents=["Trocar oleo do compressor a cada 500 horas."],
        metadatas=[
            {
                "source": "manual.pdf",
                "file_name": "manual.pdf",
                "file_hash": "a" * 64,
                "page": 1,
                "chunk_index": 0,
                "embedding_model": "fake-model-for-tests",
            }
        ],
    )
    real_service = SemanticSearchService(
        collection=collection, model=_FixedVectorModel([1.0, 0.0])
    )
    monkeypatch.setattr(server, "_service", real_service)

    def _blocked_connect(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("busca não deveria abrir socket algum")

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)

    result = search_maintenance_manual("troca de oleo do compressor")

    assert result["query"] == "troca de oleo do compressor"
    assert len(result["results"]) == 1
    hit = result["results"][0]
    assert hit["text"] == "Trocar oleo do compressor a cada 500 horas."
    assert hit["score"] == pytest.approx(1.0)  # mesmo vetor -> cosine similarity máxima
    assert hit["metadata"]["file_name"] == "manual.pdf"
    assert hit["metadata"]["page"] == 1


def test_service_singleton_loads_embedding_model_only_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """RNF-45: o modelo de embeddings é carregado uma vez por processo, não
    uma vez por chamada da tool — e é exatamente `EMBEDDING_MODEL` (RF-21:
    'não hardcodear outro modelo')."""
    load_calls: list[str] = []

    def _fake_load_embedding_model(model_name: str) -> _FixedVectorModel:
        load_calls.append(model_name)
        return _FixedVectorModel([1.0, 0.0])

    def _fake_get_collection(_path: Path):
        return get_collection(tmp_path / "chroma")

    monkeypatch.setattr(server, "load_embedding_model", _fake_load_embedding_model)
    monkeypatch.setattr(server, "get_collection", _fake_get_collection)

    search_maintenance_manual("query 1")
    search_maintenance_manual("query 2")
    search_maintenance_manual("query 3")

    assert load_calls == [
        server.EMBEDDING_MODEL
    ]  # uma única chamada, com o modelo configurado


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
