"""
Testes de `OllamaClient` (RF-22 / RF-23 / RNF-46/47, RNF-64/RNF-65).

Auditoria desta task: 0% de cobertura direta — `OllamaClient` sempre era
mockado por inteiro nos testes de `MaintenanceSuggestionService`, então seu
próprio tratamento de timeout/erro HTTP/404/JSON inválido/conteúdo vazio
nunca era exercitado. Mesmo padrão já estabelecido pelo projeto para
`TelegramNotificationAdapter` (`test_notifications.py`):
`monkeypatch.setattr(httpx.AsyncClient, "post"/"stream", ...)` com um
`httpx.Response`/stream REAL — nunca fala com a internet, mas exercita o
código de verdade (parsing, decisão condicional, mapeamento de exceção).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from src.core.exceptions import OllamaResponseError, OllamaUnavailableError
from src.services.ollama_client import OllamaClient

BASE_URL = "http://ollama-fake:11434"
MODEL = "llama3.2:3b"


def _response(status_code: int, json_body: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=json_body if json_body is not None else {},
        request=httpx.Request("POST", f"{BASE_URL}/api/chat"),
    )


# ---------------------------------------------------------------------------
# generate() — não-streaming (RF-22)
# ---------------------------------------------------------------------------


async def test_generate_success_returns_message_content(monkeypatch) -> None:
    mock_post = AsyncMock(
        return_value=_response(200, {"message": {"content": "## Plano\n1. Verificar."}})
    )
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    client = OllamaClient(base_url=BASE_URL, model=MODEL)
    result = await client.generate("system prompt", "user prompt")

    assert result == "## Plano\n1. Verificar."
    mock_post.assert_awaited_once()
    assert mock_post.await_args is not None
    args, kwargs = mock_post.await_args
    assert args[0] == f"{BASE_URL}/api/chat"
    payload = kwargs["json"]
    assert payload["model"] == MODEL
    assert payload["stream"] is False
    # System e user NUNCA concatenados numa única mensagem (RF-22 §13).
    assert payload["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
    ]


async def test_generate_timeout_raises_ollama_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx.AsyncClient, "post", AsyncMock(side_effect=httpx.TimeoutException("x"))
    )
    client = OllamaClient(base_url=BASE_URL, model=MODEL)
    with pytest.raises(OllamaUnavailableError, match="Tempo limite"):
        await client.generate("s", "u")


async def test_generate_connection_error_raises_ollama_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx.AsyncClient, "post", AsyncMock(side_effect=httpx.ConnectError("refused"))
    )
    client = OllamaClient(base_url=BASE_URL, model=MODEL)
    with pytest.raises(OllamaUnavailableError, match="conectar"):
        await client.generate("s", "u")


async def test_generate_model_not_found_raises_ollama_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx.AsyncClient, "post", AsyncMock(return_value=_response(404))
    )
    client = OllamaClient(base_url=BASE_URL, model=MODEL)
    with pytest.raises(OllamaUnavailableError, match=MODEL):
        await client.generate("s", "u")


async def test_generate_non_200_non_404_raises_ollama_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx.AsyncClient, "post", AsyncMock(return_value=_response(500))
    )
    client = OllamaClient(base_url=BASE_URL, model=MODEL)
    with pytest.raises(OllamaUnavailableError, match="erro inesperado"):
        await client.generate("s", "u")


async def test_generate_invalid_json_raises_ollama_response_error(monkeypatch) -> None:
    bad = httpx.Response(
        200, content=b"not json", request=httpx.Request("POST", f"{BASE_URL}/api/chat")
    )
    monkeypatch.setattr(httpx.AsyncClient, "post", AsyncMock(return_value=bad))
    client = OllamaClient(base_url=BASE_URL, model=MODEL)
    with pytest.raises(OllamaResponseError, match="válida"):
        await client.generate("s", "u")


@pytest.mark.parametrize(
    "message_field",
    [
        {},  # sem "message"
        {"message": {}},  # sem "content"
        {"message": {"content": ""}},  # content vazio
        {"message": {"content": "   "}},  # só espaço
        {"message": {"content": None}},  # content não-string
    ],
)
async def test_generate_empty_or_missing_content_raises_response_error(
    monkeypatch, message_field: dict[str, Any]
) -> None:
    monkeypatch.setattr(
        httpx.AsyncClient, "post", AsyncMock(return_value=_response(200, message_field))
    )
    client = OllamaClient(base_url=BASE_URL, model=MODEL)
    with pytest.raises(OllamaResponseError, match="vazio"):
        await client.generate("s", "u")


def test_base_url_trailing_slash_is_stripped() -> None:
    client = OllamaClient(base_url=f"{BASE_URL}/", model=MODEL)
    assert client._base_url == BASE_URL  # nunca "http://...//api/chat"


# ---------------------------------------------------------------------------
# generate_stream() — streaming NDJSON (RF-23)
# ---------------------------------------------------------------------------


class _FakeStreamResponse:
    """Simula o objeto devolvido por `client.stream(...)` como async context
    manager — `aiter_lines()` real (async generator), `status_code` real."""

    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self.status_code = status_code
        self._lines = lines

    async def __aenter__(self) -> "_FakeStreamResponse":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def _stream_ctx(lines: list[str], status_code: int = 200):
    """`client.stream(...)` NÃO é uma coroutine — é um método síncrono que
    devolve o context manager diretamente (mesma assinatura do httpx real)."""
    return lambda *a, **kw: _FakeStreamResponse(lines, status_code)


async def test_generate_stream_yields_each_chunk_in_order(monkeypatch) -> None:
    lines = [
        '{"message": {"content": "Ola"}, "done": false}',
        '{"message": {"content": " mundo"}, "done": false}',
        '{"message": {"content": ""}, "done": true}',
    ]
    monkeypatch.setattr(httpx.AsyncClient, "stream", _stream_ctx(lines))

    client = OllamaClient(base_url=BASE_URL, model=MODEL)
    chunks = [c async for c in client.generate_stream("s", "u")]

    assert chunks == ["Ola", " mundo"]  # chunk final vazio nunca é repassado


async def test_generate_stream_skips_malformed_json_line_without_aborting(
    monkeypatch,
) -> None:
    lines = [
        '{"message": {"content": "valido"}, "done": false}',
        "isso nao e json",
        '{"message": {"content": ""}, "done": true}',
    ]
    monkeypatch.setattr(httpx.AsyncClient, "stream", _stream_ctx(lines))

    client = OllamaClient(base_url=BASE_URL, model=MODEL)
    chunks = [c async for c in client.generate_stream("s", "u")]

    assert chunks == ["valido"]


async def test_generate_stream_ignores_blank_lines(monkeypatch) -> None:
    lines = ["", "   ", '{"message": {"content": "x"}, "done": true}']
    monkeypatch.setattr(httpx.AsyncClient, "stream", _stream_ctx(lines))

    client = OllamaClient(base_url=BASE_URL, model=MODEL)
    chunks = [c async for c in client.generate_stream("s", "u")]

    assert chunks == ["x"]


async def test_generate_stream_stops_after_done_true_even_if_more_lines_follow(
    monkeypatch,
) -> None:
    lines = [
        '{"message": {"content": "primeiro"}, "done": true}',
        '{"message": {"content": "NUNCA deveria aparecer"}, "done": false}',
    ]
    monkeypatch.setattr(httpx.AsyncClient, "stream", _stream_ctx(lines))

    client = OllamaClient(base_url=BASE_URL, model=MODEL)
    chunks = [c async for c in client.generate_stream("s", "u")]

    assert chunks == ["primeiro"]


async def test_generate_stream_empty_stream_raises_response_error(monkeypatch) -> None:
    lines = ['{"message": {"content": ""}, "done": true}']
    monkeypatch.setattr(httpx.AsyncClient, "stream", _stream_ctx(lines))

    client = OllamaClient(base_url=BASE_URL, model=MODEL)
    with pytest.raises(OllamaResponseError, match="vazio"):
        async for _ in client.generate_stream("s", "u"):
            pass


async def test_generate_stream_model_not_found_raises_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(httpx.AsyncClient, "stream", _stream_ctx([], status_code=404))

    client = OllamaClient(base_url=BASE_URL, model=MODEL)
    with pytest.raises(OllamaUnavailableError, match=MODEL):
        async for _ in client.generate_stream("s", "u"):
            pass


async def test_generate_stream_bad_status_raises_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(httpx.AsyncClient, "stream", _stream_ctx([], status_code=503))

    client = OllamaClient(base_url=BASE_URL, model=MODEL)
    with pytest.raises(OllamaUnavailableError, match="erro inesperado"):
        async for _ in client.generate_stream("s", "u"):
            pass


async def test_generate_stream_timeout_raises_unavailable(monkeypatch) -> None:
    def _raise(*a, **kw):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(httpx.AsyncClient, "stream", _raise)

    client = OllamaClient(base_url=BASE_URL, model=MODEL)
    with pytest.raises(OllamaUnavailableError, match="Tempo limite"):
        async for _ in client.generate_stream("s", "u"):
            pass


async def test_generate_stream_connection_error_raises_unavailable(monkeypatch) -> None:
    def _raise(*a, **kw):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx.AsyncClient, "stream", _raise)

    client = OllamaClient(base_url=BASE_URL, model=MODEL)
    with pytest.raises(OllamaUnavailableError, match="conectar"):
        async for _ in client.generate_stream("s", "u"):
            pass
