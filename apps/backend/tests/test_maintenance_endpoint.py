"""
Testes HTTP de `POST /v1/maintenance/suggest` (RF-22 / RNF-46).

Nível de integração do router: `MaintenanceSuggestionService` é substituído
via `dependency_overrides` (mesmo padrão de `test_predict_endpoint.py`) —
não testa MCP/Ollama reais aqui (isso é `test_maintenance_suggestion.py`
para o service, e a validação real documentada no README para o E2E).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.core.exceptions import MCPUnavailableError
from src.main import create_app
from src.routers.maintenance import get_maintenance_suggestion_service
from src.schemas.maintenance import ManualReference, MaintenanceSuggestionResponse
from src.services.maintenance_suggestion_service import SuggestionStreamEvent

_VALID_PAYLOAD: dict = {
    "failure_probability": 0.85,
    "equipment_name": "Bomba centrifuga",
    "symptom_description": "vazamento",
}


def _app_with_service(fake_service: object):
    application = create_app()
    application.dependency_overrides[get_maintenance_suggestion_service] = (
        lambda: fake_service
    )
    return application


async def _client_for(fake_service: object) -> AsyncClient:
    transport = ASGITransport(app=_app_with_service(fake_service))
    return AsyncClient(transport=transport, base_url="http://testserver")


# ---------------------------------------------------------------------------
# Contrato HTTP — não disparado
# ---------------------------------------------------------------------------


async def test_endpoint_returns_200_with_triggered_false_below_threshold() -> None:
    fake_service = AsyncMock()
    fake_service.suggest = AsyncMock(
        return_value=MaintenanceSuggestionResponse(
            triggered=False,
            failure_probability=0.5,
            markdown=None,
            references=[],
            model=None,
            message="Probabilidade de falha (0.50) não excede o limiar de 0.7 — sugestão automática não acionada.",
        )
    )
    async with await _client_for(fake_service) as client:
        response = await client.post(
            "/v1/maintenance/suggest",
            json={**_VALID_PAYLOAD, "failure_probability": 0.5},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["triggered"] is False
    assert body["markdown"] is None
    assert body["references"] == []
    assert "não excede" in body["message"]


# ---------------------------------------------------------------------------
# Contrato HTTP — disparado, plano gerado
# ---------------------------------------------------------------------------


async def test_endpoint_returns_200_with_markdown_plan_above_threshold() -> None:
    fake_service = AsyncMock()
    fake_service.suggest = AsyncMock(
        return_value=MaintenanceSuggestionResponse(
            triggered=True,
            failure_probability=0.9,
            markdown="# Plano de Manutenção\n\n## Diagnóstico provável\nX",
            references=[
                {
                    "file_name": "manual-bomba-centrifuga.pdf",
                    "page": 3,
                    "chunk_index": 0,
                    "source": "manual-bomba-centrifuga.pdf",
                    "score": 0.71,
                }
            ],
            model="llama3.2:3b",
            message=None,
        )
    )
    async with await _client_for(fake_service) as client:
        response = await client.post(
            "/v1/maintenance/suggest",
            json={**_VALID_PAYLOAD, "failure_probability": 0.9},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["triggered"] is True
    assert isinstance(body["markdown"], str) and body["markdown"].startswith("#")
    assert body["model"] == "llama3.2:3b"
    assert body["references"][0]["page"] == 3


# ---------------------------------------------------------------------------
# Contrato HTTP — MCP indisponível vira 503 padronizado (core/exceptions.py)
# ---------------------------------------------------------------------------


async def test_endpoint_returns_503_when_mcp_unavailable() -> None:
    fake_service = AsyncMock()
    fake_service.suggest = AsyncMock(side_effect=MCPUnavailableError())

    async with await _client_for(fake_service) as client:
        response = await client.post(
            "/v1/maintenance/suggest",
            json={**_VALID_PAYLOAD, "failure_probability": 0.9},
        )

    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "MCPUnavailableError"
    assert "http://" not in body["detail"]  # nunca expõe a URL interna do serviço
    assert "mcp-server" not in body["detail"]  # nem o nome do host interno


# ---------------------------------------------------------------------------
# Validação de schema — probabilidade fora de [0, 1]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("invalid_probability", [-0.1, 1.5])
async def test_endpoint_rejects_out_of_range_probability(
    invalid_probability: float,
) -> None:
    fake_service = AsyncMock()
    async with await _client_for(fake_service) as client:
        response = await client.post(
            "/v1/maintenance/suggest",
            json={**_VALID_PAYLOAD, "failure_probability": invalid_probability},
        )

    assert response.status_code == 422
    fake_service.suggest.assert_not_called()


# ---------------------------------------------------------------------------
# Contrato HTTP — SSE (RF-23 / RNF-47)
# ---------------------------------------------------------------------------


def _events_from_sse_body(body: str) -> list[tuple[str, str]]:
    """Parser mínimo só para os testes: devolve [(event_type, data_json), ...]."""
    events: list[tuple[str, str]] = []
    for block in body.strip().split("\n\n"):
        if not block.strip():
            continue
        event_type = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                event_type = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        events.append((event_type, data))
    return events


async def test_stream_endpoint_returns_only_skipped_below_threshold() -> None:
    async def fake_stream(payload):
        yield SuggestionStreamEvent(type="skipped", message="não excede 0.7")

    fake_service = AsyncMock()
    fake_service.suggest_stream = fake_stream

    async with await _client_for(fake_service) as client:
        response = await client.post(
            "/v1/maintenance/suggest/stream",
            json={**_VALID_PAYLOAD, "failure_probability": 0.5},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events_from_sse_body(response.text)
    assert len(events) == 1
    assert events[0][0] == "skipped"
    assert "não excede" in json.loads(events[0][1])["message"]


async def test_stream_endpoint_emits_token_events_then_done_with_references() -> None:
    async def fake_stream(payload):
        yield SuggestionStreamEvent(type="token", token="# Plano")
        yield SuggestionStreamEvent(type="token", token=" de manutenção")
        yield SuggestionStreamEvent(
            type="done",
            markdown="# Plano de manutenção",
            references=[
                ManualReference(
                    file_name="manual-bomba-centrifuga.pdf",
                    page=3,
                    chunk_index=0,
                    source="manual-bomba-centrifuga.pdf",
                    score=0.71,
                )
            ],
        )

    fake_service = AsyncMock()
    fake_service.suggest_stream = fake_stream

    async with await _client_for(fake_service) as client:
        response = await client.post(
            "/v1/maintenance/suggest/stream",
            json={**_VALID_PAYLOAD, "failure_probability": 0.9},
        )

    assert response.status_code == 200
    events = _events_from_sse_body(response.text)
    types = [e[0] for e in events]
    assert types == ["token", "token", "done"]

    token1 = json.loads(events[0][1])
    token2 = json.loads(events[1][1])
    assert token1["token"] == "# Plano"
    assert token2["token"] == " de manutenção"
    # Cada `data:` é JSON válido, verificado via json.loads acima — nunca
    # texto solto ou JSON malformado (RF-23 §2).

    done = json.loads(events[2][1])
    assert done["markdown"] == "# Plano de manutenção"
    assert done["references"][0]["file_name"] == "manual-bomba-centrifuga.pdf"
    assert done["references"][0]["page"] == 3


async def test_stream_endpoint_emits_error_event_on_service_exception() -> None:
    """Rede de segurança do router: uma exceção não prevista pelo service
    ainda fecha o SSE com um evento `error` estruturado, não uma conexão
    quebrada nem um traceback exposto."""

    async def fake_stream(payload):
        yield SuggestionStreamEvent(type="token", token="parcial")
        raise RuntimeError("algo inesperado quebrou")

    fake_service = AsyncMock()
    fake_service.suggest_stream = fake_stream

    async with await _client_for(fake_service) as client:
        response = await client.post(
            "/v1/maintenance/suggest/stream",
            json={**_VALID_PAYLOAD, "failure_probability": 0.9},
        )

    assert (
        response.status_code == 200
    )  # SSE já iniciado — erro vem como evento, não status HTTP
    events = _events_from_sse_body(response.text)
    assert events[0][0] == "token"
    assert events[-1][0] == "error"
    error_data = json.loads(events[-1][1])
    assert (
        "algo inesperado quebrou" not in error_data["message"]
    )  # nunca vaza a exceção crua
