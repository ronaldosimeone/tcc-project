"""
Testes HTTP de `POST /v1/maintenance/suggest` (RF-22 / RNF-46).

Nível de integração do router: `MaintenanceSuggestionService` é substituído
via `dependency_overrides` (mesmo padrão de `test_predict_endpoint.py`) —
não testa MCP/Ollama reais aqui (isso é `test_maintenance_suggestion.py`
para o service, e a validação real documentada no README para o E2E).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.core.exceptions import MCPUnavailableError
from src.main import create_app
from src.routers.maintenance import get_maintenance_suggestion_service
from src.schemas.maintenance import MaintenanceSuggestionResponse

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
