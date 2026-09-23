"""
Testes de `observability_service.get_error_rate_status` (RNF-77).

Mesmo padrão já estabelecido pelo projeto para `OllamaClient`/
`TelegramNotificationAdapter` (ver test_ollama_client.py):
`monkeypatch.setattr(httpx.AsyncClient, "get", AsyncMock(return_value=httpx.Response(...)))`
— nunca fala com um Prometheus real, mas exercita o parsing/classificação
de verdade contra um corpo de resposta no formato real da API do
Prometheus (`/api/v1/query`).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx

from src.core.config import settings
from src.services.observability_service import get_error_rate_status

_URL = f"{settings.prometheus_url}/api/v1/query"


def _prom_response(value: str) -> httpx.Response:
    """Corpo real de `/api/v1/query` com resultado escalar único."""
    return httpx.Response(
        status_code=200,
        json={
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [{"metric": {}, "value": [1758000000, value]}],
            },
        },
        request=httpx.Request("GET", _URL),
    )


def _prom_empty_response() -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json={"status": "success", "data": {"resultType": "vector", "result": []}},
        request=httpx.Request("GET", _URL),
    )


async def test_error_rate_below_warning_threshold_is_normal(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx.AsyncClient, "get", AsyncMock(return_value=_prom_response("0.001"))
    )
    result = await get_error_rate_status()
    assert result.status == "NORMAL"
    assert result.error_rate == 0.001
    assert result.prometheus_reachable is True


async def test_error_rate_between_thresholds_is_warning(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx.AsyncClient, "get", AsyncMock(return_value=_prom_response("0.02"))
    )
    result = await get_error_rate_status()
    assert result.status == "WARNING"


async def test_error_rate_above_critical_threshold_is_critical(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx.AsyncClient, "get", AsyncMock(return_value=_prom_response("0.10"))
    )
    result = await get_error_rate_status()
    assert result.status == "CRITICAL"


async def test_error_rate_exactly_at_warning_threshold_is_warning(monkeypatch) -> None:
    """Fronteira: a comparação real é `>=`, não `>` — igual ao limiar já
    classifica como WARNING/CRITICAL (mesmo estilo de teste de fronteira já
    usado em `test_model_service_unit.py`)."""
    monkeypatch.setattr(
        httpx.AsyncClient,
        "get",
        AsyncMock(
            return_value=_prom_response(str(settings.error_rate_warning_threshold))
        ),
    )
    result = await get_error_rate_status()
    assert result.status == "WARNING"


async def test_error_rate_exactly_at_critical_threshold_is_critical(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        httpx.AsyncClient,
        "get",
        AsyncMock(
            return_value=_prom_response(str(settings.error_rate_critical_threshold))
        ),
    )
    result = await get_error_rate_status()
    assert result.status == "CRITICAL"


async def test_no_traffic_yet_empty_result_is_zero_error_rate_normal(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        httpx.AsyncClient, "get", AsyncMock(return_value=_prom_empty_response())
    )
    result = await get_error_rate_status()
    assert result.status == "NORMAL"
    assert result.error_rate == 0.0
    assert result.prometheus_reachable is True


async def test_nan_from_zero_over_zero_rate_is_treated_as_zero(monkeypatch) -> None:
    """`rate()/rate()` com denominador 0 devolve NaN no Prometheus — nunca
    deve virar CRITICAL nem quebrar a serialização JSON."""
    monkeypatch.setattr(
        httpx.AsyncClient, "get", AsyncMock(return_value=_prom_response("NaN"))
    )
    result = await get_error_rate_status()
    assert result.status == "NORMAL"
    assert result.error_rate == 0.0


async def test_prometheus_connection_error_degrades_to_normal_and_unreachable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        httpx.AsyncClient,
        "get",
        AsyncMock(side_effect=httpx.ConnectError("refused")),
    )
    result = await get_error_rate_status()
    assert result.status == "NORMAL"
    assert result.error_rate == 0.0
    assert result.prometheus_reachable is False


async def test_prometheus_http_error_status_degrades_to_normal_and_unreachable(
    monkeypatch,
) -> None:
    error_response = httpx.Response(
        status_code=500, text="internal", request=httpx.Request("GET", _URL)
    )
    monkeypatch.setattr(
        httpx.AsyncClient, "get", AsyncMock(return_value=error_response)
    )
    result = await get_error_rate_status()
    assert result.prometheus_reachable is False


async def test_prometheus_query_status_not_success_degrades_to_unreachable(
    monkeypatch,
) -> None:
    bad = httpx.Response(
        status_code=200,
        json={"status": "error", "error": "bad query"},
        request=httpx.Request("GET", _URL),
    )
    monkeypatch.setattr(httpx.AsyncClient, "get", AsyncMock(return_value=bad))
    result = await get_error_rate_status()
    assert result.prometheus_reachable is False


async def test_response_includes_configured_thresholds_and_window(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx.AsyncClient, "get", AsyncMock(return_value=_prom_response("0.0"))
    )
    result = await get_error_rate_status()
    assert result.window == "5m"
    assert result.threshold_warning == settings.error_rate_warning_threshold
    assert result.threshold_critical == settings.error_rate_critical_threshold
