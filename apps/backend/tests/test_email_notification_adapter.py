"""
Testes de `EmailNotificationAdapter` (RF-25 / RNF-49, RNF-64/RNF-65).

Auditoria desta task: 37% de cobertura — só `build_critical_failure_email_text`
(helper puro) era testado indiretamente; o envio HTTP em si (`_send`) nunca
era exercitado. Mesmo padrão de mock estabelecido pelo projeto para os outros
adapters HTTP (`TelegramNotificationAdapter`/`OllamaClient`):
`monkeypatch.setattr(httpx.AsyncClient, "post", ...)` com `httpx.Response` real.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from src.core.exceptions import EmailNotificationError
from src.services.email_notification_adapter import (
    EmailNotificationAdapter,
    build_critical_failure_email_text,
)

API_BASE = "https://api.resend.fake"
FAKE_KEY = "re_fake_never_a_real_key_0000000000"


def _response(status_code: int, json_body: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=json_body if json_body is not None else {"id": "email_123"},
        request=httpx.Request("POST", f"{API_BASE}/emails"),
    )


def _adapter(**overrides: Any) -> EmailNotificationAdapter:
    kwargs: dict[str, Any] = dict(
        api_key=FAKE_KEY,
        from_email="alerts@predictiq.example",
        api_base_url=API_BASE,
    )
    kwargs.update(overrides)
    return EmailNotificationAdapter(**kwargs)


# ---------------------------------------------------------------------------
# build_critical_failure_email_text — helper puro
# ---------------------------------------------------------------------------


def test_email_text_contains_all_required_fields() -> None:
    text = build_critical_failure_email_text(
        equipment_name="Compressor A",
        probability=0.923,
        timestamp="2026-09-18T12:00:00+00:00",
        dashboard_url="http://localhost/sensors/A",
    )
    assert "Compressor A" in text
    assert "92%" in text  # :.0% — 0.923 -> "92%"
    assert "2026-09-18T12:00:00+00:00" in text
    assert "http://localhost/sensors/A" in text
    # Sem HTML — nenhuma tag deve aparecer no corpo (RF-25: sem superfície
    # de injeção de markup).
    assert "<" not in text and ">" not in text


def test_email_text_probability_rounds_to_nearest_percent() -> None:
    text = build_critical_failure_email_text(
        equipment_name="X",
        probability=0.995,
        timestamp="t",
        dashboard_url="u",
    )
    assert "100%" in text or "99%" in text  # round-half-even do format spec :.0%
    # valor exato produzido pelo format spec do Python — trava o comportamento.
    assert f"{0.995:.0%}" in text


# ---------------------------------------------------------------------------
# _send / send_critical_failure_email / send_test_email
# ---------------------------------------------------------------------------


async def test_send_critical_failure_email_success(monkeypatch) -> None:
    mock_post = AsyncMock(return_value=_response(200))
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    adapter = _adapter()
    await adapter.send_critical_failure_email(
        to="ops@empresa.com",
        equipment_name="Compressor A",
        probability=0.9,
        timestamp="2026-09-18T12:00:00+00:00",
        dashboard_url="http://localhost/sensors/A",
    )

    mock_post.assert_awaited_once()
    assert mock_post.await_args is not None
    args, kwargs = mock_post.await_args
    assert args[0] == f"{API_BASE}/emails"
    assert kwargs["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    assert kwargs["json"]["to"] == ["ops@empresa.com"]
    assert kwargs["json"]["subject"] == "PredictIQ — Falha crítica detectada"
    assert "Compressor A" in kwargs["json"]["text"]


async def test_send_test_email_uses_test_subject_never_the_failure_text(
    monkeypatch,
) -> None:
    mock_post = AsyncMock(return_value=_response(200))
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    adapter = _adapter()
    await adapter.send_test_email(to="ops@empresa.com")

    assert mock_post.await_args is not None
    _, kwargs = mock_post.await_args
    assert kwargs["json"]["subject"] == "PredictIQ — Teste de Notificação"
    assert "FALHA CRÍTICA" not in kwargs["json"]["text"]


async def test_send_accepts_201_as_success(monkeypatch) -> None:
    """Resend documenta 201 Created para POST /emails — 200 não é o único
    status de sucesso; um teste que só aceitasse 200 seria frágil demais e
    quebraria em produção."""
    monkeypatch.setattr(
        httpx.AsyncClient, "post", AsyncMock(return_value=_response(201))
    )
    adapter = _adapter()
    await adapter.send_test_email(to="ops@empresa.com")  # não deve lançar


@pytest.mark.parametrize("missing", ["api_key", "to"])
async def test_send_missing_config_or_recipient_raises_without_http_call(
    monkeypatch, missing: str
) -> None:
    mock_post = AsyncMock()
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    adapter = _adapter(api_key=None if missing == "api_key" else FAKE_KEY)
    to = "" if missing == "to" else "ops@empresa.com"

    with pytest.raises(EmailNotificationError, match="ausente"):
        await adapter.send_test_email(to=to)
    mock_post.assert_not_awaited()  # falha ANTES de qualquer chamada de rede


async def test_send_timeout_raises_email_notification_error(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx.AsyncClient, "post", AsyncMock(side_effect=httpx.TimeoutException("x"))
    )
    with pytest.raises(EmailNotificationError, match="Timeout"):
        await _adapter().send_test_email(to="ops@empresa.com")


async def test_send_connection_error_raises_email_notification_error(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        httpx.AsyncClient, "post", AsyncMock(side_effect=httpx.ConnectError("refused"))
    )
    with pytest.raises(EmailNotificationError, match="conectar"):
        await _adapter().send_test_email(to="ops@empresa.com")


@pytest.mark.parametrize("status_code", [400, 401, 429, 500])
async def test_send_non_2xx_status_raises_email_notification_error(
    monkeypatch, status_code: int
) -> None:
    monkeypatch.setattr(
        httpx.AsyncClient, "post", AsyncMock(return_value=_response(status_code))
    )
    with pytest.raises(EmailNotificationError, match=str(status_code)):
        await _adapter().send_test_email(to="ops@empresa.com")


async def test_send_invalid_json_response_raises_email_notification_error(
    monkeypatch,
) -> None:
    bad = httpx.Response(
        200, content=b"not json", request=httpx.Request("POST", f"{API_BASE}/emails")
    )
    monkeypatch.setattr(httpx.AsyncClient, "post", AsyncMock(return_value=bad))
    with pytest.raises(EmailNotificationError, match="JSON válido"):
        await _adapter().send_test_email(to="ops@empresa.com")


async def test_api_base_url_trailing_slash_is_stripped(monkeypatch) -> None:
    mock_post = AsyncMock(return_value=_response(200))
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    adapter = _adapter(api_base_url=f"{API_BASE}/")
    await adapter.send_test_email(to="ops@empresa.com")

    assert mock_post.await_args is not None
    url = mock_post.await_args.args[0]
    assert url == f"{API_BASE}/emails"  # nunca "...//emails"
