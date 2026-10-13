"""Silenciamento dos loggers de clientes HTTP (token do Telegram no path)."""

from __future__ import annotations

import logging

import pytest

from src.core.logging import configure_logging, silence_http_client_loggers

_TELEGRAM_URL = "https://api.telegram.org/bot123456:FAKE-TOKEN/sendMessage"


@pytest.mark.parametrize("name", ["httpx", "httpcore"])
def test_info_do_cliente_http_nao_e_emitido(
    name: str, caplog: pytest.LogCaptureFixture
) -> None:
    silence_http_client_loggers()
    caplog.set_level(logging.DEBUG)

    logging.getLogger(name).info("HTTP Request: POST %s", _TELEGRAM_URL)

    assert not [r for r in caplog.records if r.name == name]
    assert "FAKE-TOKEN" not in caplog.text


def test_warning_do_cliente_http_continua_visivel(
    caplog: pytest.LogCaptureFixture,
) -> None:
    silence_http_client_loggers()
    caplog.set_level(logging.DEBUG)

    logging.getLogger("httpx").warning("falha de conexao")

    assert any(r.name == "httpx" for r in caplog.records)


def test_configure_logging_aplica_o_silenciamento() -> None:
    logging.getLogger("httpx").setLevel(logging.NOTSET)

    configure_logging(debug=True)

    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING
