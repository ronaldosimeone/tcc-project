"""
RNF-63 — testes de privacidade dos logs.

Cada teste emite um evento REAL pelo ``structlog`` já configurado por
``configure_logging`` (a MESMA função usada em produção, com o MESMO
encadeamento de processadores) e inspeciona a **linha JSON efetivamente
escrita**. Não testa ``sanitize(x) == y`` isolado — isso não provaria que o
logger usa o sanitizer. Aqui, se o processor sair da cadeia em
``src/core/logging.py``, estes testes quebram.

Cobre (item 6 da RNF-63): e-mail, telefone, Authorization Bearer, API key,
senha, cookie, Telegram chat_id, payload multi-campo, objeto aninhado,
exceção contendo secret, lista/dict com PII — e também que o contexto útil
e não sensível continua aparecendo.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Iterator

import pytest
import structlog

from src.core.logging import configure_logging

_SECRETS = [
    "operador@empresa.com.br",  # e-mail
    "not-a-real-key-just-fixture-value-000000",  # API key
    "SuperSenh4!Secreta",  # senha
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.PAYLOAD.sig",  # JWT bearer
    "123456789:AAFAKE-bot-token-abcdefghijklmnop",  # Telegram bot token
    "sessioncookievalue=abc123def456",  # cookie
]


@pytest.fixture()
def log_lines() -> Iterator[Callable[..., dict]]:
    """Configura o logging de produção (JSON) escrevendo num buffer e
    devolve um callable `log_lines(event, **kw) -> dict` — cada chamada
    emite um evento real pelo `structlog` configurado e retorna a linha
    JSON já parseada (o `dict` efetivamente escrito)."""
    buf = io.StringIO()
    configure_logging(debug=False, stream=buf)
    log = structlog.get_logger("test.privacy")
    lines: list[dict] = []

    class _Emit:
        def __call__(self, event: str, **kw: object) -> dict:
            buf.seek(0)
            buf.truncate(0)
            log.info(event, **kw)
            raw = buf.getvalue().strip()
            assert raw, "logger não emitiu nada"
            parsed = json.loads(raw)
            lines.append(parsed)
            return parsed

        @property
        def raw(self) -> str:
            return buf.getvalue()

    yield _Emit()
    # Restaura a config padrão para não vazar o buffer para outros testes.
    configure_logging(debug=False)


def _assert_no_secret(blob: str) -> None:
    for secret in _SECRETS:
        assert secret not in blob, f"segredo vazou no log: {secret!r}"


# ── Casos individuais ────────────────────────────────────────────────────


def test_email_is_redacted(log_lines):
    out = log_lines("user_registered", alert_email="operador@empresa.com.br")
    _assert_no_secret(json.dumps(out))
    assert out["alert_email"] == "***"


def test_phone_number_in_message_is_scrubbed(log_lines):
    # Telefone dentro de um campo livre — o e-mail scrubber não pega número,
    # mas um campo chamado "phone" deve ser redigido por nome de chave.
    out = log_lines("contact_saved", phone="+55 11 98765-4321")
    assert out["phone"] == "***"


def test_authorization_bearer_is_redacted(log_lines):
    jwt = _SECRETS[3]
    out = log_lines("incoming_request", authorization=f"Bearer {jwt}")
    _assert_no_secret(json.dumps(out))
    assert out["authorization"] == "***"


def test_bearer_token_inside_free_text_is_scrubbed(log_lines):
    jwt = _SECRETS[3]
    out = log_lines("upstream_call", detail=f"chamou com header Bearer {jwt} e falhou")
    _assert_no_secret(json.dumps(out))
    assert "Bearer ***" in out["detail"]


def test_api_key_is_redacted(log_lines):
    out = log_lines(
        "provider_configured", api_key=_SECRETS[1], resend_api_key=_SECRETS[1]
    )
    _assert_no_secret(json.dumps(out))
    assert out["api_key"] == "***"
    assert out["resend_api_key"] == "***"


def test_password_is_redacted(log_lines):
    out = log_lines("db_connect", password=_SECRETS[2], passwd=_SECRETS[2])
    _assert_no_secret(json.dumps(out))
    assert out["password"] == "***" and out["passwd"] == "***"


def test_cookie_is_redacted(log_lines):
    out = log_lines("ws_handshake", cookie=_SECRETS[5])
    _assert_no_secret(json.dumps(out))
    assert out["cookie"] == "***"


def test_telegram_chat_id_is_redacted(log_lines):
    out = log_lines(
        "telegram_send", chat_id="-1001234567890", telegram_chat_id="99887766"
    )
    assert out["chat_id"] == "***" and out["telegram_chat_id"] == "***"


def test_telegram_bot_url_is_scrubbed(log_lines):
    url = "https://api.telegram.org/bot123456789:AAFAKE-bot-token-abcdefghijklmnop/sendMessage"
    out = log_lines("telegram_http_error", url=url)
    _assert_no_secret(json.dumps(out))
    assert "/bot***" in out["url"] and "AAFAKE" not in out["url"]


def test_database_dsn_password_is_scrubbed(log_lines):
    dsn = "postgresql+asyncpg://tccuser:s3cr3tp4ss@db:5432/tcc_db"
    out = log_lines("engine_error", error=f"could not connect: {dsn}")
    assert "s3cr3tp4ss" not in json.dumps(out)
    assert "tccuser:***@db" in out["error"]


def test_multi_field_payload(log_lines):
    out = log_lines(
        "settings_saved",
        alert_email="operador@empresa.com.br",
        admin_api_token="tok_live_admin_000",
        alert_threshold=0.85,
        telegram_enabled=True,
    )
    _assert_no_secret(json.dumps(out))
    assert out["alert_email"] == "***"
    assert out["admin_api_token"] == "***"
    # Contexto útil preservado:
    assert out["alert_threshold"] == 0.85
    assert out["telegram_enabled"] is True


def test_nested_object(log_lines):
    out = log_lines(
        "request_context",
        request={
            "path": "/v1/settings/alerts",
            "headers": {
                "authorization": "Bearer " + _SECRETS[3],
                "x-request-id": "abc",
            },
            "user": {"email": "operador@empresa.com.br", "id": 7},
        },
    )
    blob = json.dumps(out)
    _assert_no_secret(blob)
    assert out["request"]["headers"]["authorization"] == "***"
    assert out["request"]["user"]["email"] == "***"
    # Não sensível preservado:
    assert out["request"]["path"] == "/v1/settings/alerts"
    assert out["request"]["headers"]["x-request-id"] == "abc"
    assert out["request"]["user"]["id"] == 7


def test_exception_containing_secret(log_lines):
    buf_line = None
    try:
        raise RuntimeError(
            "falha ao autenticar com Bearer " + _SECRETS[3] + " no endpoint"
        )
    except RuntimeError:
        buf_line = log_lines("unhandled_exception", exc_info=True)
    assert buf_line is not None
    blob = json.dumps(buf_line)
    _assert_no_secret(blob)
    assert "Bearer ***" in blob  # traceback foi varrido, mas continua útil
    assert "RuntimeError" in blob  # tipo da exceção preservado p/ diagnóstico


def test_list_and_dict_with_pii(log_lines):
    out = log_lines(
        "batch_notified",
        recipients=["a@x.com", "b@y.com", "c@z.com"],
        meta={"secret": "topsecret", "count": 3},
    )
    blob = json.dumps(out)
    assert "a@x.com" not in blob and "topsecret" not in blob
    assert out["meta"]["secret"] == "***"
    assert out["meta"]["count"] == 3


def test_client_ip_is_hashed_not_stored(log_lines):
    out = log_lines(
        "sse_connection_opened", client="Address(host='203.0.113.45', port=54321)"
    )
    blob = json.dumps(out)
    assert "203.0.113.45" not in blob
    assert out["client"].startswith("ip#")
    # Estável: mesma entrada → mesmo hash (permite correlacionar "mesmo cliente")
    out2 = log_lines(
        "sse_connection_closed", client="Address(host='203.0.113.45', port=54321)"
    )
    assert out2["client"] == out["client"]


def test_bare_ip_value_is_hashed(log_lines):
    out = log_lines("ws_send_failed", client="198.51.100.7:8080")
    assert "198.51.100.7" not in json.dumps(out)
    assert out["client"].startswith("ip#")


# ── Contra-prova: contexto de diagnóstico NÃO sensível continua visível ──


def test_useful_diagnostic_context_is_preserved(log_lines):
    out = log_lines(
        "inference_completed",
        equipment_id="APU-Trem-042",
        probability=0.91,
        predicted_class=1,
        latency_ms=12.4,
        model="random_forest_v2",
    )
    assert out["event"] == "inference_completed"
    assert out["equipment_id"] == "APU-Trem-042"
    assert out["probability"] == 0.91
    assert out["predicted_class"] == 1
    assert out["latency_ms"] == 12.4
    assert out["model"] == "random_forest_v2"
    assert out["level"] == "info"
    assert "timestamp" in out
