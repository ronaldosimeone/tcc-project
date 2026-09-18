"""
Centralised log sanitisation — RNF-63 (privacidade dos logs).

Um único ``structlog`` processor (:func:`redact_sensitive`) roda no fim da
cadeia de processadores, imediatamente antes do renderer, e é o ÚNICO ponto
onde a redação de dados sensíveis acontece — nenhuma chamada
``log.*`` espalhada pelo código precisa se preocupar com isso (a auditoria
da RNF-63 encontrou o código já cuidadoso caso a caso, mas sem nenhuma
rede de segurança centralizada: um único ``log.info("x", user=user)``
descuidado no futuro vazaria PII).

O que é redigido
----------------
1. **Por nome de chave** (recursivo, case-insensitive): ``authorization``,
   ``token``, ``*_token``, ``api_key``/``apikey``, ``secret``, ``password``,
   ``passwd``, ``cookie``/``set-cookie``, ``x-admin-token``, ``chat_id``
   (Telegram), ``bot_token``, ``credential(s)`` → valor vira ``"***"``.
2. **Por formato do valor** (em qualquer string, recursivo): tokens
   ``Bearer <x>``, URLs de Bot API do Telegram (``/bot<id>:<token>/``),
   credenciais embutidas em DSN (``postgresql://user:senha@`` /
   ``redis://:senha@``), e endereços de e-mail → substring trocada por um
   marcador.
3. **Endereço de rede do cliente** (``client``, ``peer``, ``remote_addr``,
   ou o ``repr`` de ``starlette.datastructures.Address``): o IP é
   substituído por ``ip#<8 hex>`` — um hash estável e curto que preserva a
   capacidade de dizer "mesmo cliente ou não" entre linhas de log para
   diagnóstico, sem registrar o IP em si (RNF-63 — IP como dado pessoal).

O sanitizer NUNCA muta os objetos originais usados pela aplicação — ele
constrói uma cópia sanitizada das estruturas que toca. Só afeta a
representação destinada ao log.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Final

from structlog.types import EventDict

# Profundidade máxima de recursão em dicts/listas aninhados — além disso o
# valor é substituído por um marcador (evita loop infinito em estruturas
# cíclicas e limita o custo de logar um objeto gigante por engano).
_MAX_DEPTH: Final[int] = 6
# Nº máximo de itens percorridos numa lista/tupla antes de truncar.
_MAX_SEQ_ITEMS: Final[int] = 50
# Comprimento máximo de uma string de valor antes de truncar (defesa contra
# um payload gigante ir parar no log).
_MAX_STR_LEN: Final[int] = 2_048

_REDACTED: Final[str] = "***"

# ── 1. Chaves sensíveis ──────────────────────────────────────────────────
# Substring, case-insensitive. Uma chave que contenha qualquer um destes
# tem o valor inteiro redigido, sem olhar o formato.
_SENSITIVE_KEY_RE: Final[re.Pattern[str]] = re.compile(
    r"""(?ix)
    (
        authorization | www-authenticate
      | (^|[_\-])token($|[_\-]) | ^token$ | access[_\-]?token | refresh[_\-]?token
      | bot[_\-]?token
      | api[_\-]?key | apikey
      | secret
      | password | passwd | pwd
      | (^|[_\-])cookie($|[_\-]) | ^cookie$ | set-cookie
      | x-admin-token | admin[_\-]?token
      | chat[_\-]?id
      | credential
      | session[_\-]?id
      # ── PII direta (RNF-63) ── ancorado no FIM da chave para não pegar
      # flags booleanas tipo `email_enabled` / `alert_email_configured`.
      | (^|[_\-])e?mail$ | ^mail$
      | (^|[_\-])phone$ | ^telefone$ | (^|[_\-])tel$
      | full[_\-]?name$ | first[_\-]?name$ | last[_\-]?name$ | ^person$ | person[_\-]?name$
      | user[_\-]?name$ | display[_\-]?name$
      | (^|[_\-])address$ | ^endereco$ | ^street$ | ^cep$ | ^zipcode$ | postal[_\-]?code$
    )
    """
)

# ── 2. Formatos de valor sensível (em texto livre) ───────────────────────
_VALUE_SCRUBBERS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    # Authorization: Bearer <token>  /  "Bearer eyJ..."
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-+/=]+"), "Bearer ***"),
    # Telegram Bot API URL: https://api.telegram.org/bot<id>:<token>/sendMessage
    (re.compile(r"/bot\d+:[A-Za-z0-9_\-]+"), "/bot***"),
    # Credenciais em DSN: postgresql+asyncpg://user:senha@host  →  ...://user:***@host
    (
        re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)([^:/?#@\s]+):([^@/?#\s]+)@"),
        r"\1\2:***@",
    ),
    # redis://:senha@host  (sem usuário)
    (re.compile(r"(?i)\b(redis://):([^@/?#\s]+)@"), r"\1:***@"),
    # Endereço de e-mail  →  ***@***
    (
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        "***@***",
    ),
)

# ── 3. Endereço do cliente ───────────────────────────────────────────────
_CLIENT_KEY_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)^(client|peer|remote[_\-]?addr|x-real-ip|x-forwarded-for|host_ip)$"
)
# repr de starlette.datastructures.Address: Address(host='1.2.3.4', port=5678)
_ADDRESS_REPR_RE: Final[re.Pattern[str]] = re.compile(
    r"Address\(host=['\"]?(?P<host>[^'\",)]+)['\"]?,\s*port=(?P<port>\d+)\)"
)
_BARE_IP_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?P<host>(\d{1,3}\.){3}\d{1,3}|[0-9a-fA-F:]+:[0-9a-fA-F:]*)"
    r"(:(?P<port>\d+))?\s*$"
)


def _hash_host(host: str) -> str:
    """8 hex chars — estável por processo/host, não reversível para o IP."""
    return "ip#" + hashlib.sha256(host.encode("utf-8", "replace")).hexdigest()[:8]


def _redact_client_string(value: str) -> str | None:
    """Se ``value`` parece um endereço de cliente, devolve a versão
    redigida; senão ``None``."""
    m = _ADDRESS_REPR_RE.search(value)
    if m:
        tag = _hash_host(m.group("host"))
        return f"{tag}:{m.group('port')}" if m.group("port") else tag
    m = _BARE_IP_RE.match(value)
    if m and (":" in value or "." in value):
        tag = _hash_host(m.group("host"))
        return f"{tag}:{m.group('port')}" if m.group("port") else tag
    return None


def _scrub_str(value: str) -> str:
    if len(value) > _MAX_STR_LEN:
        value = value[:_MAX_STR_LEN] + "…[truncado]"
    for pattern, repl in _VALUE_SCRUBBERS:
        value = pattern.sub(repl, value)
    return value


def _sanitize(value: Any, *, depth: int, key_hint: str | None = None) -> Any:
    """Devolve uma cópia sanitizada de ``value``. Nunca muta a entrada."""
    if key_hint is not None and _SENSITIVE_KEY_RE.search(key_hint):
        return _REDACTED

    if depth > _MAX_DEPTH:
        return "…[profundo demais]"

    if isinstance(value, str):
        if key_hint is not None and _CLIENT_KEY_RE.match(key_hint):
            return _redact_client_string(value) or _hash_host(value)
        redacted_client = _redact_client_string(value)
        if redacted_client is not None:
            return redacted_client
        return _scrub_str(value)

    if isinstance(value, dict):
        return {
            k: _sanitize(v, depth=depth + 1, key_hint=str(k))
            for k, v in list(value.items())[: _MAX_SEQ_ITEMS * 4]
        }

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        out = [
            _sanitize(v, depth=depth + 1, key_hint=key_hint)
            for v in items[:_MAX_SEQ_ITEMS]
        ]
        if len(items) > _MAX_SEQ_ITEMS:
            out.append(f"…[+{len(items) - _MAX_SEQ_ITEMS} itens]")
        return out

    if isinstance(value, (int, float, bool)) or value is None:
        return value

    # Qualquer outro objeto (dataclass, modelo, Address, exceção…): usa o
    # repr, mas passa pelo scrub de string e pela detecção de cliente.
    text = _scrub_str(repr(value))
    return _redact_client_string(text) or text


def redact_sensitive(
    _logger: Any, _method_name: str, event_dict: EventDict
) -> EventDict:
    """``structlog`` processor — redige PII/segredos do ``event_dict``.

    Colocado no fim da cadeia (logo antes do renderer). Preserva
    ``level``/``timestamp``/``logger``/``exc_info`` como estão (chaves de
    infraestrutura); a mensagem (``event``) tem só o scrub de formato
    aplicado (Bearer/DSN/e-mail), nunca a redação por nome de chave; todo o
    resto do payload passa pela sanitização completa.
    """
    _PASSTHROUGH_KEYS = {"level", "timestamp", "logger", "exc_info"}
    out: dict[str, Any] = {}
    for key, value in event_dict.items():
        if key in _PASSTHROUGH_KEYS:
            out[key] = value
        elif key == "event":
            out[key] = _scrub_str(value) if isinstance(value, str) else value
        elif key == "exception" and isinstance(value, str):
            # Traceback já renderizado (produção): aplica só o scrub de
            # formato (Bearer/DSN/e-mail), SEM truncar — o traceback
            # completo é o que dá valor de diagnóstico.
            out[key] = _scrub_exception_text(value)
        else:
            out[key] = _sanitize(value, depth=0, key_hint=str(key))
    return out


def _scrub_exception_text(text: str) -> str:
    for pattern, repl in _VALUE_SCRUBBERS:
        text = pattern.sub(repl, text)
    return text
