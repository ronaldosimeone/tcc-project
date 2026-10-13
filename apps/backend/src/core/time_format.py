"""
Formatação de data/hora para mensagens ao operador (Telegram, e-mail).

O pipeline carrega timestamps em ISO-8601 UTC (ex.:
`2026-10-13T15:26:46.857123+00:00`) — ótimo para máquina, ilegível e
confuso para quem recebe o alerta. Aqui o instante é convertido para o
fuso configurado em `NOTIFICATION_TIMEZONE` e exibido em pt-BR, sempre com
o offset explícito para não haver ambiguidade.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.core.config import settings


def _resolve_timezone(tz_name: str) -> tzinfo:
    try:
        return ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        return timezone.utc


def _offset_label(offset: timedelta) -> str:
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    hours, minutes = divmod(abs(total_minutes), 60)
    return f"UTC{sign}{hours}" if minutes == 0 else f"UTC{sign}{hours}:{minutes:02d}"


def format_local_timestamp(timestamp: str, *, tz_name: str | None = None) -> str:
    """
    ISO-8601 → `13/10/2026 às 12:26:46 (UTC-3)`.

    Texto que não é ISO-8601 volta intacto (nunca lança) — a mensagem de
    alerta não pode deixar de ser enviada por causa de um formato inesperado.
    Datas sem fuso são tratadas como UTC, que é o contrato do pipeline.
    """
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return timestamp
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    zone = _resolve_timezone(tz_name or settings.notification_timezone)
    local = parsed.astimezone(zone)
    offset = local.utcoffset() or timedelta(0)
    return f"{local:%d/%m/%Y} às {local:%H:%M:%S} ({_offset_label(offset)})"
