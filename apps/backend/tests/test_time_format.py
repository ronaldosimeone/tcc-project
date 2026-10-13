"""Formatação de horário das notificações (Telegram/e-mail)."""

from __future__ import annotations

import pytest

from src.core.time_format import format_local_timestamp


def test_utc_iso_is_converted_to_configured_zone() -> None:
    out = format_local_timestamp(
        "2026-10-13T15:26:46.857123+00:00", tz_name="America/Sao_Paulo"
    )
    assert out == "13/10/2026 às 12:26:46 (UTC-3)"


def test_z_suffix_and_naive_datetime_are_treated_as_utc() -> None:
    expected = "13/10/2026 às 12:00:00 (UTC-3)"
    assert (
        format_local_timestamp("2026-10-13T15:00:00Z", tz_name="America/Sao_Paulo")
        == expected
    )
    assert (
        format_local_timestamp("2026-10-13T15:00:00", tz_name="America/Sao_Paulo")
        == expected
    )


def test_date_rolls_over_across_midnight() -> None:
    out = format_local_timestamp(
        "2026-10-14T01:30:00+00:00", tz_name="America/Sao_Paulo"
    )
    assert out == "13/10/2026 às 22:30:00 (UTC-3)"


def test_half_hour_offset_is_rendered() -> None:
    out = format_local_timestamp("2026-10-13T00:00:00+00:00", tz_name="Asia/Kolkata")
    assert out == "13/10/2026 às 05:30:00 (UTC+5:30)"


def test_invalid_zone_falls_back_to_utc() -> None:
    out = format_local_timestamp("2026-10-13T15:00:00+00:00", tz_name="Mars/Olympus")
    assert out == "13/10/2026 às 15:00:00 (UTC+0)"


@pytest.mark.parametrize("raw", ["t", "ontem às 3", ""])
def test_non_iso_text_passes_through_unchanged(raw: str) -> None:
    assert format_local_timestamp(raw, tz_name="America/Sao_Paulo") == raw


def test_default_zone_comes_from_settings() -> None:
    out = format_local_timestamp("2026-10-13T15:00:00+00:00")
    assert out.endswith("(UTC-3)")  # default: America/Sao_Paulo
