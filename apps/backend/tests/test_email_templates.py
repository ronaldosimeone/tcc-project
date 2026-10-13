"""Templates HTML dos e-mails (RF-25) — escape, escala de risco e fallback."""

from __future__ import annotations

import pytest

from src.services.email_templates import (
    render_critical_failure_email_html,
    render_test_email_html,
)


def _critical(**overrides: object) -> str:
    kwargs: dict[str, object] = {
        "equipment_name": "APU-Trem-042",
        "probability": 0.964999,
        "timestamp": "2026-10-13T15:14:44.242000+00:00",
        "dashboard_url": "http://localhost/sensors/APU-Trem-042",
    }
    kwargs.update(overrides)
    return render_critical_failure_email_html(**kwargs)  # type: ignore[arg-type]


def test_critical_email_contains_core_facts() -> None:
    html = _critical()

    assert html.startswith("<!DOCTYPE html>")
    assert "APU-Trem-042" in html
    assert ">96<" in html  # probabilidade arredondada, sem casas decimais
    assert "CRÍTICO" in html
    assert "13/10/2026 às 12:14:44 (UTC-3)" in html
    assert 'href="http://localhost/sensors/APU-Trem-042"' in html


def test_critical_email_escapes_equipment_name() -> None:
    html = _critical(equipment_name='<script>alert("x")</script>')

    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


@pytest.mark.parametrize(
    "url", ["javascript:alert(1)", "data:text/html,boom", "ftp://x/y", ""]
)
def test_critical_email_rejects_non_http_links(url: str) -> None:
    html = _critical(dashboard_url=url)

    assert 'href="#"' in html
    assert 'href="javascript:' not in html


@pytest.mark.parametrize(
    ("probability", "label"),
    [(0.10, "NORMAL"), (0.45, "ALERTA"), (0.65, "CRÍTICO"), (0.99, "CRÍTICO")],
)
def test_risk_label_follows_dashboard_scale(probability: float, label: str) -> None:
    assert label in _critical(probability=probability)


@pytest.mark.parametrize("probability", [0.0, 1.0, 1.7, -0.2])
def test_probability_bar_handles_extremes(probability: float) -> None:
    html = _critical(probability=probability)  # não deve lançar

    assert 'width="0%"' not in html  # largura nunca zero (some a barra)
    assert 'width="-' not in html


def test_unparseable_timestamp_passes_through_escaped() -> None:
    html = _critical(timestamp="ontem <b>às 3</b>")

    assert "ontem &lt;b&gt;às 3&lt;/b&gt;" in html


def test_test_email_is_never_a_failure_email() -> None:
    html = render_test_email_html()

    assert "falha crítica" not in html.lower()
    assert "Teste de notificação" in html
    assert "não há falha real" in html
