"""
Métricas Prometheus da segunda rodada (seção 3 de src/core/metrics.py).

1. `record_grounding` conta a decisão RN-04 e os trechos descartados (RNF-84).
2. `record_feedback` separa voto do plano e voto por trecho (RF-38).
3. Todas aparecem em GET /metrics, sem label de alta cardinalidade.
4. O painel Grafana só consulta métricas que existem.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from prometheus_client import Counter

from src.core import metrics
from src.main import create_app

_DASHBOARD = (
    Path(__file__).resolve().parents[3]
    / "infra"
    / "grafana"
    / "dashboards"
    / "predictiq-reliability.json"
)

_NEW = (
    metrics.RAG_GROUNDING_TOTAL,
    metrics.RAG_INJECTION_CHUNKS_DROPPED_TOTAL,
    metrics.RAG_SAFETY_BLOCKS_TOTAL,
    metrics.FAILURE_EVENTS_TOTAL,
    metrics.SHADOW_PREDICTIONS_TOTAL,
    metrics.TELEMETRY_MESSAGES_TOTAL,
    metrics.FEEDBACK_VOTES_TOTAL,
)


def _value(counter: Counter, **labels: str) -> float:
    target = counter.labels(**labels) if labels else counter
    return float(target._value.get())  # noqa: SLF001 — leitura direta em teste


def test_record_grounding_counts_decision_and_dropped_chunks() -> None:
    blocked = _value(metrics.RAG_GROUNDING_TOTAL, llm_called="false")
    dropped = _value(metrics.RAG_INJECTION_CHUNKS_DROPPED_TOTAL)
    metrics.record_grounding(llm_called=False, injection_chunks_dropped=2)
    assert _value(metrics.RAG_GROUNDING_TOTAL, llm_called="false") == blocked + 1
    assert _value(metrics.RAG_INJECTION_CHUNKS_DROPPED_TOTAL) == dropped + 2


def test_record_feedback_splits_plan_and_reference_votes() -> None:
    plan_yes = _value(metrics.FEEDBACK_VOTES_TOTAL, kind="plan", helpful="true")
    ref_no = _value(metrics.FEEDBACK_VOTES_TOTAL, kind="reference", helpful="false")
    metrics.record_feedback(useful=True, reference_votes=[False, False, True])
    assert (
        _value(metrics.FEEDBACK_VOTES_TOTAL, kind="plan", helpful="true")
        == plan_yes + 1
    )
    assert (
        _value(metrics.FEEDBACK_VOTES_TOTAL, kind="reference", helpful="false")
        == ref_no + 2
    )


@pytest.mark.asyncio
async def test_new_metrics_are_exposed() -> None:
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        body = (await client.get("/metrics")).text
    for counter in _NEW:
        assert f"# TYPE {counter._name}_total counter" in body  # noqa: SLF001


def test_no_high_cardinality_labels() -> None:
    allowed = {
        "llm_called",
        "transition",
        "agreement",
        "transport",
        "result",
        "kind",
        "helpful",
    }
    for counter in _NEW:
        assert set(counter._labelnames) <= allowed  # noqa: SLF001


def test_dashboard_queries_only_known_metrics() -> None:
    dash = json.loads(_DASHBOARD.read_text(encoding="utf-8"))
    exprs = [t["expr"] for p in dash["panels"] for t in p.get("targets", [])]
    used = {m for e in exprs for m in re.findall(r"predictiq_[a-z_]+", e)}
    known = {f"{c._name}_total" for c in _NEW}  # noqa: SLF001
    assert used and used <= known
