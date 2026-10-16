"""
Testes de propriedade (Hypothesis) — invariantes que valem para QUALQUER
entrada, não só para os exemplos escritos à mão.

EventTracker (RN-03)
    * episódios nunca se sobrepõem e saem em ordem;
    * todo episódio tem ≥ k_open amostras e média/pico ≥ limiar ≤ pico ≤ 1;
    * nº de episódios ≤ nº de sequências com ≥ k_open amostras acima do limiar.
Parsers expostos a dados externos (fuzzing)
    * parse_message (MQTT) só falha com ValueError, nunca com outra exceção;
    * extract_citations / is_injection / safety_violations nunca levantam.
Trilha de auditoria
    * alterar qualquer campo de uma entrada muda o hash (tamper-evidence).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from src.services.audit_service import _digest
from src.services.event_tracker import EventTracker
from src.services.grounding import extract_citations
from src.services.mqtt_telemetry import parse_message
from src.services.safety_guard import is_injection, safety_violations

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
probs = st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), max_size=400)
ks = st.integers(min_value=1, max_value=8)


def _runs_at_least(values: list[float], threshold: float, k: int) -> int:
    runs, cur = 0, 0
    for v in values:
        cur = cur + 1 if v >= threshold else 0
        if cur == k:
            runs += 1
    return runs


@settings(max_examples=300, deadline=None)
@given(probs, st.floats(min_value=0.05, max_value=0.95), ks, ks)
def test_event_tracker_invariants(
    values: list[float], threshold: float, k_open: int, k_close: int
) -> None:
    tracker = EventTracker(threshold, k_open, k_close)
    opened, closed = [], []
    for i, p in enumerate(values):
        upd = tracker.update(T0 + timedelta(seconds=10 * i), p)
        if upd.opened:
            opened.append(upd.opened)
        if upd.closed:
            closed.append(upd.closed)

    assert len(closed) <= len(opened) <= len(closed) + 1
    assert len(opened) <= _runs_at_least(values, threshold, k_open)
    for ep in closed:
        assert ep.n >= k_open
        assert threshold <= ep.mean <= ep.peak <= 1.0
        assert ep.started_at <= ep.last_above_at
    for a, b in zip(closed, closed[1:]):
        assert a.last_above_at < b.started_at


@settings(max_examples=300, deadline=None)
@given(st.text(max_size=80), st.binary(max_size=400))
def test_parse_message_only_raises_value_error(topic: str, payload: bytes) -> None:
    try:
        parse_message(topic, payload, "predictiq/fleet")
    except ValueError:
        pass


@settings(max_examples=300, deadline=None)
@given(st.text(max_size=2000))
def test_text_guards_never_raise(text: str) -> None:
    cites = extract_citations(text)
    assert len({(c.file_name, c.page) for c in cites}) == len(cites)
    assert isinstance(is_injection(text), bool)
    assert isinstance(safety_violations(text), list)


@settings(max_examples=200, deadline=None)
@given(
    st.text(min_size=1, max_size=20),
    st.text(min_size=1, max_size=20),
    st.dictionaries(st.text(max_size=5), st.integers(), max_size=3),
    st.sampled_from(["actor", "action", "resource_id", "details"]),
)
def test_audit_hash_changes_on_any_edit(
    actor: str, action: str, details: dict[str, int], field: str
) -> None:
    args = dict(
        prev_hash="0" * 64,
        created_at=T0,
        actor=actor,
        role="admin",
        action=action,
        resource_type="work_order",
        resource_id="7",
        details=details,
    )
    original = _digest(**args)  # type: ignore[arg-type]
    edited = dict(args)
    edited[field] = (
        {**details, "__tampered__": 1} if field == "details" else f"{args[field]}x"
    )
    assert _digest(**edited) != original  # type: ignore[arg-type]
