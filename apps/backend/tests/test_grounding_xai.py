"""
RF-35 (XAI -> MCP -> RAG) e RN-04 (fundamentação) — MaintenanceSuggestionService.

Garantias verificadas
---------------------
1.  Sem `sensor_contributions`, a query é idêntica à anterior (retrocompatível).
2.  Com contribuições, só os top-3 sensores POSITIVOS entram na query, em PT + EN.
3.  Toda contribuição (positiva e negativa) aparece no prompt como evidência,
    e a evidência nunca vai para o System Prompt.
4.  Trechos com score < τ nunca chegam ao prompt.
5.  Política "block": nenhum trecho >= τ -> Ollama NÃO é chamado, plano fixo de
    Limitações, status `ungrounded` (suggest e stream).
6.  Citação a arquivo/página não recuperado -> `partially_grounded` +
    `invalid_citations`; citação correta -> `grounded`.
7.  Sensor fora do vocabulário MetroPT-3 é rejeitado na validação do DTO.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from src.schemas.maintenance import MaintenanceSuggestionRequest, SensorContribution
from src.services.grounding import BLOCKED_MARKDOWN, extract_citations
from src.services.maintenance_suggestion_service import (
    SYSTEM_PROMPT,
    MaintenanceSuggestionService,
)


def _result(*chunks: tuple[str, int, float]) -> dict[str, Any]:
    return {
        "query": "q",
        "results": [
            {
                "text": f"Trecho de {name} p{page}",
                "score": score,
                "metadata": {
                    "file_name": name,
                    "source": name,
                    "page": page,
                    "chunk_index": 0,
                },
            }
            for name, page, score in chunks
        ],
    }


def _service(mcp_result: dict[str, Any], markdown: str, **kwargs: Any):
    mcp = AsyncMock()
    mcp.search_maintenance_manual = AsyncMock(return_value=mcp_result)
    ollama = AsyncMock()
    ollama.generate = AsyncMock(return_value=markdown)
    return (
        MaintenanceSuggestionService(
            mcp_client=mcp, ollama_client=ollama, model="m", **kwargs
        ),
        mcp,
        ollama,
    )


def _req(
    contribs: list[tuple[str, float]] | None = None,
) -> MaintenanceSuggestionRequest:
    return MaintenanceSuggestionRequest(
        failure_probability=0.9,
        equipment_name="Compressor de ar",
        symptom_description="queda de pressão",
        sensor_contributions=(
            [SensorContribution(sensor=s, contribution=c) for s, c in contribs]
            if contribs is not None
            else None
        ),
    )


GOOD_MD = "# Plano\n\n## Referências\n- `wap5.pdf`, página 174\n"


# ---------------------------------------------------------------------------
# RF-35 — query e prompt
# ---------------------------------------------------------------------------


def test_query_unchanged_without_contributions() -> None:
    assert (
        MaintenanceSuggestionService._build_query(_req())
        == "Compressor de ar: queda de pressão"
    )  # noqa: SLF001


def test_query_uses_top3_positive_sensors_bilingual() -> None:
    query = MaintenanceSuggestionService._build_query(  # noqa: SLF001
        _req(
            [
                ("Towers", 0.40),
                ("Oil_temperature", 0.30),
                ("Reservoirs", -0.50),
                ("DV_pressure", 0.20),
                ("MPG", 0.01),
            ]
        )
    )
    assert query == (
        "Compressor de ar: queda de pressão | sinais: "
        "torres do secador de ar e dessecante (twin tower air drier desiccant); "
        "temperatura do óleo do compressor (compressor oil temperature); "
        "pressão de descarga das torres do secador de ar (air drier tower discharge pressure)"
    )
    assert "reservatório" not in query  # contribuição negativa não entra na busca
    assert "pressostato" not in query  # só top-3


def test_query_unchanged_when_all_contributions_negative() -> None:
    q = MaintenanceSuggestionService._build_query(
        _req([("Towers", -0.2)])
    )  # noqa: SLF001
    assert q == "Compressor de ar: queda de pressão"


async def test_prompt_carries_all_evidence_but_system_prompt_does_not() -> None:
    service, _, ollama = _service(_result(("wap5.pdf", 174, 0.8)), GOOD_MD)

    await service.suggest(_req([("Towers", 0.4), ("Reservoirs", -0.1)]))

    system_prompt, user_prompt = ollama.generate.await_args.args
    assert "Evidência dos sensores" in user_prompt
    assert (
        "- Towers — torres do secador de ar e dessecante: aumenta o risco (contribuição +0.400)"
        in user_prompt
    )
    assert (
        "- Reservoirs — pressão do reservatório principal: reduz o risco (contribuição -0.100)"
        in user_prompt
    )
    assert system_prompt == SYSTEM_PROMPT
    assert "Towers" not in system_prompt


def test_unknown_sensor_rejected() -> None:
    with pytest.raises(ValidationError):
        SensorContribution(sensor="Brake_pipe", contribution=0.1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RN-04 — limiar τ e política
# ---------------------------------------------------------------------------


async def test_chunks_below_threshold_never_reach_prompt() -> None:
    service, _, ollama = _service(
        _result(("wap5.pdf", 174, 0.82), ("bomba.pdf", 3, 0.55)),
        GOOD_MD,
        grounding_threshold=0.6,
    )

    resp = await service.suggest(_req())

    _, user_prompt = ollama.generate.await_args.args
    assert "wap5.pdf" in user_prompt and "bomba.pdf" not in user_prompt
    assert [r.file_name for r in resp.references] == ["wap5.pdf"]
    assert resp.grounding is not None
    assert resp.grounding.contexts_used == 1 and resp.grounding.contexts_discarded == 1
    assert resp.grounding.max_score == pytest.approx(0.82)


async def test_block_policy_skips_llm_when_ungrounded() -> None:
    service, _, ollama = _service(
        _result(("bomba.pdf", 3, 0.5)), GOOD_MD, grounding_threshold=0.6
    )

    resp = await service.suggest(_req())

    ollama.generate.assert_not_awaited()
    assert resp.triggered is True
    assert resp.markdown == BLOCKED_MARKDOWN
    assert resp.references == [] and resp.model is None
    assert resp.grounding is not None
    assert resp.grounding.status == "ungrounded" and resp.grounding.llm_called is False


async def test_flag_policy_calls_llm_but_marks_ungrounded() -> None:
    service, _, ollama = _service(_result(), GOOD_MD, ungrounded_policy="flag")

    resp = await service.suggest(_req())

    ollama.generate.assert_awaited_once()
    assert resp.grounding is not None and resp.grounding.status == "ungrounded"
    assert resp.grounding.llm_called is True


async def test_stream_block_policy_emits_done_without_tokens() -> None:
    service, _, _ = _service(_result(), GOOD_MD)
    ollama = AsyncMock()
    ollama.generate_stream = AsyncMock(
        side_effect=AssertionError("LLM must not be called")
    )
    service._ollama = ollama  # noqa: SLF001

    events = [e async for e in service.suggest_stream(_req())]

    assert [e.type for e in events] == ["searching", "done"]
    assert events[-1].markdown == BLOCKED_MARKDOWN
    assert (
        events[-1].grounding is not None and events[-1].grounding.status == "ungrounded"
    )


# ---------------------------------------------------------------------------
# RN-04 — checagem de citações após a geração
# ---------------------------------------------------------------------------


async def test_valid_citation_is_grounded() -> None:
    service, _, _ = _service(_result(("wap5.pdf", 174, 0.8)), GOOD_MD)
    resp = await service.suggest(_req())
    assert resp.grounding is not None and resp.grounding.status == "grounded"
    assert resp.grounding.invalid_citations == []


async def test_fabricated_citation_is_flagged() -> None:
    md = "# Plano\n\n## Referências\n- `wap5.pdf`, página 174\n- `manual-x.pdf`, página 9\n"
    service, _, _ = _service(_result(("wap5.pdf", 174, 0.8)), md)

    resp = await service.suggest(_req())

    assert resp.grounding is not None
    assert resp.grounding.status == "partially_grounded"
    assert [(c.file_name, c.page) for c in resp.grounding.invalid_citations] == [
        ("manual-x.pdf", 9)
    ]


async def test_answer_without_citation_is_partially_grounded() -> None:
    service, _, _ = _service(
        _result(("wap5.pdf", 174, 0.8)), "# Plano\n\nSem referências."
    )
    resp = await service.suggest(_req())
    assert resp.grounding is not None and resp.grounding.status == "partially_grounded"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("- `manual-a.pdf`, página 3", [("manual-a.pdf", 3)]),
        (
            "manual-a.pdf, p. 12 e manual-b.pdf (page 4)",
            [("manual-a.pdf", 12), ("manual-b.pdf", 4)],
        ),
        ("manual-a.pdf, pág. 7; manual-a.pdf, página 7", [("manual-a.pdf", 7)]),
        ("nenhuma citação aqui", []),
    ],
)
def test_extract_citations_formats(text: str, expected: list[tuple[str, int]]) -> None:
    assert [(c.file_name, c.page) for c in extract_citations(text)] == expected
