"""
Grounding policy for maintenance suggestions — RN-04 (and RNF-82 evidence).

Two checks, both deterministic (no second LLM as judge):

1. **Before generation** — only manual chunks with cosine score ≥ τ may reach
   the prompt. If none survives, the suggestion is *ungrounded*: with policy
   ``"block"`` the LLM is never called and a fixed "Limitações" plan is
   returned; with ``"flag"`` the LLM is called (it is told there is no
   context) and the response is marked as ungrounded.

2. **After generation** — every ``file.pdf … página N`` citation in the
   answer must point to a chunk that was actually retrieved. A citation to a
   file/page the model never saw is a fabricated reference: the answer is
   downgraded to ``partially_grounded`` and the invalid citations are listed.

τ is a configuration value, not a constant: its default (0.6) mirrors the
MCP server's own cut-off, and RF-36 calibrates it on the annotated benchmark.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal, Protocol, TypeVar

from src.schemas.maintenance import CitedReference, GroundingReport

UngroundedPolicy = Literal["block", "flag"]

BLOCKED_MARKDOWN: str = (
    "# Plano de Manutenção\n\n"
    "## Limitações\n"
    "Nenhum trecho de manual recuperado atingiu o limiar mínimo de relevância "
    "(RN-04). Para evitar um procedimento não fundamentado, nenhum plano foi "
    "gerado. Consulte o manual do fabricante ou um técnico responsável.\n"
)

# "manual-x.pdf`, página 3" | "manual-x.pdf, p. 3" | "manual-x.pdf (page 3)"
_CITATION_RE = re.compile(
    r"([\w\-.]+\.pdf)[`'\"\)\]]*[\s,;:(\-–]*(?:p[áa]gina|page|p[áa]g\.?|p\.)\s*(\d+)",
    re.IGNORECASE,
)


class _ScoredChunk(Protocol):
    file_name: str
    page: int
    score: float


ChunkT = TypeVar("ChunkT", bound=_ScoredChunk)


def split_by_threshold(
    contexts: Sequence[ChunkT], threshold: float
) -> tuple[list[ChunkT], list[ChunkT]]:
    """``(kept, discarded)`` — kept chunks have ``score >= threshold``."""
    kept = [c for c in contexts if c.score >= threshold]
    discarded = [c for c in contexts if c.score < threshold]
    return kept, discarded


def extract_citations(markdown: str) -> list[CitedReference]:
    """Unique ``(file_name, page)`` pairs cited in the answer, in order."""
    seen: set[tuple[str, int]] = set()
    out: list[CitedReference] = []
    for match in _CITATION_RE.finditer(markdown):
        key = (match.group(1), int(match.group(2)))
        if key not in seen:
            seen.add(key)
            out.append(CitedReference(file_name=key[0], page=key[1]))
    return out


def pre_generation_report(
    contexts_kept: Sequence[_ScoredChunk],
    contexts_discarded: Sequence[_ScoredChunk],
    threshold: float,
    policy: UngroundedPolicy,
) -> GroundingReport:
    all_scores = [c.score for c in [*contexts_kept, *contexts_discarded]]
    return GroundingReport(
        status="pending" if contexts_kept else "ungrounded",
        threshold=threshold,
        policy=policy,
        max_score=max(all_scores) if all_scores else None,
        contexts_used=len(contexts_kept),
        contexts_discarded=len(contexts_discarded),
        llm_called=bool(contexts_kept) or policy == "flag",
    )


def finalize_report(
    report: GroundingReport, markdown: str, contexts_kept: Sequence[_ScoredChunk]
) -> GroundingReport:
    """Post-generation citation check. Ungrounded stays ungrounded."""
    retrieved = {(c.file_name, c.page) for c in contexts_kept}
    cited = extract_citations(markdown)
    invalid = [c for c in cited if (c.file_name, c.page) not in retrieved]

    if report.status == "ungrounded":
        status = "ungrounded"
    elif not cited or invalid:
        status = "partially_grounded"
    else:
        status = "grounded"

    return report.model_copy(
        update={"status": status, "citations": cited, "invalid_citations": invalid}
    )
