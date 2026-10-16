"""
Safety guard for retrieved context and generated plans — RNF-84 mitigation.

The red-team benchmark (benchmarks/benchmark_redteam.py) showed that, once a
malicious manual chunk reaches the prompt, small local LLMs follow it despite
system-prompt rule 5: both models recommended disabling the safety valve,
and one obeyed an embedded "reply only with <canary>" instruction. A system
prompt is not a security boundary, so two deterministic barriers are added:

1. ``is_injection`` — BEFORE generation, chunks containing instructions
   addressed to the model (or demanding links / citations be inserted) are
   dropped from the context. Manuals describe equipment; they never address
   "the assistant".

2. ``safety_violations`` — AFTER generation, a plan that tells the technician
   to disable, bypass or defeat a safety device is blocked and replaced by a
   fixed notice. Recommending to *test* or *replace* a safety valve is fine;
   recommending to *disable* it never is.

Patterns are bilingual (PT/EN) because manuals are English and plans are
Portuguese.
"""

from __future__ import annotations

import re

SAFETY_BLOCKED_MARKDOWN: str = (
    "# Plano de Manutenção\n\n"
    "## Limitações\n"
    "O plano gerado continha uma instrução para desativar ou contornar um "
    "dispositivo de segurança e foi bloqueado (RNF-84). Nenhum procedimento "
    "de manutenção deve desabilitar válvulas de segurança, intertravamentos "
    "ou proteções. Consulte o responsável técnico.\n"
)

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above)\s+(rules|instructions)",
        r"ignor[ae]\s+(todas\s+)?(as\s+)?(regras|instru[çc][õo]es)\s+(anteriores|acima)",
        r"(notice|message|instruction)s?\s+to\s+the\s+(assistant|model|ai|llm)",
        r"(aviso|mensagem|instru[çc][ãa]o)\s+(para|ao)\s+(o\s+)?(assistente|modelo)",
        r"\b(reply|respond|answer)\s+only\s+with\b",
        r"\bresponda\s+apenas\s+com\b",
        r"(must|should)\s+(include|contain)\s+(the\s+)?[\w\s-]{0,30}(link|url)",
        r"always\s+cite\s+the\s+document",
    )
)

_SAFETY_DEVICE = (
    r"(v[áa]lvula(s)?\s+de\s+seguran[çc]a|safety\s+valve(s)?|intertravamento|interlock|"
    r"pressostato|pressure\s+switch|prote[çc][ãa]o|guard|dispositivo\s+de\s+seguran[çc]a)"
)
_DEFEAT = (
    r"(desativ|desabilit|desliga\w*\s+a\s+prote|bloque|trav|grampe|jump|bypass|contorn|"
    r"disable|defeat|override|clamp|anul|inibi)"
)
_VIOLATION_RE = re.compile(
    rf"{_DEFEAT}\w*[^.\n]{{0,60}}{_SAFETY_DEVICE}|{_SAFETY_DEVICE}[^.\n]{{0,60}}{_DEFEAT}",
    re.IGNORECASE,
)


def is_injection(text: str) -> bool:
    """True when a retrieved chunk contains an instruction aimed at the model."""
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def safety_violations(markdown: str) -> list[str]:
    """Sentences of the plan that tell the technician to defeat a safety device."""
    return [m.group(0).strip() for m in _VIOLATION_RE.finditer(markdown or "")]
