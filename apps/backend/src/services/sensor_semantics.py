"""
Sensor -> physical component vocabulary — RF-35 (XAI -> MCP -> RAG).

The predictive model explains an alert in terms of MetroPT-3 sensor names
(``Oil_temperature``, ``Towers``, ``DV_pressure``...). Manuals never use
those names: they talk about "compressor oil", "twin tower air drier",
"main reservoir". This module is the bridge: it turns the top positive
SHAP contributions into a bilingual (PT + EN) retrieval query, so the MCP
semantic search looks for the component that the model actually blames.

Bilingual on purpose: the corpus-curation benchmark
(apps/mcp-server/CORPUS-CURATION.md) showed Portuguese queries losing
against English manuals with the multilingual MiniLM embedder.

Sensor meanings follow the MetroPT-3 data descriptor (Veloso et al., 2022,
Scientific Data 9:764).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.schemas.maintenance import SensorContribution


@dataclass(frozen=True)
class SensorMeaning:
    pt: str
    en: str


SENSOR_MEANINGS: dict[str, SensorMeaning] = {
    "TP2": SensorMeaning(
        "pressão de saída do compressor", "compressor delivery pressure"
    ),
    "TP3": SensorMeaning("pressão do painel pneumático", "pneumatic panel pressure"),
    "H1": SensorMeaning(
        "queda de pressão no filtro separador ciclônico",
        "cyclonic separator filter pressure drop",
    ),
    "DV_pressure": SensorMeaning(
        "pressão de descarga das torres do secador de ar",
        "air drier tower discharge pressure",
    ),
    "Reservoirs": SensorMeaning(
        "pressão do reservatório principal", "main reservoir pressure"
    ),
    "Oil_temperature": SensorMeaning(
        "temperatura do óleo do compressor", "compressor oil temperature"
    ),
    "Motor_current": SensorMeaning(
        "corrente do motor do compressor", "compressor motor current"
    ),
    "COMP": SensorMeaning(
        "válvula de admissão de ar do compressor",
        "compressor air intake valve unloader",
    ),
    "DV_eletric": SensorMeaning(
        "válvula de saída do compressor", "compressor outlet valve"
    ),
    "Towers": SensorMeaning(
        "torres do secador de ar e dessecante", "twin tower air drier desiccant"
    ),
    "MPG": SensorMeaning(
        "pressostato de partida do compressor", "compressor pressure switch start"
    ),
    "Oil_level": SensorMeaning("nível de óleo do compressor", "compressor oil level"),
}

# Only the strongest evidence goes into the query: more terms dilute the
# embedding and pull unrelated chunks.
MAX_QUERY_SENSORS: int = 3


def top_positive(
    contributions: list[SensorContribution], k: int = MAX_QUERY_SENSORS
) -> list[SensorContribution]:
    """Sensors that push TOWARDS failure, strongest first."""
    positive = [c for c in contributions if c.contribution > 0]
    return sorted(positive, key=lambda c: c.contribution, reverse=True)[:k]


def xai_query_suffix(contributions: list[SensorContribution]) -> str:
    """``"sinais: <pt> (<en>); ..."`` for the top positive sensors, or ``""``."""
    top = top_positive(contributions)
    if not top:
        return ""
    parts = [
        f"{SENSOR_MEANINGS[c.sensor].pt} ({SENSOR_MEANINGS[c.sensor].en})" for c in top
    ]
    return "sinais: " + "; ".join(parts)


def xai_prompt_lines(contributions: list[SensorContribution]) -> list[str]:
    """Human-readable evidence for the user prompt — every sensor sent,
    sorted by absolute contribution, with its direction."""
    ordered = sorted(contributions, key=lambda c: abs(c.contribution), reverse=True)
    lines: list[str] = []
    for c in ordered:
        direction = "aumenta" if c.contribution > 0 else "reduz"
        lines.append(
            f"- {c.sensor} — {SENSOR_MEANINGS[c.sensor].pt}: {direction} o risco "
            f"(contribuição {c.contribution:+.3f})"
        )
    return lines
