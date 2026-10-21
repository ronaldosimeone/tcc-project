"""
Pydantic v2 DTOs para POST /maintenance/suggest (RF-22 / RNF-46).

MaintenanceSuggestionRequest — probabilidade de falha (RF-22, threshold
estrito > 0.7) + dados mínimos para identificar o equipamento/sintoma e
compor a consulta ao MCP (search_maintenance_manual, RF-21).

MaintenanceSuggestionResponse — se o threshold não for ultrapassado,
``triggered=False`` e nenhuma chamada a MCP/Ollama foi feita (`message`
explica o motivo). Se ultrapassado, ``triggered=True`` com o plano em
Markdown e as referências (arquivo/página/score) dos trechos usados.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MetroPTSensor = Literal[
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "Oil_level",
]


class SensorContribution(BaseModel):
    """RF-35 — contribuição de um sensor físico para a probabilidade de falha
    (SHAP agregado por sensor, ver apps/ml/src/explainability.py). Positivo =
    empurra para falha."""

    sensor: MetroPTSensor
    contribution: float = Field(ge=-100.0, le=100.0)


class MaintenanceSuggestionRequest(BaseModel):
    """Entrada do endpoint de sugestão automática de manutenção."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "failure_probability": 0.87,
                "equipment_name": "Compressor de ar industrial CX-500",
                "predicted_class": 1,
                "symptom_description": "Vazamento de óleo e ruído excessivo na sucção",
            }
        }
    )

    failure_probability: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Probabilidade de falha do modelo preditivo (mesma escala de "
            "PredictResponse.failure_probability). Sugestão só é acionada "
            "quando > 0.7 — RF-22."
        ),
    )
    decision_threshold: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description=(
            "Limiar calibrado do modelo que gerou a probabilidade (card). "
            "Quando informado (eventos de falha abertos pelo pipeline), a "
            "sugestão segue a decisão do modelo: aciona com probabilidade >= "
            "limiar. Ausente (pedido manual): regra RF-22, > 0.7."
        ),
    )
    equipment_name: str = Field(
        default="Compressor de ar industrial",
        min_length=1,
        max_length=200,
        description="Nome/identificação do equipamento — usado para compor a consulta ao MCP.",
    )
    predicted_class: int | None = Field(
        default=None,
        description="Classe predita pelo modelo (0=normal, 1=falha), se disponível. Meramente informativo.",
    )
    symptom_description: str | None = Field(
        default=None,
        max_length=500,
        description=(
            'Descrição livre do sintoma/falha observada (ex.: "vazamento de '
            'óleo"). Combinada com `equipment_name` para formar a query '
            "semântica enviada ao MCP — quanto mais específica, melhor a "
            "recuperação (RF-21)."
        ),
    )
    sensor_contributions: list[SensorContribution] | None = Field(
        default=None,
        max_length=12,
        description=(
            "RF-35 — explicação XAI do alerta, por sensor. Os sensores com "
            "contribuição positiva entram na consulta ao MCP (vocabulário do "
            "componente físico, PT + EN) e todos entram no prompt como evidência."
        ),
    )


class CitedReference(BaseModel):
    """Uma citação `arquivo.pdf, página N` encontrada no texto gerado."""

    file_name: str
    page: int


GroundingStatus = Literal["pending", "grounded", "partially_grounded", "ungrounded"]


class GroundingReport(BaseModel):
    """RN-04 — quão fundamentado nos manuais está o plano gerado."""

    status: GroundingStatus = Field(
        description=(
            "grounded: há contexto ≥ τ e toda citação aponta para um trecho "
            "recuperado; partially_grounded: há contexto, mas o texto não cita "
            "nada ou cita arquivo/página não recuperado; ungrounded: nenhum "
            "trecho atingiu τ."
        )
    )
    threshold: float = Field(
        description="τ — score mínimo (cosine) para um trecho entrar no prompt."
    )
    policy: Literal["block", "flag"]
    max_score: float | None = None
    contexts_used: int = 0
    contexts_discarded: int = 0
    llm_called: bool = True
    citations: list[CitedReference] = Field(default_factory=list)
    invalid_citations: list[CitedReference] = Field(
        default_factory=list,
        description="Citações a arquivo/página que NÃO foram recuperados — referência fabricada.",
    )
    injection_chunks_dropped: int = Field(
        default=0,
        description="RNF-84 — trechos descartados por conterem instruções dirigidas ao modelo.",
    )
    safety_violations: list[str] = Field(
        default_factory=list,
        description="RNF-84 — trechos do plano que mandavam desativar/contornar dispositivo de segurança (plano bloqueado).",
    )


class ManualReference(BaseModel):
    """Uma fonte citada no plano — preserva os metadados do chunk (RF-20/RF-21),
    não recriados aqui."""

    file_name: str
    page: int
    chunk_index: int
    source: str
    score: float = Field(ge=-1.0, le=1.0, description="Cosine similarity (RF-21)")


class MaintenanceSuggestionResponse(BaseModel):
    """Saída de POST /maintenance/suggest."""

    triggered: bool = Field(
        description="True se a probabilidade excedeu 0.7 e um plano foi gerado."
    )
    failure_probability: float = Field(ge=0.0, le=1.0)
    markdown: str | None = Field(
        default=None,
        description="Plano de manutenção em Markdown, fundamentado nos manuais. None quando triggered=False.",
    )
    references: list[ManualReference] = Field(
        default_factory=list,
        description="Trechos de manual usados como contexto (RF-21) — vazio quando triggered=False ou quando o MCP não retornou resultados.",
    )
    model: str | None = Field(
        default=None,
        description="Modelo Ollama usado para gerar o plano (RNF-46) — None quando triggered=False.",
    )
    message: str | None = Field(
        default=None,
        description="Explica por que a sugestão não foi acionada — presente só quando triggered=False.",
    )
    grounding: GroundingReport | None = Field(
        default=None,
        description="RN-04 — fundamentação do plano. None quando triggered=False.",
    )


# ---------------------------------------------------------------------------
# RF-38 — feedback do técnico (human-in-the-loop)
# ---------------------------------------------------------------------------


class ReferenceVote(BaseModel):
    """Voto do técnico sobre um trecho citado: era relevante para a falha?"""

    file_name: str = Field(min_length=1, max_length=255)
    page: int = Field(ge=0)
    helpful: bool


class SuggestionFeedbackCreate(BaseModel):
    """Entrada de POST /v1/maintenance/feedback."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "equipment_name": "Compressor de ar da APU",
                "symptom_description": "dessecante das torres contaminado",
                "query": "Compressor de ar da APU: dessecante das torres contaminado",
                "model": "llama3.2:3b",
                "grounding_status": "grounded",
                "useful": True,
                "reference_votes": [
                    {
                        "file_name": "wap5-scheduled-maintenance-vol-b.pdf",
                        "page": 174,
                        "helpful": True,
                    }
                ],
                "comment": "Procedimento correto, faltou o intervalo de troca.",
            }
        }
    )

    equipment_name: str = Field(min_length=1, max_length=200)
    symptom_description: str | None = Field(default=None, max_length=500)
    query: str | None = Field(default=None, max_length=1000)
    model: str | None = Field(default=None, max_length=100)
    grounding_status: GroundingStatus | None = None
    useful: bool
    reference_votes: list[ReferenceVote] = Field(default_factory=list, max_length=20)
    comment: str | None = Field(default=None, max_length=1000)


class SuggestionFeedbackResponse(BaseModel):
    id: int
    created_at: datetime


class ReferencePrior(BaseModel):
    file_name: str
    page: int
    helpful_votes: int
    unhelpful_votes: int
    prior: float = Field(
        description="(helpful − unhelpful) / (helpful + unhelpful + 2) ∈ (−1, 1)"
    )
