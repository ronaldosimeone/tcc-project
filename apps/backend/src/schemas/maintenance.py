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

from pydantic import BaseModel, ConfigDict, Field


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
