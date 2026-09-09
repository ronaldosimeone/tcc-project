"""
SQLAlchemy ORM model for the drift_reports table — RF-27 / RNF-55.

Persiste o resultado de cada execução da análise diária de data drift
(`DriftMonitor`, `tasks/drift_tasks.py`) — nunca escrito pelo processo web
(o endpoint GET /monitoring/drift só LÊ este histórico, RF-27 §Fase 7:
"O endpoint não deve disparar a análise").

Idempotência (RF-27 §Fase 8): `analysis_date` é UNIQUE — a task diária faz
um upsert (`ON CONFLICT (analysis_date) DO UPDATE`, mesmo padrão de
`AlertSettings`/`TelegramAlertLock`) chaveado nessa coluna, então executar a
mesma janela duas vezes (erro do scheduler, retry manual) sobrescreve o
relatório daquele dia em vez de duplicar o histórico.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.core.database import Base


class DriftReport(Base):
    """
    Um relatório de data drift (RF-27) — uma linha por dia analisado.

    Columns
    -------
    analysis_date        : Data (UTC) da janela analisada — UNIQUE, chave de
                            idempotência (RF-27 §Fase 8).
    analyzed_at           : Timestamp exato em que a task rodou.
    reference_period      : Descrição textual do baseline usado (dataset,
                             tamanho da amostra, seed — RF-27 §Fase 4).
    current_period_start/end : Janela dos dados "current" (últimas 24h).
    psi                   : PSI agregado (máximo entre as features
                             analisadas) — `NULL` quando `status != "ok"`.
    drift_detected        : `psi > 0.25` (RF-27, estrito) — `NULL` quando
                             `status != "ok"`.
    features              : PSI por feature individual (transparência —
                             não só o agregado). Ex.: {"TP2": 0.03, ...}.
    status                : "ok" | "insufficient_data" | "error" — nunca um
                             PSI inventado quando não há dados suficientes
                             (RF-27 §"IMPORTANTE — SEM DADOS INVENTADOS").
    error_message         : Preenchido apenas quando status == "error".
    """

    __tablename__ = "drift_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    analysis_date: Mapped[date] = mapped_column(Date, nullable=False, unique=True)
    analyzed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    reference_period: Mapped[str] = mapped_column(String, nullable=False)
    current_period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    current_period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    psi: Mapped[float | None] = mapped_column(Float, nullable=True)
    drift_detected: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    features: Mapped[dict[str, float] | None] = mapped_column(JSON, nullable=True)

    status: Mapped[str] = mapped_column(String, nullable=False)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
