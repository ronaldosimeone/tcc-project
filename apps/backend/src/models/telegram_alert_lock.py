"""
SQLAlchemy ORM model para o rate limit de notificações críticas — RF-24.

Uma linha por `equipment_id` — `expires_at` é o fim da janela de 15 minutos
(RNF-48). Não é uma tabela de histórico: `TelegramAlertRateLimiter` faz
UPSERT nela (ver src/services/telegram_alert_rate_limiter.py), então ela
nunca cresce além do número de equipamentos distintos.

Por que Postgres em vez de Redis
---------------------------------
O projeto já usa Postgres (via SQLAlchemy async) para `predictions` — não
havia Redis/cache em nenhum lugar do repositório antes desta task (auditado
antes de implementar). Adicionar Redis só para este rate limit exigiria um
novo serviço no docker-compose.yml, volume, healthcheck e mais uma
dependência de infraestrutura — para uma necessidade que o Postgres já
resolve com a mesma garantia de atomicidade pedida pela spec (`SET NX EX`
≈ `INSERT ... ON CONFLICT ... WHERE expires_at <= now() RETURNING ...`,
ver `telegram_alert_rate_limiter.py`). Sobrevive a restart do container e a
múltiplos workers exatamente como o Postgres já faz para `predictions`.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from src.core.database import Base


class TelegramAlertLock(Base):
    """Uma linha = uma janela de rate limit ativa (ou expirada) para um
    `equipment_id`. `expires_at` no passado é equivalente a "chave expirada"
    no modelo Redis — ver `TelegramAlertRateLimiter.try_acquire`."""

    __tablename__ = "telegram_alert_locks"

    equipment_id: Mapped[str] = mapped_column(String, primary_key=True)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
