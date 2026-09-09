"""
SQLAlchemy ORM model for a alert_settings — RF-25 / RNF-49.

Configuração GLOBAL (single-tenant) do limiar de alerta crítico — não há
sistema de usuários no projeto (auditado explicitamente antes desta task:
nenhuma tabela `users`, nenhum login, só o token de admin compartilhado
`X-Admin-Token` do RF-11), então esta tabela armazena exatamente UMA linha
para o sistema inteiro, não uma configuração por usuário.

Singleton garantido em nível de banco (não só de aplicação): `id` é travado
em `1` por uma CHECK constraint — qualquer tentativa de inserir uma segunda
linha com `id` diferente de `1` falha na constraint, e a chave primária por
si só impede duas linhas com `id=1`. Combinado com o UPSERT atômico feito
pelo `AlertSettingsService` (mesmo padrão de
`services/telegram_alert_rate_limiter.py`), isso garante "no máximo uma
configuração global" sem depender de nenhuma lógica adicional em Python.

O intervalo `0.5 <= alert_threshold <= 0.95` (RF-25) também é reforçado por
CHECK constraint — a validação do Pydantic (`schemas/alert_settings.py`) é a
primeira linha de defesa, mas o banco nunca aceita um valor fora do range
mesmo que algum caminho de código futuro pule a validação da API.

`telegram_enabled`/`email_enabled`/`alert_email` (extensão desta mesma task):
liga/desliga cada canal de notificação e o endereço de e-mail de destino.
Um CHECK constraint reforça, também no banco, a mesma regra do Pydantic:
`email_enabled=true` exige `alert_email` preenchido.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.core.database import Base


class AlertSettings(Base):
    """Configuração global (singleton) de alertas — limiar + canais — RF-25."""

    __tablename__ = "alert_settings"
    __table_args__ = (
        CheckConstraint("id = 1", name="ck_alert_settings_singleton"),
        CheckConstraint(
            "alert_threshold >= 0.5 AND alert_threshold <= 0.95",
            name="ck_alert_settings_threshold_range",
        ),
        CheckConstraint(
            "(NOT email_enabled) OR (alert_email IS NOT NULL)",
            name="ck_alert_settings_email_required_when_enabled",
        ),
    )

    # Sempre 1 — não autoincrement (o singleton É a chave primária fixa).
    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=False, default=1
    )
    alert_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    # Default True — preserva o comportamento do RF-24 (Telegram sempre
    # tentado) para quem nunca abriu /settings/alerts.
    telegram_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    # Default False — não existia canal de e-mail antes desta task; ativar
    # é uma decisão explícita do operador, nunca um comportamento silencioso.
    email_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    alert_email: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
