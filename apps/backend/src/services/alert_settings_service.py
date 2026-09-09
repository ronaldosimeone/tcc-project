"""
AlertSettingsService — RF-25 / RNF-49.

Configuração GLOBAL (single-tenant) de alertas — limiar de falha crítica +
canais de notificação (Telegram/e-mail) — persistida em `alert_settings`
(única linha, `id = 1`). Consumida por `CriticalFailureNotificationService`
(RF-24) no lugar dos valores fixos anteriores sempre que existir uma
configuração salva — ver `critical_failure_notification_service.py`.

`get_settings()` é deliberadamente NÃO idempotente-com-efeito-colateral:
uma requisição `GET` nunca cria a linha no banco (mantém a semântica REST de
leitura sem efeito colateral) — se nenhuma configuração foi salva ainda,
devolve os defaults do RF-24 (threshold fixo, Telegram ligado, e-mail
desligado — ver `AlertSettingsSnapshot` abaixo), preservando o comportamento
anterior ao RF-25. A linha só passa a existir na primeira chamada a
`upsert_settings` (primeiro `PUT /v1/settings/alerts`).

`upsert_settings()` reaproveita o MESMO padrão atômico de
`telegram_alert_rate_limiter.py` (`INSERT ... ON CONFLICT ... RETURNING`,
dialect-aware Postgres/SQLite) — aqui sem cláusula `WHERE` (não há TTL,
qualquer PUT deve sempre sobrescrever a configuração anterior). Como `id`
está travado em `1` por CHECK constraint, o `ON CONFLICT (id)` nunca cria
uma segunda linha — há sempre, no máximo, uma configuração global.

`get_threshold()`/`upsert_threshold()` são mantidos por compatibilidade com
o código/testes do RF-24 (que só conheciam o limiar) — implementados em
termos de `get_settings()`/`upsert_settings()` acima, preservando os demais
campos (canais/e-mail) intactos quando só o limiar é alterado por esse
caminho legado.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import structlog
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.core.database import AsyncSessionFactory
from src.models.alert_settings import AlertSettings

log = structlog.get_logger(__name__)

# ID fixo da única linha permitida — mesmo valor do CHECK constraint
# `ck_alert_settings_singleton` (models/alert_settings.py).
_SINGLETON_ID = 1


@dataclass(frozen=True)
class AlertSettingsSnapshot:
    """Configuração global efetiva — persistida, ou os defaults do RF-24
    (`telegram_enabled=True`, `email_enabled=False`) quando nunca salva."""

    alert_threshold: float
    telegram_enabled: bool
    email_enabled: bool
    alert_email: str | None


class AlertSettingsService:
    """Leitura/escrita da configuração global (singleton) de alerta."""

    def __init__(
        self,
        default_threshold: float,
        session_factory: async_sessionmaker[AsyncSession] = AsyncSessionFactory,
    ) -> None:
        # Injetado pelo chamador (RF-24's CRITICAL_FAILURE_THRESHOLD) — este
        # módulo não importa `critical_failure_notification_service` para
        # evitar import circular (aquele módulo importa este).
        self._default_threshold = default_threshold
        self._session_factory = session_factory

    async def get_settings(self) -> AlertSettingsSnapshot:
        """Lê a configuração efetiva — defaults do RF-24 se nunca salva."""
        async with self._session_factory() as db:
            row = (
                await db.execute(
                    select(AlertSettings).where(AlertSettings.id == _SINGLETON_ID)
                )
            ).scalar_one_or_none()

        if row is None:
            return AlertSettingsSnapshot(
                alert_threshold=self._default_threshold,
                telegram_enabled=True,
                email_enabled=False,
                alert_email=None,
            )
        return AlertSettingsSnapshot(
            alert_threshold=row.alert_threshold,
            telegram_enabled=row.telegram_enabled,
            email_enabled=row.email_enabled,
            alert_email=row.alert_email,
        )

    async def get_threshold(self) -> float:
        """Compatibilidade RF-24 — só o limiar efetivo."""
        return (await self.get_settings()).alert_threshold

    async def upsert_settings(
        self,
        *,
        alert_threshold: float,
        telegram_enabled: bool,
        email_enabled: bool,
        alert_email: str | None,
    ) -> AlertSettingsSnapshot:
        """
        Salva (cria ou atualiza) a configuração global — atômico, sempre
        no máximo uma linha. Retorna a configuração efetivamente persistida.
        """
        now = datetime.now(timezone.utc)

        async with self._session_factory() as db:
            dialect_name = db.get_bind().dialect.name
            insert_fn = pg_insert if dialect_name == "postgresql" else sqlite_insert

            values = {
                "id": _SINGLETON_ID,
                "alert_threshold": alert_threshold,
                "telegram_enabled": telegram_enabled,
                "email_enabled": email_enabled,
                "alert_email": alert_email,
                "created_at": now,
                "updated_at": now,
            }
            # Mesmo comentário de `telegram_alert_rate_limiter.py`: mypy só
            # enxerga o tipo base `Insert` da ternária acima, que não expõe
            # `on_conflict_do_update` (método específico de cada dialect).
            stmt = (
                insert_fn(AlertSettings)
                .values(**values)
                .on_conflict_do_update(  # type: ignore[attr-defined]
                    index_elements=[AlertSettings.id],
                    set_={
                        "alert_threshold": alert_threshold,
                        "telegram_enabled": telegram_enabled,
                        "email_enabled": email_enabled,
                        "alert_email": alert_email,
                        "updated_at": now,
                    },
                )
                .returning(AlertSettings)
            )
            result = await db.execute(stmt)
            saved = result.scalar_one()
            await db.commit()

        snapshot = AlertSettingsSnapshot(
            alert_threshold=float(saved.alert_threshold),
            telegram_enabled=bool(saved.telegram_enabled),
            email_enabled=bool(saved.email_enabled),
            alert_email=saved.alert_email,
        )
        log.info(
            "alert_settings_updated",
            alert_threshold=snapshot.alert_threshold,
            telegram_enabled=snapshot.telegram_enabled,
            email_enabled=snapshot.email_enabled,
            # E-mail em si não é logado — só se está ou não configurado.
            alert_email_configured=snapshot.alert_email is not None,
        )
        return snapshot

    async def upsert_threshold(self, value: float) -> float:
        """Compatibilidade RF-24 — altera só o limiar, preserva os demais
        campos (canais/e-mail) já configurados."""
        current = await self.get_settings()
        updated = await self.upsert_settings(
            alert_threshold=value,
            telegram_enabled=current.telegram_enabled,
            email_enabled=current.email_enabled,
            alert_email=current.alert_email,
        )
        return updated.alert_threshold
