"""
TelegramAlertRateLimiter — RF-24 / RNF-48.

1 alerta crítico por equipamento a cada 15 minutos (900 s), com aquisição
atômica — decisão de infraestrutura documentada em
`src/models/telegram_alert_lock.py` (Postgres, não Redis: o projeto não
tinha Redis, e o Postgres já usado por `predictions` resolve a mesma
garantia de atomicidade sem infraestrutura nova).

Atomicidade: `INSERT ... ON CONFLICT (equipment_id) DO UPDATE ... WHERE
telegram_alert_locks.expires_at <= :now RETURNING equipment_id` — equivalente
a `SET NX EX 900` do Redis. Duas transações concorrentes tentando adquirir a
mesma chave serializam no nível de linha do Postgres/SQLite (UNIQUE +
upsert); só uma recebe a linha de volta em `RETURNING` — a outra não
(nenhuma linha retornada = não adquiriu).

Cada operação abre e fecha sua PRÓPRIA sessão (`session_factory`) — não
depende do ciclo de vida da sessão do chamador (rota HTTP ou o loop
contínuo de `InferencePipelineService`), garantindo que o commit da
aquisição aconteça imediatamente, isolado de qualquer outra operação de
banco em andamento.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.core.database import AsyncSessionFactory
from src.models.telegram_alert_lock import TelegramAlertLock

log = structlog.get_logger(__name__)


class TelegramAlertRateLimiter:
    """Rate limit por `equipment_id`, TTL configurável (default 900s)."""

    def __init__(
        self,
        ttl_seconds: int = 900,
        session_factory: async_sessionmaker[AsyncSession] = AsyncSessionFactory,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._session_factory = session_factory
        # Injetável só para testes de TTL determinísticos (RF-24 §12) — sem
        # isso, testar "t=14m59s ainda bloqueado / t=15m liberado" exigiria
        # esperar 15 minutos de verdade ou mockar `datetime` globalmente.
        self._clock = clock

    async def try_acquire(self, equipment_id: str) -> bool:
        """
        Tenta adquirir a janela de 15 min para `equipment_id`.

        Retorna `True` só quando este chamador é quem efetivamente adquiriu
        a janela (nenhum alerta enviado para esse equipamento nos últimos
        `ttl_seconds`, ou a janela anterior já expirou). `False` = já existe
        uma janela ativa — não enviar (rate limited).
        """
        now = self._clock()
        expires_at = now + timedelta(seconds=self._ttl_seconds)

        async with self._session_factory() as db:
            dialect_name = db.get_bind().dialect.name
            insert_fn = pg_insert if dialect_name == "postgresql" else sqlite_insert

            # `insert_fn` é `postgresql.insert` OU `sqlite.insert` (decidido
            # em runtime acima) — mypy só enxerga o tipo base `Insert` da
            # ternária e não expõe `on_conflict_do_update` (método específico
            # de cada dialect, não do `Insert` genérico do SQLAlchemy core).
            # Ambos os dialects aceitam a mesma assinatura usada aqui
            # (`index_elements`/`set_`/`where`) — comportamento idêntico nos
            # dois bancos, só o tipo estático não reflete isso.
            stmt = (
                insert_fn(TelegramAlertLock)
                .values(equipment_id=equipment_id, expires_at=expires_at)
                .on_conflict_do_update(  # type: ignore[attr-defined]
                    index_elements=[TelegramAlertLock.equipment_id],
                    set_={"expires_at": expires_at},
                    where=TelegramAlertLock.expires_at <= now,
                )
                .returning(TelegramAlertLock.equipment_id)
            )
            result = await db.execute(stmt)
            acquired = result.first() is not None
            await db.commit()

        log.debug(
            "telegram_rate_limit_check", equipment_id=equipment_id, acquired=acquired
        )
        return acquired

    async def release(self, equipment_id: str) -> None:
        """
        Libera a janela de `equipment_id` — chamado quando o envio ao
        Telegram falha (RF-24 §5): a falha não pode consumir a janela de
        rate limit, senão um retry legítimo ficaria bloqueado por 15 min
        por causa de uma falha transitória do Telegram.
        """
        async with self._session_factory() as db:
            await db.execute(
                delete(TelegramAlertLock).where(
                    TelegramAlertLock.equipment_id == equipment_id
                )
            )
            await db.commit()
        log.debug("telegram_rate_limit_released", equipment_id=equipment_id)
