"""
AuditService — append-only, hash-chained audit trail.

Every row stores ``prev_hash`` (hash of the previous row, or 64 zeros for the
first) and ``hash = SHA-256(prev_hash || canonical JSON of the row)``. Editing
or deleting any row changes a hash that the next row depends on, so
``verify`` pinpoints the first broken link. On PostgreSQL a trigger also
rejects UPDATE/DELETE (migration 0007).

Appends are serialised with a PostgreSQL transaction-level advisory lock so
two concurrent requests can never chain onto the same previous hash.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.security import Actor
from src.models.user import AuditLog

GENESIS = "0" * 64
_ADVISORY_KEY = 74_201_337  # arbitrary, project-wide constant


class AuditEntryOut(BaseModel):
    id: int
    created_at: datetime
    actor: str
    role: str
    action: str
    resource_type: str
    resource_id: str | None
    details: dict[str, Any]
    hash: str


class ChainVerification(BaseModel):
    valid: bool
    checked: int
    first_broken_id: int | None


def _digest(
    prev_hash: str,
    created_at: datetime,
    actor: str,
    role: str,
    action: str,
    resource_type: str,
    resource_id: str | None,
    details: dict[str, Any],
) -> str:
    ts = (
        created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc)
    ).isoformat()
    payload = json.dumps(
        [prev_hash, ts, actor, role, action, resource_type, resource_id, details],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AuditService:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def record(
        self,
        actor: Actor,
        action: str,
        resource_type: str,
        resource_id: str | int | None = None,
        details: dict[str, Any] | None = None,
    ) -> AuditLog:
        """Append one entry. The caller commits (same transaction as the action)."""
        if self._db.bind is not None and self._db.bind.dialect.name == "postgresql":
            await self._db.execute(
                text("SELECT pg_advisory_xact_lock(:k)"), {"k": _ADVISORY_KEY}
            )
        prev = await self._db.scalar(
            select(AuditLog.hash).order_by(AuditLog.id.desc()).limit(1)
        )
        prev_hash = prev or GENESIS
        created_at = datetime.now(timezone.utc)
        rid = None if resource_id is None else str(resource_id)
        det = details or {}
        entry = AuditLog(
            created_at=created_at,
            actor=actor.username,
            role=actor.role,
            action=action,
            resource_type=resource_type,
            resource_id=rid,
            details=det,
            prev_hash=prev_hash,
            hash=_digest(
                prev_hash,
                created_at,
                actor.username,
                actor.role,
                action,
                resource_type,
                rid,
                det,
            ),
        )
        self._db.add(entry)
        await self._db.flush()
        return entry

    async def list(
        self, limit: int = 100, actor: str | None = None, action: str | None = None
    ) -> list[AuditEntryOut]:
        q = select(AuditLog).order_by(AuditLog.id.desc()).limit(limit)
        if actor:
            q = q.where(AuditLog.actor == actor)
        if action:
            q = q.where(AuditLog.action == action)
        rows = (await self._db.scalars(q)).all()
        return [
            AuditEntryOut(
                id=r.id,
                created_at=r.created_at,
                actor=r.actor,
                role=r.role,
                action=r.action,
                resource_type=r.resource_type,
                resource_id=r.resource_id,
                details=r.details,
                hash=r.hash,
            )
            for r in rows
        ]

    async def verify(self) -> ChainVerification:
        prev = GENESIS
        checked = 0
        for row in (
            await self._db.scalars(select(AuditLog).order_by(AuditLog.id))
        ).all():
            expected = _digest(
                prev,
                row.created_at,
                row.actor,
                row.role,
                row.action,
                row.resource_type,
                row.resource_id,
                row.details,
            )
            if row.prev_hash != prev or row.hash != expected:
                return ChainVerification(
                    valid=False, checked=checked, first_broken_id=row.id
                )
            prev = row.hash
            checked += 1
        return ChainVerification(valid=True, checked=checked, first_broken_id=None)
