"""AuthService — login and user management (RBAC)."""

from __future__ import annotations

from datetime import datetime
from typing import cast

import structlog
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import ConflictError, UnauthorizedError
from src.core.security import Role, create_access_token, hash_password, verify_password
from src.models.user import User

log = structlog.get_logger(__name__)


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime
    username: str
    role: Role


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[a-zA-Z0-9_.-]+$")
    full_name: str | None = Field(default=None, max_length=200)
    password: str = Field(min_length=12, max_length=256)
    role: Role


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    full_name: str | None
    role: Role
    is_active: bool


class AuthService:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def login(self, username: str, password: str) -> TokenOut:
        user = await self._db.scalar(select(User).where(User.username == username))
        # Same error for unknown user / wrong password / inactive (no user enumeration).
        if (
            user is None
            or not user.is_active
            or not verify_password(password, user.password_hash)
        ):
            log.info("auth_login_failed", username=username)
            raise UnauthorizedError("Invalid credentials.")
        token, expires = create_access_token(user.username, user.role)
        log.info("auth_login_ok", username=username, role=user.role)
        return TokenOut(
            access_token=token,
            expires_at=expires,
            username=user.username,
            role=cast(Role, user.role),
        )

    async def create_user(self, payload: UserCreate) -> UserOut:
        if await self._db.scalar(
            select(User.id).where(User.username == payload.username)
        ):
            raise ConflictError(f"User '{payload.username}' already exists.")
        user = User(
            username=payload.username,
            full_name=payload.full_name,
            password_hash=hash_password(payload.password),
            role=payload.role,
            is_active=True,
        )
        self._db.add(user)
        await self._db.flush()
        return UserOut.model_validate(user)

    async def list_users(self) -> list[UserOut]:
        rows = (await self._db.scalars(select(User).order_by(User.username))).all()
        return [UserOut.model_validate(u) for u in rows]

    async def bootstrap_admin(self, username: str, password: str) -> bool:
        """Create the first admin only when the users table is empty."""
        if await self._db.scalar(select(func.count()).select_from(User)):
            return False
        await self.create_user(
            UserCreate(username=username, password=password, role="admin")
        )
        await self._db.commit()
        log.info("auth_bootstrap_admin_created", username=username)
        return True
