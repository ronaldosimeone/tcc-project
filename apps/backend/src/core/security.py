"""
Authentication and role-based access control (IEC 62443 FR1/FR2).

* Passwords: argon2id (``argon2-cffi`` defaults: memory-hard, salted).
* Tokens: JWT HS256 with ``sub`` (username), ``role`` and ``exp``.
* Roles are ordered: operator < technician < engineer < admin. A route
  declares the MINIMUM role it needs with ``require_role(...)``.

``AUTH_ENABLED=false`` (default, development) keeps the previous open
behaviour so the existing frontend and tests work unchanged — but the
request is NOT silently attributed to a real user: the actor is
``anonymous (auth disabled)`` and that is what the audit trail records.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from src.core.config import settings

Role = Literal["operator", "technician", "engineer", "admin"]
ROLE_LEVEL: dict[str, int] = {"operator": 1, "technician": 2, "engineer": 3, "admin": 4}
DEV_JWT_SECRET = "dev-only-jwt-secret-change-me"
ANONYMOUS = "anonymous (auth disabled)"

_hasher = PasswordHasher()
_bearer = OAuth2PasswordBearer(tokenUrl="/v1/auth/login", auto_error=False)


@dataclass(frozen=True)
class Actor:
    username: str
    role: str
    authenticated: bool


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, InvalidHashError):
        return False


def ensure_secure_config() -> None:
    """Refuse to start with auth on and the public placeholder secret."""
    if settings.auth_enabled and settings.jwt_secret == DEV_JWT_SECRET:
        raise RuntimeError(
            "AUTH_ENABLED=true requires a real JWT_SECRET (the dev placeholder is public)."
        )


def create_access_token(
    username: str, role: str, now: datetime | None = None
) -> tuple[str, datetime]:
    issued = now or datetime.now(timezone.utc)
    expires = issued + timedelta(minutes=settings.jwt_expire_minutes)
    token = jwt.encode(
        {"sub": username, "role": role, "iat": issued, "exp": expires},
        settings.jwt_secret,
        algorithm="HS256",
    )
    return token, expires


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def current_actor(token: str | None = Depends(_bearer)) -> Actor:
    if not settings.auth_enabled:
        return Actor(username=ANONYMOUS, role="admin", authenticated=False)
    if not token:
        raise _unauthorized("Authentication required.")
    try:
        claims = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError as exc:
        raise _unauthorized("Token expired.") from exc
    except jwt.InvalidTokenError as exc:
        raise _unauthorized("Invalid token.") from exc
    role = str(claims.get("role", ""))
    if role not in ROLE_LEVEL or not claims.get("sub"):
        raise _unauthorized("Invalid token claims.")
    return Actor(username=str(claims["sub"]), role=role, authenticated=True)


def require_role(minimum: Role) -> Callable[..., Awaitable[Actor]]:
    """Dependency factory: the caller must hold ``minimum`` or a higher role."""

    async def _dep(actor: Actor = Depends(current_actor)) -> Actor:
        if ROLE_LEVEL[actor.role] < ROLE_LEVEL[minimum]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{actor.role}' cannot perform this action (requires '{minimum}').",
            )
        return actor

    return _dep
