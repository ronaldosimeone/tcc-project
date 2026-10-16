"""
Auth router — login, current user, user management and audit trail.

I/O only; rules in services/auth_service.py and services/audit_service.py.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.core.security import Actor, current_actor, require_role
from src.services.audit_service import AuditEntryOut, AuditService, ChainVerification
from src.services.auth_service import AuthService, TokenOut, UserCreate, UserOut

router: APIRouter = APIRouter(prefix="/v1", tags=["Auth"])


@router.post(
    "/auth/login", response_model=TokenOut, summary="Login (OAuth2 password) → JWT"
)
async def login(
    form: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)
) -> TokenOut:
    return await AuthService(db).login(form.username, form.password)


@router.get(
    "/auth/me", response_model=dict[str, str | bool], summary="Usuário atual e papel"
)
async def me(actor: Actor = Depends(current_actor)) -> dict[str, str | bool]:
    return {
        "username": actor.username,
        "role": actor.role,
        "authenticated": actor.authenticated,
    }


@router.post(
    "/auth/users",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="Cria usuário (admin)",
)
async def create_user(
    payload: UserCreate,
    actor: Actor = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
) -> UserOut:
    user = await AuthService(db).create_user(payload)
    await AuditService(db).record(
        actor,
        "user.create",
        "user",
        user.id,
        {"username": user.username, "role": user.role},
    )
    await db.commit()
    return user


@router.get(
    "/auth/users", response_model=list[UserOut], summary="Lista usuários (admin)"
)
async def list_users(
    _: Actor = Depends(require_role("admin")), db: AsyncSession = Depends(get_db)
) -> list[UserOut]:
    return await AuthService(db).list_users()


@router.get(
    "/audit",
    response_model=list[AuditEntryOut],
    summary="Trilha de auditoria (engenheiro+)",
)
async def list_audit(
    limit: int = Query(default=100, ge=1, le=1000),
    actor_filter: str | None = Query(default=None, alias="actor", max_length=100),
    action: str | None = Query(default=None, max_length=64),
    _: Actor = Depends(require_role("engineer")),
    db: AsyncSession = Depends(get_db),
) -> list[AuditEntryOut]:
    return await AuditService(db).list(limit, actor_filter, action)


@router.get(
    "/audit/verify",
    response_model=ChainVerification,
    summary="Verifica a cadeia de hashes (admin)",
)
async def verify_audit(
    _: Actor = Depends(require_role("admin")), db: AsyncSession = Depends(get_db)
) -> ChainVerification:
    return await AuditService(db).verify()
