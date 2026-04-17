"""
Admin authentication dependency (RF-11).

All `/models` management endpoints require a pre-shared token sent in the
`X-Admin-Token` request header.  The token value is configured via the
`ADMIN_API_TOKEN` environment variable (see `Settings.admin_api_token`).

Design notes
------------
- `APIKeyHeader` integrates with FastAPI's OpenAPI schema: the security
  requirement appears in /docs automatically.
- `secrets.compare_digest` performs a constant-time comparison to prevent
  timing-based token oracle attacks.
- `auto_error=False` lets us return a custom 401 instead of FastAPI's default
  403 for missing API keys.
"""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.core.config import settings

_api_key_header = APIKeyHeader(name="X-Admin-Token", auto_error=False)


async def require_admin_token(
    token: str | None = Security(_api_key_header),
) -> None:
    """
    FastAPI dependency — raises HTTP 401 if the admin token is absent or wrong.

    Usage
    -----
    Add as a router-level or endpoint-level dependency:

        router = APIRouter(dependencies=[Depends(require_admin_token)])

    or per-endpoint:

        @router.get("/foo", dependencies=[Depends(require_admin_token)])
    """
    # Guard against None before compare_digest: both arguments must be str
    # (the AnyStr constraint requires identical concrete types).
    if not token or not settings.admin_api_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing admin token.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    if not secrets.compare_digest(token, settings.admin_api_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing admin token.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
