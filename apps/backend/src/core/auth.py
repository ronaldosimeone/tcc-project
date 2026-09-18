"""
Admin authentication dependency (RF-11).

All `/models` management endpoints require a pre-shared token sent in the
`X-Admin-Token` request header.  The token value is configured via the
`ADMIN_API_TOKEN` environment variable (see `Settings.admin_api_token`).

Dev bypass
----------
When `DEBUG=true` **and** `ADMIN_API_TOKEN` is still the placeholder default
`"change-me-in-production"`, the dependency is a no-op.  This avoids 401 noise
during local development where no real secret is configured.

RNF-62 — the bypass now requires `DEBUG=true` as well: a production/staging
deployment (`DEBUG` unset → `False`) that forgot to set `ADMIN_API_TOKEN`
**fails closed** (all `/models` and `/v1/settings` admin routes return 401)
instead of silently exposing them.  A ZAP scan of the previous behaviour
found `GET /v1/settings/alerts` reachable with no credentials because
`.env.example` shipped the placeholder value.

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

# Sentinel used by Settings.admin_api_token when no real token is configured.
# Matching this value (case-sensitive) explicitly signals "development mode".
_DEV_PLACEHOLDER_TOKEN = "change-me-in-production"


async def require_admin_token(
    token: str | None = Security(_api_key_header),
) -> None:
    """
    FastAPI dependency — raises HTTP 401 if the admin token is absent or wrong.

    Returns immediately (no-op) when the backend is running with the default
    dev placeholder token — local development requires no header.

    Usage
    -----
    Add as a router-level or endpoint-level dependency:

        router = APIRouter(dependencies=[Depends(require_admin_token)])

    or per-endpoint:

        @router.get("/foo", dependencies=[Depends(require_admin_token)])
    """
    # Dev mode: DEBUG=true AND no real secret configured → skip auth entirely.
    # RNF-62: sem o `settings.debug`, um ambiente de produção que esqueceu de
    # trocar o ADMIN_API_TOKEN ficaria com as rotas admin abertas (fail-open).
    if settings.debug and settings.admin_api_token == _DEV_PLACEHOLDER_TOKEN:
        return

    # Guard against None before compare_digest: both arguments must be str
    # (the AnyStr constraint requires identical concrete types).
    if not token:
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
