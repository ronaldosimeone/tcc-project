"""
Rate limiting — slowapi (RNF-19).

The Limiter singleton is attached to app.state in main.py.  Protected
routes use the @limiter.limit() decorator; the key function isolates
buckets by client IP address.

Deployment notes
----------------
Behind a reverse proxy (Nginx, Traefik) the real client IP is in the
X-Forwarded-For header.  Configure the proxy to set a trusted
X-Real-IP header and switch the key_func to:

    from slowapi.util import get_ipaddr
    limiter = Limiter(key_func=get_ipaddr, ...)

This keeps the rate limiting meaningful even when all requests appear
to originate from the loopback address.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ── Singleton — imported by main.py and predict.py ───────────────────────

limiter: Limiter = Limiter(
    key_func=get_remote_address,
    # No global default: limits are declared per-route via decorator so
    # that different endpoints can have different budgets.
    default_limits=[],
)

# ── Per-route limit strings ───────────────────────────────────────────────

#: Applied to POST /predict/ (RNF-19).
PREDICT_RATE_LIMIT: str = "100/minute"

# ── Custom 429 handler ────────────────────────────────────────────────────


async def rate_limit_exceeded_handler(
    request: Request,
    exc: RateLimitExceeded,
) -> JSONResponse:
    """
    Return a machine-readable JSON 429 when a client exceeds its budget.

    The response body never exposes internal limit configuration — only a
    safe, generic message and a Retry-After header so clients can back off
    gracefully.
    """
    retry_after: int = getattr(exc, "retry_after", 60)
    return JSONResponse(
        status_code=429,
        content={
            "error": "RateLimitExceeded",
            "detail": "Too many requests. Please slow down and retry later.",
        },
        headers={"Retry-After": str(retry_after)},
    )
