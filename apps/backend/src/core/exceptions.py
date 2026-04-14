"""
Custom domain exceptions and global FastAPI exception handlers.

All service-layer errors should raise one of these typed exceptions.
The handlers registered in main.py convert them to standardised JSON responses.
"""

from __future__ import annotations

from fastapi import Request, status
from fastapi.responses import JSONResponse


class AppError(Exception):
    """Base class for all application-level errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail: str = "An unexpected error occurred."

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or self.__class__.detail
        super().__init__(self.detail)


class NotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    detail = "Resource not found."


class ConflictError(AppError):
    status_code = status.HTTP_409_CONFLICT
    detail = "Resource already exists."


class UnauthorizedError(AppError):
    status_code = status.HTTP_401_UNAUTHORIZED
    detail = "Authentication required."


class ForbiddenError(AppError):
    status_code = status.HTTP_403_FORBIDDEN
    detail = "Insufficient permissions."


# ---------------------------------------------------------------------------
# Global handler – registered in main.py
# ---------------------------------------------------------------------------


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Convert any AppError subclass into a standardised JSON error response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.__class__.__name__, "detail": exc.detail},
    )
