"""
ModelRegistry — thread-safe, asyncio-native hot-swap manager (RNF-25, RF-11).

Atomicity guarantee
-------------------
Model artefacts can be hundreds of MB.  Loading them from disk takes seconds.
If that blocking I/O happened *inside* an asyncio lock, every in-flight
prediction request would time out while the lock was held.

The registry solves this with a two-phase swap:

  Phase 1 — outside the lock (slow)
      The new ModelService is fully initialised in a background thread via
      `asyncio.to_thread`.  In-flight requests continue using the *old* model
      normally.

  Phase 2 — inside the lock (fast, ~nanoseconds)
      Only the Python reference `self._service = new_service` is executed under
      the lock.  CPython's GIL guarantees that a single reference assignment is
      atomic; readers never observe a torn state.  The lock additionally
      serialises concurrent swap requests so two simultaneous PUT /models/active
      calls queue instead of racing.

Dependency injection
--------------------
`get_model_registry` is exported so that routers can declare it as a FastAPI
`Depends` and tests can override it via `app.dependency_overrides`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from fastapi import Request

from src.core.exceptions import ModelNotAvailableError

if TYPE_CHECKING:
    from src.services.model_service import ModelService

logger: logging.Logger = logging.getLogger(__name__)

# Single source of truth — must stay in sync with the branches in
# model_service.load_model_by_name and the Literal in schemas.models.ModelName.
# Adding a new model name here also requires:
#   1. A branch (or registry entry) in model_service.load_model_by_name.
#   2. The same name in schemas/models.py::ModelName.
KNOWN_MODELS: frozenset[str] = frozenset(
    {
        "random_forest",
        "xgboost",
        "mlp",
        "random_forest_v2",
        "xgboost_v2",
    }
)


class ModelRegistry:
    """
    In-process registry that owns exactly one active `ModelService` at a time
    and provides an atomic hot-swap API.
    """

    def __init__(self) -> None:
        self._lock: asyncio.Lock = asyncio.Lock()
        self._service: ModelService | None = None
        self._active_name: str = ""

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def active_name(self) -> str:
        """Name of the currently active model, or empty string if none loaded."""
        return self._active_name

    @property
    def is_loaded(self) -> bool:
        """True when a model has been successfully loaded."""
        return self._service is not None

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def get(self) -> ModelService:
        """
        Return the active ModelService.

        Raises
        ------
        ModelNotAvailableError
            When no model has been loaded yet (e.g., startup artefact missing).
        """
        service = self._service  # single read — atomic under CPython GIL
        if service is None:
            raise ModelNotAvailableError()
        return service

    async def swap(self, model_name: str) -> str:
        """
        Load *model_name* and atomically replace the active ModelService.

        Parameters
        ----------
        model_name : str
            One of the values in `KNOWN_MODELS`.

        Returns
        -------
        str
            The name of the model that was active before the swap (empty string
            on first load).

        Raises
        ------
        ValueError
            If *model_name* is not in `KNOWN_MODELS`.
        FileNotFoundError
            If the model artefact does not exist on disk.
        """
        model_name = model_name.lower().strip()
        if model_name not in KNOWN_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Valid choices: {sorted(KNOWN_MODELS)}"
            )

        # Deferred import to break the circular dependency:
        # model_registry → model_service (ok)
        # model_service does NOT import model_registry
        from src.services.model_service import load_model_by_name

        logger.info("[RNF-25] Loading model '%s' in background thread …", model_name)

        # ── Phase 1: load outside the lock (blocking I/O in thread pool) ──
        new_service: ModelService = await asyncio.to_thread(
            load_model_by_name, model_name
        )

        # ── Phase 2: atomic pointer swap inside the lock (nanoseconds) ───
        async with self._lock:
            previous = self._active_name
            self._service = new_service
            self._active_name = model_name

        logger.info("[RNF-25] Model swap complete: '%s' → '%s'", previous, model_name)
        return previous


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


def get_model_registry(request: Request) -> ModelRegistry:
    """
    FastAPI dependency — resolves the ModelRegistry from app.state.

    Exported so that routers use it as `Depends(get_model_registry)` and
    tests can replace it via `app.dependency_overrides[get_model_registry]`.
    """
    registry: ModelRegistry | None = getattr(request.app.state, "model_registry", None)
    if registry is None:
        raise RuntimeError(
            "ModelRegistry is not initialised on app.state. "
            "Check the FastAPI lifespan."
        )
    return registry
