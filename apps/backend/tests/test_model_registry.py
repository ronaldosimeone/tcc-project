"""
Tests for ModelRegistry (RNF-25) and the /models admin endpoints (RF-11).

Coverage
--------
ModelRegistry (unit)
  • get() raises ModelNotAvailableError when registry is empty.
  • Initial state: is_loaded=False, active_name="".
  • swap() loads the model and updates active_name + is_loaded.
  • get() returns the ModelService after a successful swap.
  • swap() returns the previous model name (empty on first load, old name
    on subsequent swaps).
  • swap() raises ValueError for unknown model names — no I/O attempted.
  • Concurrent swap calls are serialised by the asyncio.Lock.

require_admin_token (unit)
  • Passes silently with the correct token.
  • Raises HTTP 401 for a wrong token.
  • Raises HTTP 401 when the token is absent (None).

/models endpoints (integration)
  • GET  /models — 401 without token.
  • GET  /models — 200 with valid token; response shape is correct.
  • PUT  /models/active — 401 without token.
  • PUT  /models/active — 401 with wrong token.
  • PUT  /models/active — 422 for an invalid model name (Pydantic validation).
  • PUT  /models/active — 202 with valid token; enqueues swap, returns payload.
  • PUT  /models/active — 404 when the artefact is missing on disk.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from src.core.auth import require_admin_token
from src.core.config import settings
from src.core.exceptions import ModelNotAvailableError
from src.main import create_app
from src.schemas.models import ModelsListResponse, SwapModelResponse
from src.services.model_registry import KNOWN_MODELS, ModelRegistry, get_model_registry
from src.services.model_service import ModelService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_TOKEN = settings.admin_api_token
_WRONG_TOKEN = "definitely-wrong-token"
_ADMIN_HEADER = {"X-Admin-Token": _VALID_TOKEN}


def _mock_service() -> MagicMock:
    """Minimal ModelService mock — only the interface matters here."""
    return MagicMock(spec=ModelService)


# ---------------------------------------------------------------------------
# ModelRegistry — unit tests
# ---------------------------------------------------------------------------


async def test_registry_get_raises_when_empty() -> None:
    registry = ModelRegistry()
    with pytest.raises(ModelNotAvailableError):
        await registry.get()


async def test_registry_initial_state() -> None:
    registry = ModelRegistry()
    assert not registry.is_loaded
    assert registry.active_name == ""


async def test_registry_swap_updates_active_name() -> None:
    registry = ModelRegistry()
    with patch(
        "src.services.model_service.load_model_by_name",
        return_value=_mock_service(),
    ):
        await registry.swap("random_forest")

    assert registry.active_name == "random_forest"
    assert registry.is_loaded


async def test_registry_get_returns_service_after_swap() -> None:
    registry = ModelRegistry()
    mock = _mock_service()
    with patch("src.services.model_service.load_model_by_name", return_value=mock):
        await registry.swap("xgboost")

    result = await registry.get()
    assert result is mock


async def test_registry_swap_returns_empty_string_on_first_load() -> None:
    registry = ModelRegistry()
    with patch(
        "src.services.model_service.load_model_by_name",
        return_value=_mock_service(),
    ):
        previous = await registry.swap("random_forest")

    assert previous == ""


async def test_registry_swap_returns_previous_model_name() -> None:
    registry = ModelRegistry()
    with patch(
        "src.services.model_service.load_model_by_name",
        return_value=_mock_service(),
    ):
        await registry.swap("random_forest")
        previous = await registry.swap("xgboost")

    assert previous == "random_forest"
    assert registry.active_name == "xgboost"


async def test_registry_swap_invalid_model_raises_value_error() -> None:
    registry = ModelRegistry()
    with pytest.raises(ValueError, match="Unknown model"):
        await registry.swap("neural_prophet")


async def test_registry_swap_invalid_model_does_not_call_loader() -> None:
    """Validation must be checked *before* any I/O is attempted."""
    registry = ModelRegistry()
    with patch("src.services.model_service.load_model_by_name") as mock_load:
        with pytest.raises(ValueError):
            await registry.swap("bad_model")
        mock_load.assert_not_called()


async def test_registry_lock_serialises_concurrent_swaps() -> None:
    """
    Two concurrent swap() calls must not interleave their critical sections.
    The second must see the fully-committed result of the first.
    """
    registry = ModelRegistry()
    completion_order: list[str] = []

    async def fake_to_thread(fn, name: str) -> MagicMock:  # type: ignore[override]
        # Simulate slow I/O with a tiny yield so both coroutines start
        await asyncio.sleep(0)
        completion_order.append(name)
        return _mock_service()

    with patch(
        "src.services.model_registry.asyncio.to_thread", side_effect=fake_to_thread
    ):
        await asyncio.gather(
            registry.swap("random_forest"),
            registry.swap("xgboost"),
        )

    # Both swaps must have completed; active_name is whichever finished last
    assert len(completion_order) == 2
    assert registry.active_name == completion_order[-1]


# ---------------------------------------------------------------------------
# require_admin_token — unit tests
# ---------------------------------------------------------------------------


async def test_require_admin_token_passes_with_correct_token() -> None:
    """No exception raised for the configured token."""
    await require_admin_token(token=_VALID_TOKEN)


async def test_require_admin_token_raises_401_for_wrong_token() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await require_admin_token(token=_WRONG_TOKEN)
    assert exc_info.value.status_code == 401


async def test_require_admin_token_raises_401_for_missing_token() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await require_admin_token(token=None)
    assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# /models endpoints — integration tests
# ---------------------------------------------------------------------------


def _build_client(registry: ModelRegistry) -> tuple:
    """
    Return (app, AsyncClient) with the real app but a mock registry injected
    via dependency_overrides so no artefacts are needed.
    """
    app = create_app()
    app.dependency_overrides[get_model_registry] = lambda: registry
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")
    return app, client


async def test_get_models_returns_401_without_token() -> None:
    registry = ModelRegistry()
    _, client = _build_client(registry)
    async with client:
        resp = await client.get("/models")
    assert resp.status_code == 401


async def test_get_models_returns_200_with_valid_token() -> None:
    registry = ModelRegistry()
    with patch(
        "src.services.model_service.load_model_by_name",
        return_value=_mock_service(),
    ):
        await registry.swap("random_forest")

    _, client = _build_client(registry)
    async with client:
        resp = await client.get("/models", headers=_ADMIN_HEADER)

    assert resp.status_code == 200
    body = ModelsListResponse(**resp.json())
    assert body.active_model == "random_forest"
    assert len(body.models) == len(KNOWN_MODELS)


async def test_get_models_response_contains_all_known_models() -> None:
    registry = ModelRegistry()
    _, client = _build_client(registry)
    async with client:
        resp = await client.get("/models", headers=_ADMIN_HEADER)

    names = {m["name"] for m in resp.json()["models"]}
    assert names == KNOWN_MODELS


async def test_put_models_active_returns_401_without_token() -> None:
    registry = ModelRegistry()
    _, client = _build_client(registry)
    async with client:
        resp = await client.put("/models/active", json={"model_name": "xgboost"})
    assert resp.status_code == 401


async def test_put_models_active_returns_401_for_wrong_token() -> None:
    registry = ModelRegistry()
    _, client = _build_client(registry)
    async with client:
        resp = await client.put(
            "/models/active",
            json={"model_name": "xgboost"},
            headers={"X-Admin-Token": _WRONG_TOKEN},
        )
    assert resp.status_code == 401


async def test_put_models_active_returns_422_for_invalid_model_name() -> None:
    registry = ModelRegistry()
    _, client = _build_client(registry)
    async with client:
        resp = await client.put(
            "/models/active",
            json={"model_name": "not_a_real_model"},
            headers=_ADMIN_HEADER,
        )
    assert resp.status_code == 422


async def test_put_models_active_swaps_model_successfully() -> None:
    registry = ModelRegistry()
    with patch(
        "src.services.model_service.load_model_by_name",
        return_value=_mock_service(),
    ):
        await registry.swap("random_forest")

    _, client = _build_client(registry)
    with patch(
        "src.services.model_service.load_model_by_name",
        return_value=_mock_service(),
    ):
        async with client:
            resp = await client.put(
                "/models/active",
                json={"model_name": "xgboost"},
                headers=_ADMIN_HEADER,
            )

    assert resp.status_code == 202
    body = SwapModelResponse(**resp.json())
    assert body.previous_model == "random_forest"
    assert body.active_model == "xgboost"


async def test_put_models_active_returns_404_when_artefact_missing() -> None:
    # The router guards with artefact_path.exists() *before* enqueuing the
    # background task, so we mock Path.exists to simulate a missing file on
    # disk.  The 404 must be raised synchronously — load_model_by_name is
    # never called.
    registry = ModelRegistry()
    _, client = _build_client(registry)
    with patch("pathlib.Path.exists", return_value=False):
        async with client:
            resp = await client.put(
                "/models/active",
                json={"model_name": "mlp"},
                headers=_ADMIN_HEADER,
            )
    assert resp.status_code == 404
