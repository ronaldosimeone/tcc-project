"""
Integration tests for POST /predict.

Strategy
--------
The ModelService singleton is injected via `get_model_service`.  Tests override
that dependency with a lightweight mock that returns deterministic outputs —
no .joblib file or trained model is required to run this suite.

Since RF-09 made POST /predict also persist to the DB, get_db is now
overridden with an in-memory SQLite session so no live PostgreSQL is needed.

The lifespan still executes (it handles FileNotFoundError gracefully), so the
test verifies the full request/response cycle through the real router and
Pydantic validation layers.

Fixtures
--------
mock_service       — A ModelService wrapping a sklearn DummyClassifier.
app_with_mock_model — FastAPI instance with both model and DB overridden.
async_client       — httpx.AsyncClient bound to the test app via ASGITransport.
"""

from __future__ import annotations

from datetime import datetime
from typing import AsyncGenerator
from unittest.mock import MagicMock

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.database import Base, get_db
from src.main import create_app
from src.models.prediction import Prediction  # noqa: F401 — registers with Base
from src.schemas.predict import PredictRequest
from src.services.model_service import ModelService, get_model_service

# ---------------------------------------------------------------------------
# Module-level in-memory SQLite (shared across all tests in this module)
# ---------------------------------------------------------------------------

_TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
_test_engine: AsyncEngine = create_async_engine(_TEST_DB_URL, echo=False)
_TestSessionFactory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=_test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
    """SQLite-backed async session for test isolation (no PostgreSQL required)."""
    async with _TestSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@pytest_asyncio.fixture(autouse=True, scope="module", loop_scope="module")
async def _setup_db_schema() -> AsyncGenerator[None, None]:
    """Create the predictions table once for the entire test module."""
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_PAYLOAD: dict = {
    "TP2": 5.02,
    "TP3": 9.21,
    "H1": 8.97,
    "DV_pressure": 2.10,
    "Reservoirs": 8.85,
    "Oil_temperature": 72.3,
    "Motor_current": 4.5,
    "COMP": 1.0,
    "DV_eletric": 0.0,
    "Towers": 1.0,
    "MPG": 1.0,
    "Pressure_switch": 1.0,
    "Oil_level": 1.0,
    "Caudal_impulses": 1.0,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mock_service() -> ModelService:
    """
    ModelService backed by a MagicMock that simulates a trained classifier.

    predict()      → class 1  (fault detected)
    predict_proba() → [[0.15, 0.85]]
    """
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.15, 0.85]])
    return ModelService(model=mock_model)


@pytest.fixture()
def app_with_mock_model(mock_service: ModelService) -> object:
    """FastAPI test app with both model and DB dependencies overridden."""
    application = create_app()
    application.dependency_overrides[get_model_service] = lambda: mock_service
    application.dependency_overrides[get_db] = _override_get_db
    return application


@pytest_asyncio.fixture()
async def async_client(app_with_mock_model) -> AsyncClient:
    """Async HTTP client bound to the test app."""
    transport = ASGITransport(app=app_with_mock_model)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_returns_200(async_client: AsyncClient) -> None:
    """POST /predict with a valid payload must return HTTP 200."""
    response = await async_client.post("/predict/", json=_VALID_PAYLOAD)
    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_predict_response_has_required_fields(async_client: AsyncClient) -> None:
    """[RF-05] Response must contain predicted_class, failure_probability, timestamp."""
    response = await async_client.post("/predict/", json=_VALID_PAYLOAD)
    body: dict = response.json()

    assert "predicted_class" in body, "Missing 'predicted_class'"
    assert "failure_probability" in body, "Missing 'failure_probability'"
    assert "timestamp" in body, "Missing 'timestamp'"


@pytest.mark.asyncio
async def test_predict_class_is_binary_int(async_client: AsyncClient) -> None:
    """predicted_class must be an integer with value 0 or 1."""
    body = (await async_client.post("/predict/", json=_VALID_PAYLOAD)).json()

    assert isinstance(body["predicted_class"], int)
    assert body["predicted_class"] in {0, 1}


@pytest.mark.asyncio
async def test_predict_probability_in_unit_range(async_client: AsyncClient) -> None:
    """failure_probability must be a float in [0.0, 1.0]."""
    body = (await async_client.post("/predict/", json=_VALID_PAYLOAD)).json()

    prob: float = body["failure_probability"]
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


@pytest.mark.asyncio
async def test_predict_timestamp_is_iso8601(async_client: AsyncClient) -> None:
    """timestamp must be parseable as a valid ISO 8601 datetime string."""
    body = (await async_client.post("/predict/", json=_VALID_PAYLOAD)).json()

    ts: str = body["timestamp"]
    assert isinstance(ts, str)
    # Raises ValueError if format is invalid
    parsed = datetime.fromisoformat(ts)
    assert parsed.tzinfo is not None, "Timestamp must include timezone info (UTC)."


@pytest.mark.asyncio
async def test_predict_mock_returns_fault_class(async_client: AsyncClient) -> None:
    """Mock is configured to return class 1 — verify the pipeline carries it through."""
    body = (await async_client.post("/predict/", json=_VALID_PAYLOAD)).json()
    assert body["predicted_class"] == 1


@pytest.mark.asyncio
async def test_predict_mock_failure_probability_matches(
    async_client: AsyncClient,
) -> None:
    """Mock returns probability 0.85 for class 1 — verify round-trip precision."""
    body = (await async_client.post("/predict/", json=_VALID_PAYLOAD)).json()
    assert abs(body["failure_probability"] - 0.85) < 1e-4


# ---------------------------------------------------------------------------
# Input validation (Pydantic layer)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_missing_field_returns_422(async_client: AsyncClient) -> None:
    """Omitting a required sensor field must return HTTP 422 Unprocessable Entity."""
    incomplete = {k: v for k, v in _VALID_PAYLOAD.items() if k != "TP2"}
    response = await async_client.post("/predict/", json=incomplete)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_non_numeric_value_returns_422(async_client: AsyncClient) -> None:
    """Passing a string where a float is expected must return HTTP 422."""
    bad_payload = {**_VALID_PAYLOAD, "TP2": "not-a-number"}
    response = await async_client.post("/predict/", json=bad_payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_digital_sensor_out_of_range_returns_422(
    async_client: AsyncClient,
) -> None:
    """Digital sensor columns have ge=0 le=1 constraint — value 2.0 must fail."""
    bad_payload = {**_VALID_PAYLOAD, "COMP": 2.0}
    response = await async_client.post("/predict/", json=bad_payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_empty_body_returns_422(async_client: AsyncClient) -> None:
    """An empty JSON body must return HTTP 422."""
    response = await async_client.post("/predict/", json={})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Model not available (HTTP 503 path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_returns_503_when_model_not_loaded() -> None:
    """
    When get_model_service raises ModelNotAvailableError (model_service=None),
    the endpoint must return HTTP 503.
    """
    from src.core.exceptions import ModelNotAvailableError

    app_no_model = create_app()
    app_no_model.dependency_overrides[get_model_service] = lambda: (
        _ for _ in ()
    ).throw(ModelNotAvailableError())

    transport = ASGITransport(app=app_no_model)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/predict/", json=_VALID_PAYLOAD)

    assert response.status_code == 503
    body: dict = response.json()
    assert "detail" in body or "error" in body


# ---------------------------------------------------------------------------
# ModelService unit tests (no HTTP layer)
# ---------------------------------------------------------------------------


class TestModelServiceUnit:
    """Direct unit tests for ModelService.predict — bypass FastAPI entirely."""

    def _make_service(self, predicted_class: int, probability: float) -> ModelService:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([predicted_class])
        mock_model.predict_proba.return_value = np.array(
            [[1.0 - probability, probability]]
        )
        return ModelService(model=mock_model)

    def test_predict_returns_predict_response(self) -> None:
        from src.schemas.predict import PredictResponse

        service = self._make_service(0, 0.12)
        result = service.predict(PredictRequest(**_VALID_PAYLOAD))
        assert isinstance(result, PredictResponse)

    def test_predict_class_zero_propagates(self) -> None:
        service = self._make_service(0, 0.12)
        result = service.predict(PredictRequest(**_VALID_PAYLOAD))
        assert result.predicted_class == 0

    def test_predict_class_one_propagates(self) -> None:
        service = self._make_service(1, 0.91)
        result = service.predict(PredictRequest(**_VALID_PAYLOAD))
        assert result.predicted_class == 1

    def test_failure_probability_rounded(self) -> None:
        service = self._make_service(1, 0.912345678)
        result = service.predict(PredictRequest(**_VALID_PAYLOAD))
        # Must not overflow float precision (6 decimal places)
        assert result.failure_probability == pytest.approx(0.912346, abs=1e-5)

    def test_feature_matrix_has_correct_shape(self) -> None:
        """_build_feature_row must produce a (1, 38) DataFrame (V2).

        Feature count breakdown:
          12  raw sensor inputs
           1  TP2_delta
          21  rolling features (std_5, ma_5, ma_15) × 7 analogue sensors
           4  V2 cross-sensor (TP2_TP3_diff, TP2_TP3_ratio, work_per_pressure,
              reservoir_drop)
          ─────────────────
          38  total
        """
        service = self._make_service(0, 0.1)
        req = PredictRequest(**_VALID_PAYLOAD)
        df = service._build_feature_row(req)
        assert df.shape == (1, 38), f"Expected (1, 38), got {df.shape}"

    def test_feature_matrix_has_no_nulls(self) -> None:
        service = self._make_service(0, 0.1)
        req = PredictRequest(**_VALID_PAYLOAD)
        df = service._build_feature_row(req)
        assert df.isnull().sum().sum() == 0

    def test_delta_is_zero_for_single_row(self) -> None:
        service = self._make_service(0, 0.1)
        req = PredictRequest(**_VALID_PAYLOAD)
        df = service._build_feature_row(req)
        assert df["TP2_delta"].iloc[0] == pytest.approx(0.0)

    def test_rolling_std_is_zero_for_single_row(self) -> None:
        service = self._make_service(0, 0.1)
        req = PredictRequest(**_VALID_PAYLOAD)
        df = service._build_feature_row(req)
        assert df["TP2_std_5"].iloc[0] == pytest.approx(0.0)

    def test_ma_equals_raw_value_for_single_row(self) -> None:
        service = self._make_service(0, 0.1)
        req = PredictRequest(**_VALID_PAYLOAD)
        df = service._build_feature_row(req)
        assert df["TP2_ma_5"].iloc[0] == pytest.approx(req.TP2)
        assert df["TP2_ma_15"].iloc[0] == pytest.approx(req.TP2)
