# Backend — Predictive Maintenance API

FastAPI service that exposes fault-prediction endpoints backed by trained ML models
(Random Forest, XGBoost, or MLP/ONNX) and provides admin endpoints to hot-swap the
active model at runtime without restarting the server.

---

## Setup

```bash
cd apps/backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ACTIVE_MODEL` | `random_forest` | Model loaded at startup. One of `random_forest`, `xgboost`, `mlp`. |
| `MODEL_PATH` | *(auto)* | Absolute path to `random_forest_final.joblib`. |
| `XGBOOST_MODEL_PATH` | *(auto)* | Absolute path to `xgboost_v1.joblib`. |
| `MLP_ONNX_PATH` | *(auto)* | Absolute path to `mlp_v1.onnx`. |
| `MLP_SCALER_PATH` | *(auto)* | Absolute path to `mlp_scaler.joblib`. |
| `ADMIN_API_TOKEN` | `change-me-in-production` | **Must be overridden in production.** Token required for all `/models` admin endpoints. |
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/tcc_db` | Async PostgreSQL DSN. |
| `ALLOWED_ORIGINS` | `["http://localhost:3000"]` | CORS allowed origins (JSON list). |
| `DEBUG` | `false` | Enables verbose structured logging. |

---

## Running

```bash
# Development (auto-reload)
uvicorn src.main:app --reload --port 8000

# Select model at startup
ACTIVE_MODEL=xgboost uvicorn src.main:app --reload

# Production
ACTIVE_MODEL=random_forest ADMIN_API_TOKEN=<secret> uvicorn src.main:app --workers 4
```

---

## API Endpoints

### Inference

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict/` | Fault prediction from a sensor snapshot (RF-05). |
| `GET` | `/predictions/` | Paginated history of persisted predictions (RF-09). |
| `GET` | `/health` | Liveness probe. |

### Model Management (RF-11) — requires `X-Admin-Token` header

| Method | Path | Description |
|---|---|---|
| `GET` | `/models` | List all registered models, their active status, and whether the artefact is present on disk. |
| `PUT` | `/models/active` | Hot-swap the active inference model without restarting the server (RNF-25). |

#### GET /models

```bash
curl -H "X-Admin-Token: $ADMIN_API_TOKEN" http://localhost:8000/models
```

```json
{
  "active_model": "random_forest",
  "models": [
    {"name": "mlp",           "active": false, "artefact_ready": true},
    {"name": "random_forest", "active": true,  "artefact_ready": true},
    {"name": "xgboost",       "active": false, "artefact_ready": true}
  ]
}
```

#### PUT /models/active

```bash
curl -X PUT \
  -H "X-Admin-Token: $ADMIN_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "xgboost"}' \
  http://localhost:8000/models/active
```

```json
{
  "previous_model": "random_forest",
  "active_model": "xgboost",
  "message": "Active model swapped from 'random_forest' to 'xgboost' successfully."
}
```

Valid values for `model_name`: `"random_forest"`, `"xgboost"`, `"mlp"`.

**Error responses:**
- `401` — Missing or incorrect `X-Admin-Token`.
- `404` — Model artefact file not found on disk.
- `422` — Invalid `model_name` value.

---

## Atomic Hot-Swap (RNF-25)

The `ModelRegistry` performs the swap in two phases:

1. **Phase 1 — outside the lock:** The new model artefact is loaded from disk
   in a background thread (`asyncio.to_thread`). In-flight prediction requests
   continue using the old model normally during this phase.

2. **Phase 2 — inside the lock (nanoseconds):** Only the Python reference
   `self._service = new_service` executes under the `asyncio.Lock`. CPython's
   GIL ensures a single reference assignment is atomic; no request ever sees a
   partially-initialised model. The lock also serialises concurrent swap requests.

---

## Tests

```bash
pytest                    # all tests
pytest tests/test_model_registry.py -v   # registry + admin endpoints only
```
