"""
FastAPI application entrypoint.

Responsibilities of this module:
- Configure structured logging (structlog) before any I/O.
- Create and configure the FastAPI instance (CORS, lifespan).
- Register all routers.
- Register global exception handlers (domain errors, rate limits, catch-all).
- Nothing else — business logic belongs in services/.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.core.config import settings
from src.core.database import engine
from src.core.exceptions import (
    AppError,
    app_error_handler,
    unhandled_exception_handler,
)
from src.core.logging import configure_logging
from src.core.metrics import setup_http_instrumentation
from src.core.rate_limit import limiter, rate_limit_exceeded_handler
from src.routers import health as health_router
from src.core.database import AsyncSessionFactory
from src.core.security import ensure_secure_config
from src.core.tracing import setup_tracing
from src.routers import auth as auth_router
from src.routers import history as history_router
from src.services.auth_service import AuthService
from src.services.asset_tracking import AssetTracking
from src.services.failure_event_service import FailureEventService
from src.services.asset_processor import AssetProcessor
from src.services.shadow_service import ShadowEvaluator, is_canary
from src.services.fleet_service import FleetService
from src.services.mqtt_telemetry import (
    MqttConfig,
    MqttTelemetryIngestor,
    MqttTelemetryPublisher,
)
from src.services.root_cause_service import RootCauseService
from src.services.simulator import get_simulator
from src.routers import maintenance as maintenance_router
from src.routers import models as models_router
from src.routers import monitoring as monitoring_router
from src.routers import observability as observability_router
from src.routers import predict as predict_router
from src.routers import predictions as predictions_router
from src.routers import settings as settings_router
from src.routers import stream as stream_router
from src.routers import alerts_ws
from src.routers import simulator as simulator_router
from src.services.alert_service import get_alert_service
from src.services.inference_cache import InferenceCache, create_redis_client
from src.services.inference_pipeline import InferencePipelineService
from src.services.model_registry import ModelRegistry
from src.services.sensor_stream_service import get_sensor_stream_service

# ── Bootstrap structured logging immediately ──────────────────────────────
# Must happen before any log.* call so processors are fully configured.
configure_logging(debug=settings.debug)

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage startup and shutdown of shared resources.

    Startup:
        - Configure the rate limiter on app.state.
        - Load the Random Forest model into memory as a singleton (RNF-11).

    Shutdown:
        - Dispose the SQLAlchemy async connection pool gracefully.
    """
    # ── Startup ──────────────────────────────────────────────────────────
    ensure_secure_config()  # AUTH_ENABLED=true exige JWT_SECRET real
    if settings.bootstrap_admin_username and settings.bootstrap_admin_password:
        try:
            async with AsyncSessionFactory() as db:
                await AuthService(db).bootstrap_admin(
                    settings.bootstrap_admin_username, settings.bootstrap_admin_password
                )
        except Exception as exc:  # banco ainda sem a migração 0007, etc.
            log.warning("auth_bootstrap_skipped", error=str(exc))
    app.state.limiter = limiter

    # RNF-70/71 — cliente Redis do InferenceCache, singleton do lifespan
    # (mesmo padrão do ModelRegistry abaixo). Não bloqueante: redis.asyncio
    # só conecta de fato no primeiro comando — se o Redis estiver fora do
    # ar no startup, o app sobe normalmente e InferenceCache degrada para
    # MISS/no-op (ver services/inference_cache.py).
    redis_client = create_redis_client(settings.redis_cache_url)
    app.state.inference_cache = InferenceCache(redis_client)

    registry = ModelRegistry()
    app.state.model_registry = registry

    try:
        await registry.swap(settings.active_model)
        log.info("model_loaded", model=settings.active_model)
    except FileNotFoundError:
        log.warning(
            "model_artefact_missing",
            model=settings.active_model,
            consequence="POST /predict will return HTTP 503 until a model is loaded via PUT /models/active",
        )

    # Inicia o pipeline de inferência contínua: SSE stream → ML → DB → alertas.
    # O pipeline subscreve ao mesmo SensorStreamService que o frontend SSE usa,
    # garantindo que mudanças de modo (PUT /simulator/mode) sejam refletidas
    # imediatamente nas predições e nos alertas WebSocket.
    # Histórico real: o mesmo rastreador RN-03 + causa raiz serve o ativo
    # principal e a frota simulada (cada ativo com seu tracker interno).
    events = FailureEventService(
        threshold=settings.event_threshold,
        k_open=settings.event_k_open,
        k_close=settings.event_k_close,
        critical_at=settings.event_critical_at,
    )
    tracking = AssetTracking(
        events, RootCauseService(), lambda: get_simulator().normal_medians()
    )
    app.state.asset_tracking = tracking

    # Champion / challenger: shadow no ativo principal + canary na frota.
    challenger: ModelRegistry | None = None
    if settings.challenger_model:
        challenger = ModelRegistry()
        try:
            await challenger.swap(settings.challenger_model)
            log.info("challenger_loaded", model=settings.challenger_model)
        except Exception as exc:
            log.warning(
                "challenger_unavailable",
                model=settings.challenger_model,
                error=str(exc),
            )
            challenger = None
    app.state.challenger_registry = challenger

    def processor_for(asset_id: str) -> AssetProcessor:
        use_challenger = challenger is not None and is_canary(
            asset_id, settings.canary_fraction
        )
        return AssetProcessor(
            asset_id,
            challenger if use_challenger and challenger else registry,
            tracking,
        )

    pipeline = InferencePipelineService(
        stream_service=get_sensor_stream_service(),
        registry=registry,
        alert_service=get_alert_service(),
        tracking=tracking,
        shadow=ShadowEvaluator(challenger) if challenger is not None else None,
    )
    pipeline.start()
    app.state.inference_pipeline = pipeline
    log.info("inference_pipeline_ready")

    # Telemetria da frota: in-process (default) ou via broker MQTT.
    publisher: MqttTelemetryPublisher | None = None
    ingestor: MqttTelemetryIngestor | None = None
    if settings.telemetry_transport == "mqtt":
        mqtt_config = MqttConfig(
            host=settings.mqtt_host,
            port=settings.mqtt_port,
            username=settings.mqtt_username,
            password=settings.mqtt_password,
            prefix=settings.mqtt_topic_prefix,
        )
        ingestor = MqttTelemetryIngestor(mqtt_config, processor_for)
        ingestor.start()
        if settings.fleet_size > 0:
            publisher = MqttTelemetryPublisher(mqtt_config)
            try:
                await publisher.start()
            except Exception as exc:
                log.warning("mqtt_publisher_unavailable", error=str(exc))
                publisher = None
    app.state.mqtt_ingestor = ingestor

    fleet: FleetService | None = None
    if settings.fleet_size > 0:
        fleet = FleetService(
            settings.fleet_size,
            registry,
            tracking,
            get_simulator(),
            noise_frac=settings.fleet_noise,
            seed=settings.fleet_seed,
            publisher=publisher,
            canary_registry=challenger,
            canary_fraction=settings.canary_fraction,
        )
        fleet.start()
    app.state.fleet = fleet

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    await pipeline.stop()
    if fleet is not None:
        await fleet.stop()
    if ingestor is not None:
        await ingestor.stop()
    if publisher is not None:
        await publisher.stop()
    await tracking.aclose()
    await engine.dispose()
    log.info("db_pool_disposed")
    await app.state.inference_cache.close()
    log.info("inference_cache_redis_closed")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Construct and fully configure the FastAPI application."""

    # RNF-62 — a UI interativa do OpenAPI (`/docs`, `/redoc`) e o schema
    # (`/openapi.json`) só ficam expostos com `DEBUG=true`. Em produção
    # (`DEBUG` desligado) o ZAP não tem mais o Swagger UI para apontar
    # (CSP `unsafe-inline`, SRI ausente, JS de CDN externa — inerentes ao
    # Swagger UI, não à API) e a superfície de enumeração de endpoints some.
    _docs_enabled = settings.debug
    app: FastAPI = FastAPI(
        title=settings.project_name,
        version=settings.version,
        docs_url="/docs" if _docs_enabled else None,
        redoc_url="/redoc" if _docs_enabled else None,
        openapi_url="/openapi.json" if _docs_enabled else None,
        root_path="/api",
        lifespan=lifespan,
    )

    # ── Rate limiter — must be on app.state BEFORE the first request ────────
    # SlowAPIMiddleware reads app.state.limiter on every request.  We set it
    # here (in create_app) rather than relying solely on the lifespan, because
    # httpx ASGITransport used in tests does not trigger the ASGI lifespan.
    # Setting it twice (here + lifespan) is idempotent and harmless.
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)

    # ── Observabilidade — RNF-76 ─────────────────────────────────────────
    # GET /metrics: aparece no Swagger só quando DEBUG=true, mesma regra
    # de exposição que /docs/redoc/openapi.json acima (RNF-62 §127-132) —
    # não é uma decisão nova, só aplica o padrão já existente no projeto.
    setup_http_instrumentation(app, include_in_schema=_docs_enabled)

    # ── CORS ─────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception handlers — ordered from specific to generic ─────────────
    # FastAPI walks the exception MRO, so subclasses (AppError) always win
    # over the generic Exception catch-all, regardless of registration order.
    # We register explicitly for clarity and documentation purposes.
    app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)  # type: ignore[arg-type]
    # RNF-18: catch-all — any unhandled exception returns HTTP 500 with a
    # safe JSON body; the full traceback is written to structured logs only.
    app.add_exception_handler(Exception, unhandled_exception_handler)  # type: ignore[arg-type]

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(health_router.router)
    app.include_router(predict_router.router)
    app.include_router(predictions_router.router)
    app.include_router(models_router.router)
    app.include_router(stream_router.router)
    app.include_router(simulator_router.router)
    app.include_router(alerts_ws.router)
    app.include_router(maintenance_router.router)
    app.include_router(history_router.router)
    app.include_router(auth_router.router)
    app.include_router(settings_router.router)
    app.include_router(monitoring_router.router)
    app.include_router(observability_router.router)

    setup_tracing(app)  # OTEL_ENABLED=true → traces OTLP (API → MCP → Ollama)

    return app


app: FastAPI = create_app()
