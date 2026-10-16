"""
Distributed tracing with OpenTelemetry (complements the Prometheus metrics).

When ``OTEL_ENABLED=true`` every HTTP request handled by the API becomes a
trace, and the outgoing HTTP calls it makes — MCP search (``search_manuals``)
and Ollama generation — become child spans automatically (httpx
instrumentation propagates ``traceparent``). The RAG pipeline adds its own
spans with domain attributes (contexts kept/discarded by τ, grounding
status, safety guard decisions), so one trace answers "why was this plan
slow / ungrounded?".

Export: OTLP/HTTP to ``OTEL_EXPORTER_OTLP_ENDPOINT`` (Jaeger all-in-one in
docker-compose). Disabled (default) → the OpenTelemetry API stays a no-op,
so ``tracer.start_as_current_span`` costs nothing in tests and dev.
"""

from __future__ import annotations

import structlog
from fastapi import FastAPI
from opentelemetry import trace

from src.core.config import settings

log = structlog.get_logger(__name__)

_EXCLUDED_URLS = "metrics,health,stream/sensors,ws/alerts"


def setup_tracing(app: FastAPI) -> bool:
    """Install the tracer provider + auto-instrumentation. Returns True if enabled."""
    if not settings.otel_enabled:
        return False
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(
        resource=Resource.create({SERVICE_NAME: settings.otel_service_name})
    )
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=f"{settings.otel_exporter_otlp_endpoint.rstrip('/')}/v1/traces"
            )
        )
    )
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app, excluded_urls=_EXCLUDED_URLS)
    HTTPXClientInstrumentor().instrument()
    log.info("otel_tracing_enabled", endpoint=settings.otel_exporter_otlp_endpoint)
    return True


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)
