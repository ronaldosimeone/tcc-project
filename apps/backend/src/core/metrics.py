"""
Métricas Prometheus — RNF-76.

Dois grupos, propositalmente separados:

1. HTTP genérico — `setup_http_instrumentation()` liga o
   `prometheus-fastapi-instrumentator`, que expõe `http_requests_total`
   (method/status/handler) e `http_request_duration_seconds`
   (method/handler) a partir de toda requisição real que passa pela app.
   `handler` usa o TEMPLATE da rota (ex. "/predict/"), nunca a URL
   resolvida — sem IDs dinâmicos no label (RNF-76 Fase 2).

2. ML customizado — Counters/Histogram de módulo (singletons; criados uma
   única vez no processo, nunca dentro de `create_app()`, que é chamada
   repetidas vezes pela suíte de testes — ver docstring de
   `test_metrics.py` para o motivo de isso importar).

Cardinalidade dos labels (RNF-76 regra 16/17):
  - `model`: um dos 9 nomes em `ModelRegistry.KNOWN_MODELS` — conjunto
    fixo e pequeno.
  - `status`: "success" | "error" — 2 valores.
  - `prediction_class`: "0" | "1" — classificador binário, 2 valores.
Nenhum UUID, ID de equipamento, request ID ou timestamp entra como label.
"""

from __future__ import annotations

from fastapi import FastAPI
from prometheus_client import CollectorRegistry, Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator, metrics

# ---------------------------------------------------------------------------
# 1. HTTP — prometheus-fastapi-instrumentator
# ---------------------------------------------------------------------------
#
# Registry dedicado por chamada de `setup_http_instrumentation`, NUNCA o
# `prometheus_client.REGISTRY` global — achado real desta task: a suíte de
# testes chama `create_app()` repetidas vezes NO MESMO processo (cada
# arquivo de teste cria sua própria app via `create_app()`, e
# `from src.main import create_app` já executa `app = create_app()` uma
# vez ao nível de módulo antes disso). Com o registry global, a SEGUNDA
# chamada a `metrics.default()` detecta "métrica já registrada" e devolve
# `None` silenciosamente (comportamento documentado da própria lib, pensado
# para `app.build_middleware_stack()` ser chamado várias vezes na MESMA
# app) — resultado: a segunda app em diante fica com
# `instrumentations=[]`, um middleware que existe mas nunca grava nada.
# Um registry novo por chamada elimina a colisão inteiramente; em produção
# (`create_app()` roda uma única vez por processo) o comportamento é
# idêntico a usar o registry global.
#
# As métricas de ML (seção 2, abaixo) são singletons de módulo — criadas
# uma única vez por processo (cache de import do Python), nunca dentro de
# `create_app()` — não sofrem esse problema. Para aparecerem em
# `GET /metrics` junto com as métricas HTTP (que vivem no registry desta
# chamada), são reanexadas ao registry desta app via `registry.register()`
# — um Collector pode estar em vários `CollectorRegistry` ao mesmo tempo
# sem duplicar contagem: cada registry só LÊ o valor atual do mesmo objeto.


def setup_http_instrumentation(app: FastAPI, *, include_in_schema: bool) -> None:
    """
    Liga a instrumentação HTTP padrão e expõe `GET /metrics`.

    `should_group_status_codes=False` — mantém o código de status exato
    (200/422/429/500/503/...) como label em vez de agrupar em "2xx"/"5xx":
    conjunto de códigos que a API realmente devolve é pequeno e fixo, e o
    PromQL de RNF-77 (Fase 4) precisa distinguir 4xx de 5xx sem regex.

    `should_ignore_untemplated=True` — requisições sem rota casada (ex.:
    probes de bot em `/wp-admin`) NUNCA viram uma métrica: sem isso, o
    label `handler` receberia a URL bruta e a cardinalidade cresceria sem
    limite (exatamente o que a regra 16 proíbe).

    `excluded_handlers=["/metrics"]` — o próprio scrape do Prometheus não
    conta como tráfego de aplicação.
    """
    registry = CollectorRegistry()
    for collector in _ML_COLLECTORS:
        registry.register(collector)

    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_group_untemplated=True,
        excluded_handlers=["/metrics"],
        registry=registry,
    )
    instrumentator.add(metrics.default(registry=registry))
    instrumentator.instrument(app)
    instrumentator.expose(app, endpoint="/metrics", include_in_schema=include_in_schema)


# ---------------------------------------------------------------------------
# 2. ML customizado — RNF-76 Fase 3
# ---------------------------------------------------------------------------
#
# Utilidade operacional de cada métrica (por que existe, não só o que mede):
#
#   INFERENCE_TOTAL / INFERENCE_DURATION_SECONDS
#       Cobrem TODA inferência single-sample real: POST /predict E o
#       pipeline contínuo (InferencePipelineService, 1 execução por
#       leitura SSE). É o único jeito de ver erro/latência de inferência
#       do pipeline em background — esse loop nunca passa por uma rota
#       HTTP, então http_requests_total não o enxerga (RNF-76 Fase 4: não
#       confundir erro HTTP com erro de inferência).
#
#   PREDICTIONS_TOTAL
#       Distribuição de classe prevista (0=normal, 1=falha) por modelo —
#       permite ver no Grafana a taxa de "alarme" do modelo ativo ao longo
#       do tempo sem consultar o Postgres.
#
#   BATCH_REQUESTS_TOTAL / BATCH_SAMPLES_TOTAL
#       POST /predict/batch é UMA chamada vetorizada para N amostras — não
#       é semanticamente "N inferências" (não passa por
#       predict_from_features), por isso tem contadores próprios em vez
#       de somar em INFERENCE_TOTAL.

INFERENCE_TOTAL = Counter(
    "predictiq_inference_total",
    "Total de inferências single-sample (POST /predict + pipeline contínuo).",
    ["model", "status"],
)

INFERENCE_DURATION_SECONDS = Histogram(
    "predictiq_inference_duration_seconds",
    "Duração da inferência single-sample, em segundos (predict_proba isolado).",
    ["model", "status"],
)

PREDICTIONS_TOTAL = Counter(
    "predictiq_predictions_total",
    "Distribuição de classe prevista (0=normal, 1=falha) por modelo.",
    ["model", "prediction_class"],
)

BATCH_REQUESTS_TOTAL = Counter(
    "predictiq_batch_requests_total",
    "Total de chamadas a POST /predict/batch.",
    ["model", "status"],
)

BATCH_SAMPLES_TOTAL = Counter(
    "predictiq_batch_samples_total",
    "Total de amostras processadas com sucesso via POST /predict/batch.",
    ["model"],
)


def record_inference(
    model: str,
    status: str,
    duration_seconds: float,
    predicted_class: int | None = None,
) -> None:
    """Registra uma inferência single-sample. `predicted_class=None` em erro."""
    INFERENCE_TOTAL.labels(model=model, status=status).inc()
    INFERENCE_DURATION_SECONDS.labels(model=model, status=status).observe(
        duration_seconds
    )
    if predicted_class is not None:
        PREDICTIONS_TOTAL.labels(
            model=model, prediction_class=str(predicted_class)
        ).inc()


def record_batch(model: str, status: str, sample_count: int) -> None:
    """Registra uma chamada a POST /predict/batch. `sample_count` só soma em sucesso."""
    BATCH_REQUESTS_TOTAL.labels(model=model, status=status).inc()
    if status == "success":
        BATCH_SAMPLES_TOTAL.labels(model=model).inc(sample_count)


# Referenciado por `setup_http_instrumentation()` — ver comentário da seção 1.
_ML_COLLECTORS: tuple[Counter | Histogram, ...] = (
    INFERENCE_TOTAL,
    INFERENCE_DURATION_SECONDS,
    PREDICTIONS_TOTAL,
    BATCH_REQUESTS_TOTAL,
    BATCH_SAMPLES_TOTAL,
)
