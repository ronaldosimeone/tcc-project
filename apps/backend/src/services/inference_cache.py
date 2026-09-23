"""
InferenceCache — RNF-70 (p95 < 50ms em cache hit) / RNF-71 (TTL de 60s).

Cache-aside para POST /predict (src/routers/predict.py): uma requisição com
os MESMOS 12 sensores + o MESMO modelo ativo, dentro da janela de 60s, pula
inferência ONNX/sklearn, o INSERT em `predictions` e o processamento de
alerta inteiro — devolve a MESMA PredictResponse já computada.

Infra reaproveitada — RNF-50/51 já trouxe Redis para o projeto (broker do
Celery, `src/core/celery_app.py`). Este módulo usa a MESMA instância, só
logicamente isolada num DB Redis diferente (`settings.redis_cache_url`,
db=1 — broker fica em db=0) para eviction/FLUSHDB do cache nunca tocar a
fila de notificações.

Fallback — Redis indisponível (connection refused, timeout, DNS) NUNCA pode
derrubar POST /predict: get()/set() engolem qualquer exceção do cliente
Redis, logam e se comportam como MISS/no-op — o endpoint volta a se
comportar exatamente como antes desta feature existir.
"""

from __future__ import annotations

import hashlib
import json
import logging

import redis.asyncio as redis_asyncio
from fastapi import Request

from src.core.config import settings
from src.schemas.predict import PredictRequest, PredictResponse

logger: logging.Logger = logging.getLogger(__name__)

# RNF-71 — TTL fixo em código, deliberadamente NÃO exposto como variável de
# ambiente: se fosse configurável por .env, um deploy poderia silenciosamente
# violar o requisito "60 segundos". Único lugar do projeto que define esse
# valor — src/routers/predict.py importa esta constante em vez de repetir
# "60" em outro lugar.
INFERENCE_CACHE_TTL_SECONDS: int = 60

_KEY_PREFIX: str = "predict-cache:v1"

# Os 12 campos de PredictRequest — EXATAMENTE os inputs que
# ModelService._build_feature_row (model_service.py) usa para montar o vetor
# de features do caminho stateless de POST /predict. Nenhum outro dado
# (timestamp, headers, etc.) influencia o resultado da inferência.
_SENSOR_FIELDS: tuple[str, ...] = (
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Motor_current",
    "Oil_temperature",
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "Oil_level",
)


def build_cache_key(payload: PredictRequest, model_name: str) -> str:
    """
    Chave determinística — RNF-70 §6.

    Inclui tudo que pode alterar o resultado para um request OUTRAMENTE
    idêntico:
      - os 12 sensores brutos (valores exatos — o caminho stateless de
        ModelService.predict() é função pura desses campos);
      - o MODELO ATIVO (`ModelRegistry.active_name`, RF-10). Um hot-swap via
        PUT /models/active pode mudar predicted_class/failure_probability
        para o MESMO snapshot de sensores; sem isso na chave, uma entrada
        cacheada do modelo antigo serviria silenciosamente uma predição do
        modelo errado por até 60s.

    Deliberadamente NÃO inclui:
      - `decision_threshold` — derivado 1:1 do model_name via o model card
        (ver model_service.py::_resolve_threshold); incluir seria
        redundante, não uma proteção adicional.
      - `timestamp` — não é input da inferência, é OUTPUT (gerado no
        momento do cálculo).

    Hash (não os valores brutos) na chave Redis — chave de tamanho fixo,
    nenhum dado de sensor exposto em `redis-cli KEYS`/logs de infra.
    """
    canonical = json.dumps(
        {field: getattr(payload, field) for field in _SENSOR_FIELDS},
        sort_keys=True,
    )
    digest = hashlib.sha256(f"{model_name}|{canonical}".encode("utf-8")).hexdigest()
    return f"{_KEY_PREFIX}:{model_name}:{digest}"


def create_redis_client(redis_url: str) -> redis_asyncio.Redis:
    """
    Cliente Redis assíncrono (RNF-70) — `redis-py` >= 4.2 (`redis.asyncio`),
    já uma dependência de produção deste projeto (`redis==5.0.8`,
    requirements.txt, usado hoje só como broker do Celery). Nenhuma
    dependência nova.

    Timeouts curtos (300ms) — RNF-70 §8: se o Redis estiver fora do ar (ou
    o container simplesmente parar, sem devolver connection-refused), cada
    tentativa de get/set precisa falhar rápido (vira MISS/no-op logo, ver
    InferenceCache) em vez de segurar POST /predict pelo timeout default do
    socket. Validado empiricamente: com o timeout default (vários segundos)
    um Redis parado inflava a resposta de /predict para ~4s (2 tentativas —
    get + set — no caminho de MISS); 300ms é generoso frente ao round-trip
    local típico (<5ms) e mantém o pior caso do fallback abaixo de ~600ms.
    """
    return redis_asyncio.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=0.3,
        socket_timeout=0.3,
    )


class InferenceCache:
    """Cache-aside Redis para PredictResponse — RNF-70/RNF-71."""

    def __init__(self, client: redis_asyncio.Redis) -> None:
        self._client = client

    async def get(self, key: str) -> PredictResponse | None:
        """MISS em qualquer falha do Redis ou entrada corrompida — nunca lança."""
        try:
            raw: str | None = await self._client.get(key)
        except Exception:
            logger.warning("inference_cache_get_failed", exc_info=True)
            return None

        if raw is None:
            return None

        try:
            return PredictResponse.model_validate_json(raw)
        except Exception:
            logger.warning("inference_cache_corrupt_entry", exc_info=True)
            return None

    async def set(self, key: str, value: PredictResponse) -> None:
        """No-op em qualquer falha do Redis — nunca lança, nunca bloqueia a resposta."""
        try:
            await self._client.set(
                key,
                value.model_dump_json(),
                ex=INFERENCE_CACHE_TTL_SECONDS,
            )
        except Exception:
            logger.warning("inference_cache_set_failed", exc_info=True)

    async def close(self) -> None:
        await self._client.aclose()


class _NullInferenceCache:
    """
    No-op — MISS sempre, `set` não faz nada, nenhum I/O de rede.

    Usado quando `app.state.inference_cache` não existe (lifespan não
    rodou — `ASGITransport` nos testes, ver comentário em src/main.py).

    Achado real desta task: a primeira versão deste fallback construía um
    `InferenceCache` de verdade contra `settings.redis_cache_url` e contava
    com esse host "provavelmente" não resolver/recusar conexão em ambiente
    de teste. Rodando a suíte inteira DENTRO do container `api` (onde o
    hostname `redis` resolve de verdade) isso quebrou 8 testes em 3 arquivos
    (test_predictions_endpoint.py, test_infrastructure.py,
    test_dependency_injection.py) — todos reusam o mesmo payload de
    exemplo, então o 2º+ POST /predict de cada um virava HIT de um cache
    Redis real e pulava a persistência/exceção que o teste esperava.
    Depender de "a rede provavelmente vai falhar" para isolar testes é
    frágil por definição — este sentinel elimina a dependência de rede por
    completo: sem lifespan, NUNCA há cache, determinístico em qualquer
    ambiente. Os testes de cache (test_inference_cache.py,
    test_predict_cache_integration.py) continuam funcionando porque
    sobrescrevem `get_inference_cache` explicitamente com um
    InferenceCache real.
    """

    async def get(self, key: str) -> PredictResponse | None:
        return None

    async def set(self, key: str, value: PredictResponse) -> None:
        return None


_null_cache = _NullInferenceCache()


# ---------------------------------------------------------------------------
# Dependências FastAPI
# ---------------------------------------------------------------------------


async def get_inference_cache(request: Request) -> InferenceCache | _NullInferenceCache:
    """
    Resolve o InferenceCache do `app.state` (criado uma vez no lifespan —
    mesmo padrão de ModelRegistry, CLAUDE.md §4 "singleton no lifespan").

    Sem lifespan (app.state.inference_cache ausente), devolve o sentinel
    `_NullInferenceCache` — MISS sempre, sem nenhuma tentativa de rede (ver
    docstring da classe acima para o porquê deste design).
    """
    cache = getattr(request.app.state, "inference_cache", None)
    if cache is not None:
        return cache
    return _null_cache


async def get_active_model_name(request: Request) -> str:
    """
    Nome do modelo ativo para compor a cache key (RNF-70 §6) — leitura
    "soft" do `ModelRegistry` em app.state, mesmo padrão de tolerância a
    lifespan ausente já usado por `model_service.get_model_service`.
    `get_model_registry` (model_registry.py) NÃO serve aqui porque ele
    lança `RuntimeError` quando o registry não está em app.state — correto
    para as rotas administrativas de /models, errado para o cache (que deve
    degradar para "sem cache", nunca quebrar a predição).
    """
    registry = getattr(request.app.state, "model_registry", None)
    if registry is not None:
        return registry.active_name
    return settings.active_model
