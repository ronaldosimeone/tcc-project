"""
Testes de InferenceCache — RNF-70 / RNF-71.

Três camadas, propósito diferente cada uma:

1. `TestBuildCacheKey` — função pura, sem Redis. Determinismo/unicidade da
   chave (RNF-70 §6).

2. `TestInferenceCacheUnit` — InferenceCache contra um cliente FALSO
   (`_FakeRedisClient`, dict em memória) — orquestração de get/set e
   fallback em erro (RNF-70 §8), sem depender de infraestrutura externa.
   Mesmo padrão do resto do projeto (test_celery_notifications.py: "não
   depende de Redis real rodando").

3. `TestInferenceCacheRealRedis` — a EXCEÇÃO deliberada a esse padrão: RNF-71
   pede TTL observável, e um mock não PROVA que o Redis real expira a chave
   — só prova que `set(..., ex=60)` foi chamado. Estes testes exigem um
   Redis de verdade (`TEST_REDIS_URL`, default `redis://localhost:6379/15`
   — DB 15, isolado do cache de dev (db=1) e do broker do Celery (db=0);
   CI sobe um `services: redis` dedicado no job `test-python`). Não há
   fallback/skip se o Redis não estiver acessível — a suíte deve FALHAR
   nesse caso (RNF-70 §16: "não marque testes como skip").

   O teste de expiração usa `PEXPIRE` para encurtar o TTL da chave ESPECÍFICA
   depois que o código de produção já aplicou o `ex=60` real (verificado
   separadamente, sem encurtar nada) — evita um `sleep(60)` real no CI sem
   alterar a constante de produção, e ainda assim exercita a expiração REAL
   do Redis (não uma simulação em Python).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from types import SimpleNamespace
from typing import Any

import pytest
import redis.asyncio as redis_asyncio

from src.schemas.predict import PredictRequest, PredictResponse
from src.services.inference_cache import (
    INFERENCE_CACHE_TTL_SECONDS,
    InferenceCache,
    build_cache_key,
    create_redis_client,
    get_active_model_name,
    get_inference_cache,
)

_VALID_PAYLOAD: dict[str, float] = {
    "TP2": 5.02,
    "TP3": 9.21,
    "H1": 8.97,
    "DV_pressure": 2.10,
    "Reservoirs": 8.85,
    "Motor_current": 4.5,
    "Oil_temperature": 72.3,
    "COMP": 1.0,
    "DV_eletric": 0.0,
    "Towers": 1.0,
    "MPG": 1.0,
    "Oil_level": 1.0,
}

_TEST_REDIS_URL: str = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15")


def _make_response(prob: float = 0.42) -> PredictResponse:
    return PredictResponse(
        predicted_class=0,
        failure_probability=prob,
        timestamp="2026-01-01T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# 1. build_cache_key — função pura
# ---------------------------------------------------------------------------


class TestBuildCacheKey:
    def test_same_payload_same_model_same_key(self) -> None:
        req = PredictRequest(**_VALID_PAYLOAD)
        k1 = build_cache_key(req, model_name="random_forest_v2")
        k2 = build_cache_key(req, model_name="random_forest_v2")
        assert k1 == k2

    def test_different_sensor_value_different_key(self) -> None:
        req_a = PredictRequest(**_VALID_PAYLOAD)
        req_b = PredictRequest(
            **{**_VALID_PAYLOAD, "TP2": _VALID_PAYLOAD["TP2"] + 0.01}
        )
        assert build_cache_key(req_a, "random_forest_v2") != build_cache_key(
            req_b, "random_forest_v2"
        )

    def test_different_model_different_key(self) -> None:
        req = PredictRequest(**_VALID_PAYLOAD)
        assert build_cache_key(req, "random_forest_v2") != build_cache_key(
            req, "xgboost_v2"
        )

    def test_key_contains_model_name_and_prefix(self) -> None:
        req = PredictRequest(**_VALID_PAYLOAD)
        key = build_cache_key(req, "random_forest_v2")
        assert key.startswith("predict-cache:v1:random_forest_v2:")

    def test_key_matches_known_golden_digest(self) -> None:
        """
        Pino o SHA256 exato para um payload conhecido — sem isso, um mutante
        que altera a string de entrada do hash (ex.: concatenação errada,
        separador trocado) ainda passaria em "mesma entrada -> mesma
        chave"/"entrada diferente -> chave diferente" (mutmut confirmou:
        sobrevivente real encontrado ao rodar mutation testing neste
        arquivo). Valor calculado uma vez com o algoritmo real de
        build_cache_key e também confirmado batendo com a chave observada
        de verdade no Redis durante o teste manual desta task.
        """
        req = PredictRequest(**_VALID_PAYLOAD)
        key = build_cache_key(req, "random_forest_v2")
        assert key == (
            "predict-cache:v1:random_forest_v2:"
            "dab7bc4c7142e23a61ee672685d58f7f70c74e27616077d6cf628cb4c91a0268"
        )

    def test_field_order_does_not_affect_key(self) -> None:
        """PredictRequest é sempre construído com os mesmos campos — a chave
        usa json.dumps(sort_keys=True), então a ORDEM de construção do dict
        de entrada nunca deveria importar; valida isso construindo o mesmo
        payload em duas ordens de kwargs diferentes."""
        items = list(_VALID_PAYLOAD.items())
        req_a = PredictRequest(**dict(items))
        req_b = PredictRequest(**dict(reversed(items)))
        assert build_cache_key(req_a, "m") == build_cache_key(req_b, "m")


# ---------------------------------------------------------------------------
# 2. InferenceCache — cliente falso (unit, sem infraestrutura)
# ---------------------------------------------------------------------------


class _FakeRedisClient:
    """Dublê mínimo de `redis.asyncio.Redis` — só o que InferenceCache usa."""

    def __init__(self, *, raise_on: set[str] | None = None) -> None:
        self._store: dict[str, str] = {}
        self._raise_on = raise_on or set()

    async def get(self, key: str) -> str | None:
        if "get" in self._raise_on:
            raise ConnectionError("simulated redis outage")
        return self._store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        if "set" in self._raise_on:
            raise ConnectionError("simulated redis outage")
        self._store[key] = value


class TestInferenceCacheUnit:
    @pytest.mark.asyncio
    async def test_miss_returns_none(self) -> None:
        cache = InferenceCache(_FakeRedisClient())  # type: ignore[arg-type]
        assert await cache.get("missing-key") is None

    @pytest.mark.asyncio
    async def test_set_then_get_round_trips(self) -> None:
        cache = InferenceCache(_FakeRedisClient())  # type: ignore[arg-type]
        response = _make_response(0.77)
        await cache.set("k", response)
        got = await cache.get("k")
        assert got is not None
        assert got.failure_probability == pytest.approx(0.77)
        assert got.predicted_class == response.predicted_class
        assert got.timestamp == response.timestamp

    @pytest.mark.asyncio
    async def test_get_failure_falls_back_to_miss(self) -> None:
        """RNF-70 §8 — Redis indisponível no GET nunca lança, vira MISS."""
        cache = InferenceCache(_FakeRedisClient(raise_on={"get"}))  # type: ignore[arg-type]
        assert await cache.get("any-key") is None

    @pytest.mark.asyncio
    async def test_set_failure_never_raises(self) -> None:
        """RNF-70 §8 — Redis indisponível no SET nunca lança (no-op)."""
        cache = InferenceCache(_FakeRedisClient(raise_on={"set"}))  # type: ignore[arg-type]
        await cache.set("any-key", _make_response())  # não deve lançar

    @pytest.mark.asyncio
    async def test_corrupt_entry_falls_back_to_miss(self) -> None:
        """Entrada corrompida no Redis (payload não-JSON válido para
        PredictResponse) é tratada como MISS, não como erro 500."""
        fake = _FakeRedisClient()
        await fake.set("bad-key", "{not valid json")
        cache = InferenceCache(fake)  # type: ignore[arg-type]
        assert await cache.get("bad-key") is None

    @pytest.mark.asyncio
    async def test_get_failure_logs_event_name_with_traceback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        Fixa o nome do evento de log e `exc_info=True` — sem isso, mutation
        testing (mutmut) encontrou sobreviventes reais aqui: o texto do
        evento (`inference_cache_get_failed`) e a flag `exc_info` podiam
        mudar sem nenhum teste notar. Nomes de evento estáveis importam
        neste projeto (mesmo padrão de `alert_triggered`/`alert_skipped`
        em alert_service.py) — são o que um operador busca nos logs pra
        diagnosticar Redis fora do ar.
        """
        cache = InferenceCache(_FakeRedisClient(raise_on={"get"}))  # type: ignore[arg-type]
        with caplog.at_level(logging.WARNING):
            await cache.get("any-key")
        assert len(caplog.records) == 1
        assert caplog.records[0].getMessage() == "inference_cache_get_failed"
        # `exc_info=False` grava literalmente `False` no record (não `None`)
        # — checagem truthy é a única que distingue True de False de verdade
        # (achado real: `is not None` deixava passar `exc_info=False` batido).
        assert caplog.records[0].exc_info

    @pytest.mark.asyncio
    async def test_corrupt_entry_logs_event_name_with_traceback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        fake = _FakeRedisClient()
        await fake.set("bad-key", "{not valid json")
        cache = InferenceCache(fake)  # type: ignore[arg-type]
        with caplog.at_level(logging.WARNING):
            await cache.get("bad-key")
        assert len(caplog.records) == 1
        assert caplog.records[0].getMessage() == "inference_cache_corrupt_entry"
        assert caplog.records[0].exc_info

    @pytest.mark.asyncio
    async def test_set_failure_logs_event_name_with_traceback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        cache = InferenceCache(_FakeRedisClient(raise_on={"set"}))  # type: ignore[arg-type]
        with caplog.at_level(logging.WARNING):
            await cache.set("any-key", _make_response())
        assert len(caplog.records) == 1
        assert caplog.records[0].getMessage() == "inference_cache_set_failed"
        assert caplog.records[0].exc_info

    @pytest.mark.asyncio
    async def test_create_redis_client_returns_async_redis(self) -> None:
        """Só a construção (sem I/O — redis.asyncio conecta lazy no 1º
        comando) — cobre a fábrica usada pelo lifespan em src/main.py.
        Fecha o cliente ao final — sem isso, o pool de conexão não fechado
        dispara um PytestUnraisableExceptionWarning no GC de outro teste."""
        client = create_redis_client("redis://localhost:6379/15")
        assert isinstance(client, redis_asyncio.Redis)
        await client.aclose()

    @pytest.mark.asyncio
    async def test_create_redis_client_uses_exact_config(self) -> None:
        """
        Fixa os 3 kwargs reais de conexão — sem isso, mutation testing
        (mutmut) encontrou 2 sobreviventes reais aqui: `decode_responses`
        virando `False` (get() passaria a devolver bytes, quebrando
        `PredictResponse.model_validate_json`) e `socket_connect_timeout`
        virando `1.3` (o timeout curto É a correção documentada em
        create_redis_client — um Redis parado voltaria a inflar /predict
        para ~4s em vez de ~0.6s, ver RELATORIO-RNF-70-RNF-71.md §4.4).
        Nenhum teste anterior inspecionava os kwargs de verdade.
        """
        client = create_redis_client("redis://localhost:6379/15")
        try:
            kwargs = client.connection_pool.connection_kwargs
            assert kwargs["decode_responses"] is True
            assert kwargs["socket_connect_timeout"] == pytest.approx(0.3)
            assert kwargs["socket_timeout"] == pytest.approx(0.3)
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_close_delegates_to_client(self) -> None:
        closed = {"called": False}

        class _ClosableFake(_FakeRedisClient):
            async def aclose(self) -> None:
                closed["called"] = True

        cache = InferenceCache(_ClosableFake())  # type: ignore[arg-type]
        await cache.close()
        assert closed["called"] is True


# ---------------------------------------------------------------------------
# Dependências FastAPI — resolução de app.state (com e sem lifespan)
# ---------------------------------------------------------------------------


class TestFastAPIDependencies:
    @pytest.mark.asyncio
    async def test_get_inference_cache_returns_null_cache_without_lifespan(
        self,
    ) -> None:
        """Sem app.state.inference_cache (ASGITransport nos testes) — MISS
        determinístico, nenhuma tentativa de rede (ver _NullInferenceCache)."""
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
        cache = await get_inference_cache(request)  # type: ignore[arg-type]
        assert await cache.get("any") is None
        await cache.set("any", _make_response())  # não deve lançar

    @pytest.mark.asyncio
    async def test_get_inference_cache_returns_app_state_cache_when_present(
        self,
    ) -> None:
        """Com lifespan real (produção), devolve exatamente a instância
        singleton criada no startup — não uma nova."""
        sentinel = InferenceCache(_FakeRedisClient())  # type: ignore[arg-type]
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(inference_cache=sentinel))
        )
        resolved = await get_inference_cache(request)  # type: ignore[arg-type]
        assert resolved is sentinel

    @pytest.mark.asyncio
    async def test_get_active_model_name_falls_back_to_settings_without_registry(
        self,
    ) -> None:
        from src.core.config import settings

        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
        name = await get_active_model_name(request)  # type: ignore[arg-type]
        assert name == settings.active_model

    @pytest.mark.asyncio
    async def test_get_active_model_name_reads_registry_active_name(self) -> None:
        registry = SimpleNamespace(active_name="xgboost_v2")
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(model_registry=registry))
        )
        name = await get_active_model_name(request)  # type: ignore[arg-type]
        assert name == "xgboost_v2"


# ---------------------------------------------------------------------------
# 3. InferenceCache — Redis REAL (RNF-71 — TTL observável)
# ---------------------------------------------------------------------------


@pytest.fixture()
async def real_redis_client() -> Any:
    client = redis_asyncio.from_url(
        _TEST_REDIS_URL, decode_responses=True, socket_connect_timeout=5.0
    )
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.aclose()


class TestInferenceCacheRealRedis:
    """
    Exige Redis real acessível em TEST_REDIS_URL — sem fallback/skip
    (RNF-70 §16). Localmente: `docker compose up -d redis` e rode com
    `TEST_REDIS_URL=redis://localhost:6379/15` (porta publicada) ou execute
    dentro do container `api` com `redis://redis:6379/15`. No CI: service
    container dedicado do job `test-python` (ver .github/workflows/ci.yml).
    """

    @pytest.mark.asyncio
    async def test_hit_after_miss_real_redis(self, real_redis_client: Any) -> None:
        cache = InferenceCache(real_redis_client)
        key = "test:hit-after-miss"

        assert await cache.get(key) is None  # 1. MISS real

        response = _make_response(0.55)
        await cache.set(key, response)  # 2. SET real

        got = await cache.get(key)  # 3. HIT real
        assert got is not None
        assert got.failure_probability == pytest.approx(0.55)

    @pytest.mark.asyncio
    async def test_ttl_is_exactly_60_seconds(self, real_redis_client: Any) -> None:
        """RNF-71 — o TTL de produção (INFERENCE_CACHE_TTL_SECONDS) é 60, e o
        Redis REAL confirma isso via TTL após um `set()` real — não um
        mock verificando que `ex=60` foi passado como argumento."""
        assert INFERENCE_CACHE_TTL_SECONDS == 60

        cache = InferenceCache(real_redis_client)
        key = "test:ttl-exact"
        await cache.set(key, _make_response())

        ttl = await real_redis_client.ttl(key)
        # Janela pequena (execução real do teste consome frações de segundo
        # entre o SET e este TTL) — nunca > 60, e bem próximo dele.
        assert 55 <= ttl <= 60

    @pytest.mark.asyncio
    async def test_key_expires_and_next_read_is_a_real_miss(
        self, real_redis_client: Any
    ) -> None:
        """
        RNF-71 — comportamento OBSERVÁVEL de expiração, não só a config.

        Fluxo: 1) SET real com o TTL de produção (60s, prova que o código
        real aplica o valor certo); 2) usa PEXPIRE do PRÓPRIO Redis para
        encurtar o TTL DESSA chave para 300ms — não altera
        INFERENCE_CACHE_TTL_SECONDS nem o código de produção, só acelera o
        relógio do Redis para esta chave específica, evitando um
        `sleep(60)` real no pipeline; 3) espera passar da nova expiração;
        4) confirma MISS real — InferenceCache.get() não inventa "ainda
        vivo", ele lê o que o Redis realmente tem.
        """
        cache = InferenceCache(real_redis_client)
        key = "test:ttl-expiry"

        await cache.set(key, _make_response())
        ttl_before = await real_redis_client.ttl(key)
        assert 55 <= ttl_before <= 60  # confirma TTL real de produção antes de acelerar

        await real_redis_client.pexpire(key, 300)  # acelera SÓ esta chave
        await asyncio.sleep(0.6)

        assert await real_redis_client.exists(key) == 0  # o Redis já a removeu
        assert await cache.get(key) is None  # e InferenceCache reporta MISS real

    @pytest.mark.asyncio
    async def test_expired_key_triggers_fresh_miss_not_stale_hit(
        self, real_redis_client: Any
    ) -> None:
        """Depois de expirar, uma nova gravação para a MESMA chave funciona
        normalmente (não fica "presa" em algum estado inconsistente)."""
        cache = InferenceCache(real_redis_client)
        key = "test:ttl-then-reset"

        await cache.set(key, _make_response(0.11))
        await real_redis_client.pexpire(key, 300)
        await asyncio.sleep(0.6)
        assert await cache.get(key) is None

        await cache.set(key, _make_response(0.99))
        got = await cache.get(key)
        assert got is not None
        assert got.failure_probability == pytest.approx(0.99)

    @pytest.mark.asyncio
    async def test_real_redis_roundtrip_elapsed_time_is_fast(
        self, real_redis_client: Any
    ) -> None:
        """Sanity check de performance — get/set contra Redis real (mesmo
        que localhost) não deveria, sozinho, aproximar o orçamento de 50ms
        do RNF-70; a evidência oficial de p95 é o Locust (RELATORIO)."""
        cache = InferenceCache(real_redis_client)
        key = "test:roundtrip-speed"
        await cache.set(key, _make_response())

        start = time.perf_counter()
        await cache.get(key)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 50
