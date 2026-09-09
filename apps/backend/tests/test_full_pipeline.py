"""
Suíte de integração do pipeline completo — RF-26 / RNF-52.

    Sensor Data
        v
    Detecção (ONNX — já coberto por testes de ML dedicados; aqui
              representado pelo CONTRATO EXATO de saída que
              InferencePipelineService._process entrega a AlertService, ver
              item "Detecção" abaixo)
        v
    AlertService.process_prediction()
        v
    ┌── RF-14: broadcast WebSocket (probability > 0.70)
    └── RF-24/25: CriticalFailureNotificationService.notify_if_critical()
            v
        RNF-50: enqueue_notification(payload) — Celery, NUNCA envio síncrono
            v
        celery-worker (testado aqui via chamada direta e controlada da task)
            v
        TelegramNotificationAdapter / EmailNotificationAdapter (mockados)

    (ramo independente, mesmo evento de detecção)
        v
    MaintenanceSuggestionService.suggest()  — RF-22/23
        v
    MCPSearchClient.search_maintenance_manual()  — RF-20/21 (mockado no
        limite real de rede que MaintenanceSuggestionService usa)
        v
    OllamaClient.generate()  — RF-22 (mockado)
        v
    MaintenanceSuggestionResponse (Markdown validado + referências)

AUDITORIA — achado arquitetural documentado (não inventado para o teste)
-------------------------------------------------------------------------
`MaintenanceSuggestionService` NUNCA é chamado a partir de `AlertService`/
`CriticalFailureNotificationService`/`InferencePipelineService` na produção
real (confirmado via grep em todo `src/` — o único ponto de construção é
`routers/maintenance.py`, acionado sob demanda pelo frontend). Os dois
"ramos" desta suíte (notificação crítica e sugestão de manutenção) são,
hoje, fluxos INDEPENDENTES que só compartilham a mesma predição de origem —
não uma cadeia de chamadas única. Este teste exercita os dois ramos a partir
do MESMO evento de detecção determinístico, sem inventar uma integração
direta entre os serviços que não existe no código real.

Por que MCP é mockado, não uma ChromaDB real neste arquivo
-------------------------------------------------------------------------
`SemanticSearchService`/ChromaDB vivem em `apps/mcp-server` — um pacote
Python e um container Docker inteiramente separados de `apps/backend`, sem
`chromadb`/`sentence-transformers` nas dependências do backend (auditado —
nenhum dos dois módulos está instalado neste container). O backend só fala
com o MCP pela rede (`MCPSearchClient`, protocolo streamable-http) — nunca
em processo. Mockar em `MCPSearchClient.search_maintenance_manual()` é o
MENOR ponto de integração real (a fronteira que `MaintenanceSuggestionService`
de fato usa), evitando tanto duplicar a lógica de produção (reimplementar
cosine similarity aqui) quanto inflar o backend com uma dependência pesada
só para este teste. A cobertura real, com ChromaDB efêmero + embeddings
determinísticos capazes de distinguir bomba/motor/irrelevante, já existe e
foi estendida nesta mesma task em
`apps/mcp-server/tests/test_semantic_search.py::test_distinguishes_pump_motor_and_irrelevant_queries_deterministically`.

RNF-52 — determinismo
-------------------------------------------------------------------------
Sem `random`/`uuid4` sem controle, sem `datetime.now()` nas ENTRADAS do
teste (timestamps/queries/scores fixos), sem download de modelo, sem Ollama/
MCP/Telegram/Resend/Redis/Postgres reais. `AlertService.process_prediction`
(código de produção, não alterado) gera internamente um `message_id`
(`uuid4()`) e um timestamp de servidor (`datetime.now()`) — nenhuma
asserção deste arquivo depende do valor exato desses dois campos; RNF-52
exige que as ENTRADAS do teste sejam determinísticas, não que código de
produção pré-existente pare de usar relógio/uuid real internamente. A
medição de RF-26 usa `time.perf_counter()` (tempo relativo decorrido, não
relógio de parede) — legítimo para um gate de regressão de latência.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from src.core.exceptions import MCPUnavailableError, OllamaUnavailableError
from src.core.ws_manager import ConnectionManager
from src.schemas.maintenance import MaintenanceSuggestionRequest
from src.services.alert_service import AlertService
from src.services.critical_failure_notification_service import (
    CriticalFailureNotificationService,
)
from src.services.maintenance_suggestion_service import MaintenanceSuggestionService
from src.tasks.notification_tasks import send_critical_failure_notification_task

# ---------------------------------------------------------------------------
# Fixtures determinísticas — RNF-52
# ---------------------------------------------------------------------------

# Timestamp/equipamento fixos — nunca datetime.now()/uuid4() nas entradas.
FIXED_TIMESTAMP = "2026-09-09T12:00:00+00:00"
EQUIPMENT_NAME = "Bomba Centrífuga CX-500"


def deterministic_sensor_reading() -> dict[str, float]:
    """
    Leitura de sensor determinística — os 12 canais reais do MetroPT-3 (mesmo
    schema de `SensorReading`/`PredictRequest`), representando uma condição
    de degradação capaz de produzir uma probabilidade de falha alta (valores
    fora da faixa normal de operação — pressão baixa, corrente alta).
    Documentação do formato de ENTRADA da detecção; a inferência ONNX em si
    já é coberta pelos testes de ML dedicados (test_model_registry.py,
    test_predict_endpoint.py) — não duplicada aqui.
    """
    return {
        "TP2": -0.012,
        "TP3": 9.34,
        "H1": -0.024,
        "DV_pressure": 9.34,
        "Reservoirs": 9.3,
        "Motor_current": 8.5,  # bem acima do normal (~4) — assinatura de falha
        "Oil_temperature": 71.2,
        "COMP": 0.0,
        "DV_eletric": 0.0,
        "Towers": 1.0,
        "MPG": 0.0,
        "Oil_level": 0.0,
    }


def detection_result(probability: float) -> dict[str, Any]:
    """
    Saída da etapa de Detecção — MESMO contrato exato que
    `InferencePipelineService._process` entrega a
    `AlertService.process_prediction()` em produção (ver
    inference_pipeline.py: ``{"probability", "predicted_class", "timestamp",
    "inference_latency_ms"}``). O modelo ONNX em si é estável/determinístico
    dado o mesmo input, mas não é reinvocado aqui (ver docstring do módulo)
    — este dict representa diretamente o RESULTADO que uma leitura crítica
    real produziria, no formato exato usado pela integração real.
    """
    return {
        "probability": probability,
        "predicted_class": 1 if probability > 0.5 else 0,
        "timestamp": FIXED_TIMESTAMP,
        "inference_latency_ms": 12.3,
    }


# ---------------------------------------------------------------------------
# Fake MCP — mocka no limite real de rede (MCPSearchClient), RF-20/21
# ---------------------------------------------------------------------------

PUMP_MANUAL_RESULT: dict[str, Any] = {
    "text": (
        "Em caso de ruído excessivo na sucção, verificar o estado do "
        "rolamento e a vedação mecânica; substituir se houver folga."
    ),
    "score": 0.82,
    "metadata": {
        "source": "bomba-centrifuga-cx500.pdf",
        "file_name": "bomba-centrifuga-cx500.pdf",
        "page": 4,
        "chunk_index": 2,
    },
}

MOTOR_MANUAL_RESULT: dict[str, Any] = {
    "text": "Lubrificar os rolamentos do motor a cada 2000 horas de operação.",
    "score": 0.75,
    "metadata": {
        "source": "motor-eletrico-me200.pdf",
        "file_name": "motor-eletrico-me200.pdf",
        "page": 2,
        "chunk_index": 0,
    },
}


def fake_mcp_client(
    routes: dict[str, list[dict[str, Any]]], raise_exc: Exception | None = None
) -> AsyncMock:
    """
    Substitui `MCPSearchClient` na fronteira real usada por
    `MaintenanceSuggestionService` — devolve `structured_content` no MESMO
    formato que o mcp-server real produz (ver
    `apps/mcp-server/server.py`/`semantic_search.py`, já validado com
    ChromaDB real em `test_semantic_search.py`). Roteamento por palavra-chave
    na query — determinístico, sem embeddings/rede/modelo.

    `AsyncMock` (não uma classe própria) — mesmo padrão já estabelecido em
    `test_maintenance_suggestion.py` (RF-22): `MaintenanceSuggestionService`
    é tipado estritamente contra `MCPSearchClient`/`OllamaClient` (CLAUDE.md
    §2 — proibido enfraquecer tipos), e `AsyncMock` satisfaz mypy por
    duck-typing onde uma classe fake própria não satisfaria uma checagem
    nominal estrita.
    """
    calls: list[str] = []

    async def _search(query: str) -> dict[str, Any]:
        calls.append(query)
        if raise_exc is not None:
            raise raise_exc
        lowered = query.lower()
        for keyword, results in routes.items():
            if keyword in lowered:
                return {"query": query, "results": results}
        return {"query": query, "results": []}

    mock = AsyncMock()
    mock.search_maintenance_manual = AsyncMock(side_effect=_search)
    mock.calls = calls
    return mock


# ---------------------------------------------------------------------------
# Fake Ollama — mocka no limite real de rede (OllamaClient), RF-22/23
# ---------------------------------------------------------------------------

FIXED_MARKDOWN = """# Plano de Manutenção

## Diagnóstico provável

Possível falha relacionada ao equipamento, compatível com o sintoma relatado.

## Procedimento recomendado

1. Verificar o equipamento conforme o manual.
2. Consultar o procedimento indicado na seção de referências.

## Ferramentas / peças

Não especificado nos trechos recuperados.

## Cuidados de segurança

Nenhum aviso de segurança específico encontrado nos trechos recuperados.

## Referências

- `bomba-centrifuga-cx500.pdf`, página 4
"""

LIMITATION_MARKDOWN = """# Plano de Manutenção

## Limitações

O manual recuperado não contém informações suficientes para determinar com \
segurança o procedimento necessário.
"""


def fake_ollama_client(
    response: str = FIXED_MARKDOWN, raise_exc: Exception | None = None
) -> AsyncMock:
    """Substitui `OllamaClient` (mesmo padrão de `fake_mcp_client` acima) —
    resposta Markdown fixa e conhecida, ou uma exceção controlada para os
    testes de fallback. Registra os prompts recebidos para provar que o LLM
    realmente recebeu o contexto do MCP."""
    calls: list[tuple[str, str]] = []

    async def _generate(system_prompt: str, user_prompt: str) -> str:
        calls.append((system_prompt, user_prompt))
        if raise_exc is not None:
            raise raise_exc
        return response

    mock = AsyncMock()
    mock.generate = AsyncMock(side_effect=_generate)
    mock.calls = calls
    return mock


# ---------------------------------------------------------------------------
# Notificação — mocka no limite real de rede (adapters), RF-24/25 + RNF-50
# ---------------------------------------------------------------------------


def build_notifier(
    enqueue_notification: Any,
) -> tuple[CriticalFailureNotificationService, AsyncMock]:
    """
    `CriticalFailureNotificationService` real, com:
      - `rate_limiter` mockado (sempre adquire) — evita depender de Postgres
        real (RNF-52); o comportamento do rate limiter em si já é coberto
        exaustivamente por `test_notifications.py`/`test_alert_settings.py`,
        não duplicado aqui (item 19 do enunciado desta task).
      - `adapter` (Telegram) mockado — NUNCA deve ser chamado diretamente
        quando `enqueue_notification` está presente (prova de RNF-50: envio
        não é síncrono).
      - `settings_service=None` — usa os defaults do RF-24
        (`CRITICAL_FAILURE_THRESHOLD=0.85`, Telegram sempre ligado) sem
        precisar de Postgres para ler configuração (RF-25).
    """
    rate_limiter = AsyncMock()
    rate_limiter.try_acquire = AsyncMock(return_value=True)
    adapter = AsyncMock()
    notifier = CriticalFailureNotificationService(
        adapter=adapter,
        rate_limiter=rate_limiter,
        dashboard_url="http://localhost",
        enqueue_notification=enqueue_notification,
    )
    return notifier, adapter


# ---------------------------------------------------------------------------
# Teste principal — fluxo completo, caso crítico
# ---------------------------------------------------------------------------


async def test_full_pipeline_critical_prediction() -> None:
    """
    Sensor -> Detecção -> AlertService -> (RF-14 WS + RF-24/25 enqueue) e,
    do mesmo evento, MaintenanceSuggestionService -> MCP -> LLM -> plano
    validado. RF-26: tempo total (excluindo geração real do LLM — mockado,
    retorna imediatamente) < 5s.
    """
    start = time.perf_counter()

    # 1-2. Sensor Data + Detecção — probabilidade determinística e crítica.
    reading = deterministic_sensor_reading()
    assert reading["Motor_current"] > 5.0  # documenta a condição de falha
    detection = detection_result(probability=0.93)

    # 3-5. AlertService -> threshold RF-14 -> RF-24/25 -> RNF-50 enqueue.
    enqueue_mock = Mock()
    notifier, telegram_adapter = build_notifier(enqueue_mock)
    ws_manager = AsyncMock(spec=ConnectionManager)
    alert_service = AlertService(ws_manager, critical_notifier=notifier)

    alert_payload = await alert_service.process_prediction(detection)

    assert alert_payload["triggered"] is True  # RF-14: 0.93 > 0.70
    ws_manager.broadcast_alert.assert_awaited_once()

    enqueue_mock.assert_called_once()
    (enqueued_payload,) = enqueue_mock.call_args.args
    json.dumps(enqueued_payload)  # RNF-50 §6/§9.2 — 100% serializável
    assert enqueued_payload["probability"] == 0.93
    assert enqueued_payload["telegram_enabled"] is True
    # RNF-50 — a notificação NUNCA é enviada de forma síncrona: o adapter
    # nunca é chamado diretamente pelo processo que decidiu o alerta.
    telegram_adapter.send_critical_failure.assert_not_awaited()

    # 6-9. Maintenance Suggestion (ramo independente, mesma detecção) —
    # MCP/RF-20-21 -> contexto -> LLM/RF-22.
    mcp = fake_mcp_client(
        {"bomba": [PUMP_MANUAL_RESULT], "motor": [MOTOR_MANUAL_RESULT]}
    )
    ollama = fake_ollama_client()
    suggestion_service = MaintenanceSuggestionService(
        mcp_client=mcp, ollama_client=ollama, model="llama3.2:3b"
    )
    request = MaintenanceSuggestionRequest(
        failure_probability=detection["probability"],
        equipment_name=EQUIPMENT_NAME,
        symptom_description="ruído excessivo na sucção da bomba",
    )

    response = await suggestion_service.suggest(request)

    # 8. Contexto MCP realmente chegou ao LLM — não é um passo isolado que
    # "passa" sem estar de fato conectado ao próximo (item 18 do enunciado).
    assert mcp.calls == ["Bomba Centrífuga CX-500: ruído excessivo na sucção da bomba"]
    assert len(ollama.calls) == 1
    _, user_prompt = ollama.calls[0]
    assert PUMP_MANUAL_RESULT["text"] in user_prompt
    assert "bomba-centrifuga-cx500.pdf" in user_prompt
    assert MOTOR_MANUAL_RESULT["text"] not in user_prompt  # contexto do motor não vazou

    # 9-10. LLM mock retornou, Markdown validado.
    assert response.triggered is True
    assert response.markdown == FIXED_MARKDOWN
    assert len(response.references) == 1
    assert response.references[0].file_name == "bomba-centrifuga-cx500.pdf"
    assert response.references[0].score == pytest.approx(0.82)

    # 11-13. Notification Service já provado nos passos 3-5 acima (mesmo
    # evento) — nenhum HTTP externo real foi executado em nenhum dos dois
    # ramos (adapters/OllamaClient/MCPSearchClient reais nunca instanciados
    # neste teste).

    elapsed = time.perf_counter() - start
    # RF-26 — < 5s desconsiderando geração REAL do LLM (aqui mockado,
    # retorna instantaneamente); o gate mede a orquestração
    # (threshold -> rate limit -> enqueue -> MCP -> montagem de prompt ->
    # validação), não uma promessa de latência de produção com Ollama real.
    assert elapsed < 5.0, f"pipeline levou {elapsed:.3f}s, esperado < 5s"


def test_enqueued_payload_is_consumed_by_the_real_celery_task(monkeypatch) -> None:
    """
    Fecha o loop pipeline -> enqueue -> task -> adapter (RNF-50, item 19):
    usa o payload EXATO que o passo 3-5 acima produziria e executa a task
    Celery real diretamente (sem broker/worker reais — RNF-52), provando que
    o payload enfileirado é utilizável pelo consumidor real.

    Função SÍNCRONA de propósito (não `async def`): a task Celery real faz
    `asyncio.run(...)` internamente (ver `notification_tasks.py`) — chamar
    isso de dentro de um teste `async def` (já rodando sob o event loop do
    pytest-asyncio) levantaria `RuntimeError: asyncio.run() cannot be
    called from a running event loop`, exatamente como um worker Celery
    real (processo síncrono) chamaria. O setup assíncrono (`process_prediction`)
    roda seu próprio `asyncio.run()` isolado, fora do event loop da task.
    """
    enqueued_payloads: list[dict[str, Any]] = []
    notifier, _ = build_notifier(lambda payload: enqueued_payloads.append(payload))
    ws_manager = AsyncMock(spec=ConnectionManager)
    alert_service = AlertService(ws_manager, critical_notifier=notifier)

    asyncio.run(alert_service.process_prediction(detection_result(probability=0.93)))
    assert len(enqueued_payloads) == 1

    telegram_mock = AsyncMock()
    telegram_mock.send_critical_failure = AsyncMock(return_value=None)
    worker_service = CriticalFailureNotificationService(
        adapter=telegram_mock,
        rate_limiter=AsyncMock(),
        dashboard_url="http://localhost",
    )
    monkeypatch.setattr(
        "src.tasks.notification_tasks._build_service_for_worker", lambda: worker_service
    )

    send_critical_failure_notification_task(enqueued_payloads[0])

    telegram_mock.send_critical_failure.assert_awaited_once()


# ---------------------------------------------------------------------------
# Thresholds — RF-14 (0.70) / RF-22 (0.70) / RF-24 (0.85) são independentes
# ---------------------------------------------------------------------------


async def test_below_maintenance_threshold_skips_mcp_llm_and_notification() -> None:
    """0.65 — abaixo de TODOS os limiares (RF-14/RF-22 em 0.70, RF-24 em
    0.85): nada dispara em nenhum dos dois ramos."""
    enqueue_mock = Mock()
    notifier, telegram_adapter = build_notifier(enqueue_mock)
    ws_manager = AsyncMock(spec=ConnectionManager)
    alert_service = AlertService(ws_manager, critical_notifier=notifier)

    alert_payload = await alert_service.process_prediction(detection_result(0.65))
    assert alert_payload["triggered"] is False
    ws_manager.broadcast_alert.assert_not_awaited()
    enqueue_mock.assert_not_called()
    telegram_adapter.send_critical_failure.assert_not_awaited()

    mcp = fake_mcp_client({"bomba": [PUMP_MANUAL_RESULT]})
    ollama = fake_ollama_client()
    suggestion_service = MaintenanceSuggestionService(mcp, ollama, "llama3.2:3b")
    response = await suggestion_service.suggest(
        MaintenanceSuggestionRequest(
            failure_probability=0.65, equipment_name=EQUIPMENT_NAME
        )
    )
    assert response.triggered is False
    assert mcp.calls == []
    assert ollama.calls == []


async def test_between_maintenance_and_critical_threshold_suggests_but_does_not_notify() -> (
    None
):
    """
    0.80 — acima do limiar de sugestão de manutenção (RF-22, `> 0.70`) MAS
    abaixo do limiar de falha crítica (RF-24, `> 0.85`).

    Nota de auditoria: o enunciado original desta task usa 0.80 como exemplo
    de "abaixo do threshold" partindo da premissa de um único limiar único
    para todo o pipeline. A auditoria obrigatória (item 1) confirmou que
    RF-22 e RF-24 usam CONSTANTES DIFERENTES e INDEPENDENTES
    (`MAINTENANCE_SUGGESTION_THRESHOLD=0.70`, `CRITICAL_FAILURE_THRESHOLD=0.85`
    — documentado desde RF-24/RF-25). Este teste reflete o comportamento
    REAL do projeto (item 13 do enunciado: "use o threshold real configurado
    pelo projeto, sem hardcode incorreto") em vez de forçar uma premissa
    equivocada: em 0.80, a sugestão de manutenção (MCP/LLM) É acionada, mas
    a notificação crítica NÃO é enfileirada.
    """
    enqueue_mock = Mock()
    notifier, telegram_adapter = build_notifier(enqueue_mock)
    ws_manager = AsyncMock(spec=ConnectionManager)
    alert_service = AlertService(ws_manager, critical_notifier=notifier)

    alert_payload = await alert_service.process_prediction(detection_result(0.80))
    assert alert_payload["triggered"] is True  # RF-14: 0.80 > 0.70
    ws_manager.broadcast_alert.assert_awaited_once()
    enqueue_mock.assert_not_called()  # RF-24: 0.80 <= 0.85 — não enfileira
    telegram_adapter.send_critical_failure.assert_not_awaited()

    mcp = fake_mcp_client({"bomba": [PUMP_MANUAL_RESULT]})
    ollama = fake_ollama_client()
    suggestion_service = MaintenanceSuggestionService(mcp, ollama, "llama3.2:3b")
    response = await suggestion_service.suggest(
        MaintenanceSuggestionRequest(
            failure_probability=0.80,
            equipment_name=EQUIPMENT_NAME,
            symptom_description="ruído excessivo na sucção da bomba",
        )
    )
    assert response.triggered is True  # RF-22: 0.80 > 0.70 — MCP/LLM SÃO chamados
    assert mcp.calls
    assert ollama.calls


# ---------------------------------------------------------------------------
# Fallbacks — item 11/12 do enunciado
# ---------------------------------------------------------------------------


async def test_fallback_llm_unavailable_raises_structured_error_no_fabricated_plan() -> (
    None
):
    """
    RF-22/23 real (auditado): `MaintenanceSuggestionService.suggest()` NÃO
    captura `OllamaUnavailableError` — propaga para o router, que deixa o
    `AppError` handler global (main.py) responder 503. Nenhum plano
    fabricado é devolvido; o teste confirma exatamente esse comportamento
    real, não um comportamento inventado. Confirma também que o MCP já
    tinha sido consultado com sucesso antes da falha (a falha é isolada ao
    LLM, não ao pipeline inteiro).
    """
    mcp = fake_mcp_client({"bomba": [PUMP_MANUAL_RESULT]})
    ollama = fake_ollama_client(
        raise_exc=OllamaUnavailableError(
            "Serviço de geração de sugestões indisponível."
        )
    )
    service = MaintenanceSuggestionService(mcp, ollama, "llama3.2:3b")
    request = MaintenanceSuggestionRequest(
        failure_probability=0.90,
        equipment_name=EQUIPMENT_NAME,
        symptom_description="ruído excessivo na sucção da bomba",
    )

    with pytest.raises(OllamaUnavailableError):
        await service.suggest(request)

    assert mcp.calls  # chegou até o LLM — a falha não é de conectividade MCP


async def test_fallback_mcp_unavailable_raises_structured_error() -> None:
    """Mesmo raciocínio acima, para o MCP indisponível (não coberto
    explicitamente pelo enunciado, mas simétrico e barato de confirmar)."""
    mcp = fake_mcp_client(
        {}, raise_exc=MCPUnavailableError("Não foi possível conectar ao servidor MCP.")
    )
    ollama = fake_ollama_client()
    service = MaintenanceSuggestionService(mcp, ollama, "llama3.2:3b")
    request = MaintenanceSuggestionRequest(
        failure_probability=0.90, equipment_name=EQUIPMENT_NAME
    )

    with pytest.raises(MCPUnavailableError):
        await service.suggest(request)

    assert ollama.calls == []  # nunca gasta uma chamada de LLM sem contexto


async def test_fallback_no_manual_indexed_llm_still_called_with_empty_context() -> None:
    """
    Chroma vazio / nenhum manual indexado -> MCP real devolveria
    `{"results": []}` (RF-21: nenhum candidato ultrapassa o threshold de
    similaridade). Auditoria confirmou que `MaintenanceSuggestionService`
    NÃO tem um curto-circuito para contexto vazio — `_build_prompt` insere
    o placeholder "(Nenhum trecho de manual relevante foi recuperado...)" e
    o Ollama É chamado mesmo assim (a System Prompt, Regra 4, instrui o LLM
    a declarar isso na seção "Limitações" em vez de inventar). Este teste
    confirma exatamente esse comportamento real — não um "não chamar o LLM"
    que o código não implementa.
    """
    mcp = fake_mcp_client({})  # nenhuma palavra-chave nunca combina -> []
    ollama = fake_ollama_client(response=LIMITATION_MARKDOWN)
    service = MaintenanceSuggestionService(mcp, ollama, "llama3.2:3b")
    request = MaintenanceSuggestionRequest(
        failure_probability=0.90,
        equipment_name="Equipamento sem manual indexado",
        symptom_description="falha não catalogada",
    )

    response = await service.suggest(request)

    assert response.triggered is True
    assert response.references == []  # nenhuma referência inventada
    assert len(ollama.calls) == 1
    _, user_prompt = ollama.calls[0]
    assert "Nenhum trecho de manual relevante" in user_prompt
    assert response.markdown == LIMITATION_MARKDOWN
