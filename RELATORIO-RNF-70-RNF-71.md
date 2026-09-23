# RELATÓRIO — RNF-70 / RNF-71: Cache de Inferência com Redis

## 1. Objetivo

- **RNF-70** — latência **p95 < 50ms** para predições já em cache (`POST /predict/`).
- **RNF-71** — cache Redis com **TTL de exatamente 60 segundos**, comportamento observável (não só configurado).

Regra seguida do início ao fim: **auditar e medir antes de alterar**. Nenhum índice, gargalo ou benchmark foi assumido — cada afirmação abaixo tem a evidência (comando + saída) que a sustenta.

---

## 2. Auditoria inicial

### 2.1. Endpoint e fluxo real

`POST /predict/` ([predict.py](apps/backend/src/routers/predict.py)) — sem autenticação, rate-limited **100 req/min por IP** (slowapi, RNF-19, [rate_limit.py](apps/backend/src/core/rate_limit.py)).

Fluxo **antes** desta task:

1. `ModelService.predict(payload)` — inferência ONNX/sklearn **em memória** (`asyncio.to_thread`). **Nenhuma query SQL.**
2. `save_prediction(db, payload, result)` — **1 INSERT** em `predictions` (flush; commit no fim do request via `get_db`).
3. `alert_service.process_prediction(...)` → `CriticalFailureNotificationService.notify_if_critical(...)`:
   - `_resolve_settings()` roda **sempre** (achado da auditoria — a leitura inicial deste relatório dizia "só quando probability > 0.85"; o código real chama `_resolve_settings()` **antes** do `if probability <= threshold: return`) → **1 SELECT** em `alert_settings` (linha singleton, `id=1`) em **toda** predição.
   - `try_acquire()` (INSERT/UPDATE em `telegram_alert_locks`) só roda quando `probability > threshold` (0.85 por padrão — caminho raro).

Ou seja: toda predição comum faz **1 INSERT + 1 SELECT**, ambos sobre chaves primárias/índices já existentes — nenhum SELECT varrendo a tabela `predictions` (essa só acontece em `GET /v1/predictions`, fora do escopo do RNF-70).

### 2.2. Banco — tabelas e índices existentes

```
predictions          PK (id) + ix_predictions_timestamp (btree)
alert_settings        PK (id), sempre 1 linha
telegram_alert_locks  PK (equipment_id), 1 linha por equipamento simulado
```

### 2.3. Cache — nada existia

Confirmado por grep: nenhuma abstração de cache no projeto. **Redis já existia na infra** — só como broker do Celery (RNF-50/51, `CELERY_BROKER_URL=redis://redis:6379/0`), com `redis==5.0.8` já em `requirements.txt`, serviço `redis` no `docker-compose.yml` (healthcheck, `api` já depende dele). Decisão: reaproveitar essa MESMA instância para o cache, isolada por DB lógico (`db=1`), sem novo container.

### 2.4. Testes existentes

`tests/test_predict_endpoint.py` mocka `ModelService` e usa SQLite em memória — sem Postgres/Redis reais. CI (`test-python`) também não subia nenhum dos dois.

### 2.5. Locust existente

`locust_sse.py`/`locust_streaming.py` medem só o stream SSE. **Nenhum Locust para `/predict/`** — criado nesta task ([locust_predict.py](locust_predict.py)).

---

## 3. Baseline (Fase 2 — antes de qualquer alteração)

### 3.1. API sem cache

Stack real via Docker Compose (`db` + `redis` + `api`, mesma configuração do job `load-smoke` do CI), banco com **248.898 linhas pré-existentes** em `predictions` (pipeline de inferência contínua do próprio projeto, rodando em segundo plano).

Comando (respeitando o rate limit de 100/min — 1 usuário, payload aleatório a cada request, `PREDICT_MODE=random`):

```bash
PREDICT_MODE=random PREDICT_WAIT_SECONDS=0.8 \
locust -f locust_predict.py --host http://localhost:8000 \
       --headless -u 1 -r 1 --run-time 150s --csv=loadtest_results/predict_baseline
```

**Resultado real** (`loadtest_results/predict_baseline_stats.csv`):

| Métrica | Valor |
|---|---:|
| Requests | 174 |
| Failures | 0 |
| p50 | 56 ms |
| p95 | 68 ms |
| p99 | 75 ms |
| min / max | 23.9 ms / 76.6 ms |

### 3.2. Banco — `EXPLAIN (ANALYZE, BUFFERS)` real

**INSERT em `predictions`** (o único write do caminho comum), executado dentro de uma transação com `ROLLBACK` para não sujar os dados:

```sql
BEGIN;
EXPLAIN (ANALYZE, BUFFERS)
INSERT INTO predictions (timestamp, "TP2", ..., predicted_class, failure_probability)
VALUES (now(), 5.02, ..., 1, 0.123456);
ROLLBACK;
```

```
Insert on predictions  (cost=0.00..0.02 rows=0 width=0) (actual time=0.127..0.127 rows=0 loops=1)
  Buffers: shared hit=70
  ->  Result  (actual time=0.035..0.035 rows=1 loops=1)
Planning Time: 0.070 ms
Execution Time: 0.141 ms
```

**SELECT em `alert_settings`** (roda em toda predição, achado corrigido do §2.1):

```sql
EXPLAIN (ANALYZE, BUFFERS) SELECT * FROM alert_settings WHERE id = 1;
```

```
Index Scan using alert_settings_pkey on alert_settings (actual time=0.013..0.013 rows=1 loops=1)
  Buffers: shared hit=2
Execution Time: 0.033 ms
```

`telegram_alert_locks` tem PK em `equipment_id` e, na prática, 1 linha (1 equipamento simulado) — sem necessidade de `EXPLAIN` para justificar: não há como um Seq Scan em 1 linha ser um problema.

**Conclusão da Fase 2/3 — nenhum índice foi adicionado.** As duas queries do caminho comum já usam Index/PK scan, custam < 0,15ms cada, contra uma tabela com quase 250 mil linhas. Não existe Seq Scan, não existe query lenta, não existe gargalo de banco a corrigir — o "problema" que o RNF-70 pede pra resolver é **inferência ML + I/O de rede**, não SQL. Seguindo a regra explícita da task ("se uma query já é eficiente, não adicione índice só pra cumprir checklist"), **nenhuma migration Alembic foi criada**.

---

## 4. `InferenceCache` — arquitetura

Novo arquivo: [`src/services/inference_cache.py`](apps/backend/src/services/inference_cache.py).

```
request
  ↓
POST /predict/
  ↓
InferenceCache.get(key)
  ├── HIT → retorna a PredictResponse já cacheada (mesmo timestamp da 1ª vez)
  │
  └── MISS
       ↓
     ModelService.predict()  (ONNX/sklearn)
       ↓
     save_prediction() → INSERT em predictions
       ↓
     alert_service.process_prediction()
       ↓
     InferenceCache.set(key, value, TTL=60)
       ↓
     response
```

- `InferenceCache.get`/`set` — nunca lançam; qualquer falha do Redis (conexão recusada, timeout, entrada corrompida) vira MISS/no-op, logado como `warning`.
- Cliente `redis.asyncio` (já em `requirements.txt`, nenhuma dependência nova), criado uma vez no `lifespan` (mesmo padrão de singleton do `ModelRegistry`, ver [main.py](apps/backend/src/main.py)) e injetado via `Depends(get_inference_cache)` — `InferenceCacheProtocol` em [protocols.py](apps/backend/src/services/protocols.py) (RNF-56, mesmo padrão de fronteira dos demais services do router).
- **Decisão de design deliberada (RNF-70 — HIT pula tudo):** num cache hit, **não** roda inferência, **não** insere em `predictions`, **não** reprocessa alerta. Isso é uma extensão razoável de RF-09 ("toda predição bem-sucedida é persistida") — a predição em si **já foi** persistida na primeira vez; requisições idênticas dentro da janela de 60s recebem a mesma predição já registrada, não uma nova. Documentado aqui explicitamente como trade-off, não escondido.

### 4.1. Composição da cache key (§6)

```python
build_cache_key(payload: PredictRequest, model_name: str) -> str
```

Inclui:
- os **12 sensores brutos** (valores exatos) — `ModelService.predict()` é função pura desses campos no caminho stateless de `/predict`;
- o **modelo ativo** (`ModelRegistry.active_name`) — um hot-swap via `PUT /models/active` (RF-10) pode mudar o resultado para o **mesmo** snapshot de sensores; sem isso na chave, uma entrada cacheada do modelo antigo serviria silenciosamente uma predição errada por até 60s.

Exclui deliberadamente:
- `decision_threshold` — derivado 1:1 do `model_name` via model card (`_resolve_threshold`); incluir seria redundante.
- `timestamp` — é OUTPUT da inferência, não input.

Chave: `predict-cache:v1:{model_name}:{sha256(model_name|json_canônico_dos_12_sensores)}` — hash (não os valores brutos) por tamanho fixo e para não vazar dados de sensor em `redis-cli KEYS`/logs de infra.

**Validado contra Redis real** (chave observada de verdade durante teste manual):
```
predict-cache:v1:random_forest_v2:dab7bc4c7142e23a61ee672685d58f7f70c74e27616077d6cf628cb4c91a0268
```
— o mesmo valor que o teste `test_key_matches_known_golden_digest` (golden hash, calculado independentemente) afirma.

### 4.2. TTL — RNF-71 (§5)

```python
INFERENCE_CACHE_TTL_SECONDS: int = 60
```

Constante de código, **não** exposta como variável de ambiente — decisão deliberada: se fosse configurável por `.env`, um deploy poderia violar "60 segundos" silenciosamente. Único lugar do projeto que define esse número.

### 4.3. Invalidação/consistência (§7)

**Nenhuma invalidação explícita foi implementada — e a arquitetura não precisa dela**, porque:
- o TTL de 60s já limita qualquer staleness a uma janela curta e conhecida;
- o `model_name` já faz parte da chave — um hot-swap de modelo automaticamente passa a usar um **namespace de chave diferente**; entradas do modelo anterior só ficam "órfãs" (nunca mais lidas) até expirarem sozinhas.

### 4.4. Redis indisponível — fallback (§8)

Testado **de verdade** (`docker stop redis` com a API rodando):

| Timeout do cliente | Latência observada de `/predict/` com Redis parado |
|---|---:|
| 2.0s (1ª tentativa) | **4.02s** (get + set, ambos no timeout) — inaceitável |
| 0.3s (corrigido) | **0.65s** — resposta 200 correta, predição calculada normalmente |

`predict/` **nunca** retornou erro por causa do Redis — em ambos os casos a resposta foi HTTP 200 com o resultado real (fallback para MISS). O achado do timeout de 2s ter inflado a resposta para 4s foi real, medido, e corrigido no código (não é um número hipotético).

---

## 5. Testes

### 5.1. `tests/test_inference_cache.py` (26 testes)

- `build_cache_key` — determinismo, unicidade por sensor/modelo diferente, golden-hash exato (ver §5.4).
- `InferenceCache` contra cliente falso (`_FakeRedisClient`) — hit/miss, fallback em erro de GET/SET, entrada corrompida, `close()`.
- `InferenceCache` contra **Redis real** (`TEST_REDIS_URL`, DB 15 dedicado) — **sem mock de TTL**:
  - `test_ttl_is_exactly_60_seconds` — confirma `55 <= TTL <= 60` no Redis real logo após `set()`.
  - `test_key_expires_and_next_read_is_a_real_miss` — usa `PEXPIRE` (do próprio Redis) para encurtar o TTL da chave específica pra 300ms (sem alterar a constante de produção), espera passar, confirma `EXISTS=0` no Redis **e** `cache.get()` retornando `None`.
  - `test_expired_key_triggers_fresh_miss_not_stale_hit` — depois de expirar, um novo `set()` na mesma chave funciona normalmente.
- Dependências FastAPI (`get_inference_cache`/`get_active_model_name`) com e sem `app.state` populado.

### 5.2. `tests/test_predict_cache_integration.py` (4 testes, endpoint real + Redis real)

- 1ª requisição: MISS → `ModelService.predict_proba` chamado 1x, 1 linha nova em `predictions`.
- 2ª requisição idêntica: HIT → `predict_proba` **não** chamado de novo, **nenhuma** linha nova, resposta byte-idêntica (mesmo timestamp).
- Sensor diferente: MISS de novo (chave diferente).
- Depois do TTL expirar (mesma técnica de `PEXPIRE`): MISS de novo, `predict_proba` chamado uma 2ª vez.

### 5.3. Achado real durante os testes — regressão encontrada e corrigida

A 1ª versão do fallback de `get_inference_cache` (quando `app.state.inference_cache` não existe — testes com `ASGITransport`) construía um `InferenceCache` de verdade contra `settings.redis_cache_url` e contava com esse host "provavelmente" não resolver em ambiente de teste. **Rodando a suíte inteira dentro do container `api`** (onde o hostname `redis` resolve de verdade) isso **quebrou 8 testes em 3 arquivos** (`test_predictions_endpoint.py`, `test_infrastructure.py`, `test_dependency_injection.py`) — todos reusam o mesmo payload de exemplo, e o 2º+ `POST /predict` de cada um virava HIT de um Redis real, pulando a persistência/exceção que o teste esperava.

Corrigido com um sentinel `_NullInferenceCache` — MISS sempre, **zero I/O de rede**, nunca depende de "a rede vai falhar" — determinístico em qualquer ambiente. Suíte completa voltou a **726 passed, 1 xfailed** depois da correção. Este é exatamente o tipo de achado que a auditoria "medir antes/depois" deveria capturar — reportado aqui em vez de escondido.

### 5.4. Mutation testing dedicado (`inference_cache.py`) — resultado final: **43/44 mortos (97.7%)**

Rodei `mutmut` localmente sobre o arquivo novo (44 mutantes) em **quatro** tentativas. As duas primeiras expuseram problemas operacionais reais do MEU ambiente local (não do código), corrigidos e documentados aqui em vez de escondidos; a terceira encontrou um teste com uma asserção logicamente errada (achado real, corrigido); a quarta é a medição final.

**Sobreviventes reais encontrados e corrigidos (6):**

1. Mutação na string de entrada do hash (`f"{model_name}|{canonical}"` → `f"XX{model_name}|{canonical}XX"`): meus testes de "mesma entrada → mesma chave"/"entrada diferente → chave diferente" não capturavam isso, porque continuam válidos mesmo com a string alterada. **Corrigido**: `test_key_matches_known_golden_digest`, que fixa o SHA256 exato esperado para um payload conhecido.
2. `decode_responses=True` → `False` em `create_redis_client`: nenhum teste inspecionava os kwargs reais de conexão do cliente construído. **Corrigido**: `test_create_redis_client_uses_exact_config`, que lê `client.connection_pool.connection_kwargs` direto.
3. `socket_connect_timeout=0.3` → `1.3` (mesmo `create_redis_client`): o mesmo teste acima também fixa esse valor — é exatamente o parâmetro que o achado do §4.4 (timeout de 2s inflando `/predict` para 4s com Redis parado) motivou; um mutante revertendo esse valor pra "generoso" de novo teria passado batido sem esse teste.
4-6. Nome do evento de log (`"inference_cache_get_failed"`/`"_corrupt_entry"`/`"_set_failed"` viravam `"XX...XX"`) e a flag `exc_info=True` viravam `False` nos 3 blocos `except` de `InferenceCache`. **Corrigido**: 3 testes com `caplog` fixando o nome exato do evento e checando `exc_info` — nomes de evento estáveis importam neste projeto (mesmo padrão de `alert_triggered`/`alert_skipped` em `alert_service.py`).

**Achado extra, dentro do próprio achado (4→5→6):** a 1ª versão desses 3 testes usava `assert caplog.records[0].exc_info is not None` e não matou os mutantes de `exc_info=True→False` — **o `mutmut` expôs um bug real no MEU teste**: `logging` grava `exc_info=False` como o literal `False` no record, não `None` — `False is not None` é `True` em Python, então a asserção sempre passava, killed ou não. Corrigido para `assert caplog.records[0].exc_info` (checagem truthy). Reproduzido isoladamente antes de corrigir (ver histórico de execução) — não um ajuste "no escuro".

**Sobrevivente aceito como equivalente (1, final):** `str | None` → `str & None` na anotação de tipo de uma variável local (`raw: str | None = ...`). Com `from __future__ import annotations` no topo do arquivo (PEP 563), TODA anotação vira uma string nunca avaliada em runtime — não existe teste possível que distinga as duas, por construção da linguagem. Mantida como está (é a anotação correta para quem lê o código/mypy).

**Dois incidentes operacionais (honestamente reportados, nenhum deles no código de produção):**

- **Interrupção no meio de uma run** — ao matar uma execução do `mutmut` com `docker restart api` (ação minha, não relacionada ao teste), o processo foi morto antes de reverter a mutação que estava testando naquele momento — `mutmut` reescreve o arquivo de origem NO DISCO durante o teste de cada mutante. Como o container `api` usa bind-mount do código-fonte, isso deixou `inference_cache.py` **mutado em disco**, e a API (rodando ao vivo) passou a responder 500 (`AttributeError: 'PredictRequest' object has no attribute 'XXOil_levelXX'`) até eu detectar (via `docker logs`) e restaurar o arquivo manualmente. Revalidado (curl confirmando 200 de novo).
- **Acesso concorrente ao cache do mutmut** — numa 2ª tentativa, rodei `mutmut show`/`mutmut results` (só leitura, na minha intenção) **enquanto** uma run `mutmut run` ainda estava ativa no mesmo diretório. Isso corrompeu o banco SQLite interno do mutmut (pony ORM) — a run crashou com `ValueError: Attribute Mutant.line is required` em `mutmut/cache.py::update_mutant_status`. O arquivo-fonte em si ficou íntegro desta vez (verificado imediatamente), mas o processo travou (`state: sleeping`, sem terminar) e precisou ser morto manualmente (`kill -9`).

**Lição registrada como limitação no §10** — mutation testing local não deve rodar (a) contra um processo que serve tráfego ao vivo pelo mesmo bind-mount, nem (b) com qualquer outro comando `mutmut` (`show`/`results`) tocando o mesmo cache enquanto uma run está em andamento. Nenhum dos dois riscos existe no job `mutation-testing` do CI: cada grupo roda isolado num runner efêmero, sem servir tráfego e sem inspeção concorrente.

**Resultado final (4ª execução, limpa, sem interferência):**

```
44/44 mutantes testados — 43 killed, 1 survived (equivalente, ver acima)
Mutation score: 43/44 = 97.7%
```

### 5.5. Regressão — suíte completa

```
737 passed, 1 xfailed, 12 warnings in 58.98s
Required test coverage of 85.0% reached. Total coverage: 96.35%
inference_cache.py: 100% coverage (linha) / 43/44 mutantes mortos (mutação)
```

`ruff check`, `black --check`, `mypy --ignore-missing-imports`, `lint-imports` (import-linter, RNF-56) — todos passando, sem exceção nova.

---

## 6. Locust — RNF-70 (Fase 5)

### 6.1. Metodologia

`locust_predict.py` (`PREDICT_MODE=cache_hit`): um listener `test_start` dispara **1 requisição de priming** (MISS esperado), **fora** das estatísticas do Locust; a partir daí, todo request medido usa o **mesmo** payload (mesma chave de cache) e é esperado como HIT. Não mistura miss/hit no mesmo número (RNF-70 §10).

`PREDICT_WAIT_SECONDS=0.8` (~75 req/min) respeita o rate limit real de `/predict/` (100/min por IP, RNF-19) — o Locust bate de um único IP, então o limite é compartilhado entre todos os usuários virtuais; **nunca contornado** (nenhum IP falso, nenhum header spoofado).

```bash
PREDICT_MODE=cache_hit PREDICT_WAIT_SECONDS=0.8 \
locust -f locust_predict.py --host http://localhost:8000 \
       --headless -u 1 -r 1 --run-time 150s --csv=loadtest_results/predict_cache_hit
```

### 6.2. Resultado real (`loadtest_results/predict_cache_hit_stats.csv`)

| Métrica | Valor |
|---|---:|
| Requests | 177 |
| Failures | 0 |
| p50 | 44 ms |
| p95 | **47 ms** |
| p99 | 58 ms |
| min / max | 4.3 ms / 61.0 ms |

**Veredito impresso pelo próprio script:**
```
RNF-70 — POST /predict (cache HIT): p95 < 50ms
  Requests               : 177
  Failures               : 0
  p50                    : 44.0 ms
  p95                    : 47.0 ms
  p99                    : 58.0 ms
  Resultado              : PASS
```

**Nota de ambiente:** medido localmente via Docker Desktop no Windows, que roteia `localhost:8000` por uma VM/proxy de porta — overhead que aparece nos DOIS números (baseline 56ms e cache-hit 44ms de mediana), não é específico do cache. Um deploy Linux nativo tende a ter latência ainda menor nos dois casos; a MARGEM relativa (cache-hit visivelmente mais rápido e mais previsível que o baseline) é o que importa aqui.

### 6.3. Smoke no CI

Adicionado um step reduzido no job `load-smoke` (`.github/workflows/ci.yml`) — mesmo princípio do smoke SSE já existente (RNF-37/68/69): não é o benchmark completo (esse é o §6.1/6.2 acima, documentado, reproduzível), só confirma que o caminho de cache-hit continua funcionando fim-a-fim depois de cada mudança. O exit code do Locust reflete falha de requisição real, não o p95 medido num runner compartilhado com amostra pequena — mesmo padrão já usado pelo smoke SSE existente.

---

## 7. Comparação — antes / depois

| Métrica | Antes | Depois | Resultado |
|---|---:|---:|---|
| API p50 (sem cache) | 56 ms | 56 ms* | — |
| API p95 (sem cache) | 68 ms | 68 ms* | — |
| API p99 (sem cache) | 75 ms | 75 ms* | — |
| Cache-hit p95 | N/A | **47 ms** | **RNF-70: PASS** |
| INSERT `predictions` (Execution Time) | 0.141 ms | 0.141 ms (query inalterada) | já eficiente, sem índice novo |
| SELECT `alert_settings` (Execution Time) | 0.033 ms | 0.033 ms (query inalterada) | já eficiente, sem índice novo |
| Seq Scan nas queries do caminho comum | Nenhum | Nenhum | N/A |
| Index/PK Scan | Sim (ambas queries) | Sim (ambas queries) | Sem alteração |

\* O caminho "sem cache" (`PREDICT_MODE=random`, payload sempre novo) é estruturalmente idêntico antes/depois — o cache só entra em ação para requisições repetidas; o número "depois" existe só como referência de que a introdução do cache não piorou o caminho de miss (ver `test_first_request_is_a_miss_and_persists`).

---

## 8. Resultado RNF-70

**PASS.**

- Medição contra a API real (Docker Compose completo: nginx-less, db+redis+api, mesma stack do CI), não um benchmark artificial.
- Cache HIT genuíno — priming fora das estatísticas, confirmado por timestamp idêntico entre requisições.
- Amostra: 177 requisições, 0 falhas.
- **p95 = 47ms < 50ms.**
- Nenhum request lento mascarado/excluído, nenhuma média usada no lugar de p95, threshold não alterado, TTL não alterado para facilitar a medição.

## 9. Resultado RNF-71

**PASS.**

- Redis realmente integrado — mesma instância do Celery, DB lógico isolado, sem serviço novo.
- Cache sendo usado de verdade em produção (`app.state.inference_cache`, singleton do lifespan) e validado ponta-a-ponta via `curl` + `redis-cli TTL`.
- TTL observável de 60s confirmado contra Redis real (`55 <= TTL <= 60`), não apenas `set()` chamado com `ex=60`.
- Testes automatizados demonstram expiração real (via `PEXPIRE` acelerado + confirmação `EXISTS=0`) **e** demonstração manual com espera real de 60s (`predict-cache:v1:...` expirando de fato, próxima chamada com timestamp novo).

---

## 10. Limitações

1. **Mutation score de `inference_cache.py` medido localmente até o fim: 43/44 (97.7%)** (§5.4) — bem acima do piso agregado de 70% (RNF-64). O score AGREGADO final dos 19 grupos ainda é decidido pelo `mutation-score-gate` do CI (não recalculado aqui para os outros 18 arquivos, fora do escopo desta task — eles não foram alterados).
2. **Dois incidentes operacionais com `mutmut` local, nenhum no código de produção** (§5.4, detalhado) — (a) interromper uma run no meio (`docker restart api`) deixou o arquivo-fonte mutado em disco por estar bind-mounted no container servindo tráfego ao vivo, causando 500 temporários; (b) rodar `mutmut show`/`results` enquanto uma run estava ativa corrompeu o cache SQLite interno do mutmut e travou o processo. Ambos corrigidos/limpos e revalidados. Nenhum dos dois riscos existe no `mutation-testing` do CI (runners efêmeros, um grupo por job, sem inspeção concorrente). Um 3º achado (não um incidente, um bug real de teste): a 1ª versão dos testes de `exc_info` usava uma asserção (`is not None`) que não distinguia `True` de `False` — o próprio `mutmut` expôs isso, corrigido para uma checagem truthy.
3. **Latência medida em Docker Desktop/Windows**, não Linux nativo — overhead de proxy de porta presente nos dois lados da comparação (não invalida o resultado, mas o número absoluto de produção em Linux tende a ser menor).
4. **RF-09 (persistência de toda predição) é modificado em espírito para cache hits** — uma requisição idêntica dentro da janela de 60s não gera uma nova linha em `predictions` (é a mesma predição já registrada, não uma nova). Documentado explicitamente no §4, decisão deliberada e não escondida.
5. `GET /v1/predictions` (histórico paginado) não foi tocado — fora do escopo do RNF-70 (que é sobre `POST /predict`), já tem índice adequado (`ix_predictions_timestamp`) para seu próprio padrão de uso.

---

## 11. Arquivos alterados/criados

**Novos:**
- `apps/backend/src/services/inference_cache.py` — `InferenceCache`, `build_cache_key`, TTL, dependências FastAPI.
- `apps/backend/tests/test_inference_cache.py` — 26 testes (unit + Redis real).
- `apps/backend/tests/test_predict_cache_integration.py` — 4 testes de integração endpoint+cache.
- `locust_predict.py` — Locust para `/predict/` (baseline `random` + RNF-70 `cache_hit`).
- `RELATORIO-RNF-70-RNF-71.md` — este relatório.

**Modificados:**
- `apps/backend/src/routers/predict.py` — cache-aside (get antes, set depois), sem lógica de negócio nova no router.
- `apps/backend/src/main.py` — cria/fecha o cliente Redis no lifespan.
- `apps/backend/src/core/config.py` — `redis_cache_url` (`REDIS_CACHE_URL`).
- `apps/backend/src/services/protocols.py` — `InferenceCacheProtocol`.
- `apps/backend/pyproject.toml` — `inference_cache.py` no escopo do mutmut (grupo 19), com justificativa.
- `.github/workflows/ci.yml` — Redis real em `test-python` e `mutation-testing` (RNF-71 exige TTL observável, não mock), grupo 19 do mutmut, `REDIS_CACHE_URL` no smoke, step de Locust cache-hit no `load-smoke`.
- `docker-compose.yml` — comentário documentando o reaproveitamento do Redis existente.
- `.env` / `.env.example` — `REDIS_CACHE_URL=redis://redis:6379/1`.
- `.gitignore` — `.mutmut-cache` (evita commit acidental do cache de mutation testing local).

**Nenhuma migration Alembic criada** — ver §3.2 (nenhum índice foi justificado pela evidência).

---

## 12. Comandos para reproduzir

```bash
# Stack real (mesmo que o CI usa no job load-smoke)
docker compose -f docker-compose.yml -f docker-compose.ci-load-smoke.yml up -d --build db redis api

# Baseline (sem cache — payload sempre novo)
PREDICT_MODE=random PREDICT_WAIT_SECONDS=0.8 \
locust -f locust_predict.py --host http://localhost:8000 \
       --headless -u 1 -r 1 --run-time 150s --csv=loadtest_results/predict_baseline

# RNF-70 (cache hit)
PREDICT_MODE=cache_hit PREDICT_WAIT_SECONDS=0.8 \
locust -f locust_predict.py --host http://localhost:8000 \
       --headless -u 1 -r 1 --run-time 150s --csv=loadtest_results/predict_cache_hit

# Testes de cache (exige Redis real — TEST_REDIS_URL)
cd apps/backend
TEST_REDIS_URL=redis://redis:6379/15 pytest tests/test_inference_cache.py tests/test_predict_cache_integration.py -v

# Suíte completa + cobertura
TEST_REDIS_URL=redis://redis:6379/15 pytest --cov=src --cov-report=term-missing

# EXPLAIN ANALYZE (dentro do container db)
docker exec -i db psql -U user -d tcc_db -c "EXPLAIN (ANALYZE, BUFFERS) SELECT * FROM alert_settings WHERE id = 1;"
```

---

## 13. Recomendação sobre commit

Estado pronto para commit, com as ressalvas do §10 registradas (não escondidas). Nenhum commit foi feito automaticamente — aguardando autorização explícita, conforme CLAUDE.md §6.
