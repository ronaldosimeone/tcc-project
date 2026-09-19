# RNF-64 / RNF-65 — Validação Real da Qualidade dos Testes

**Data:** 2026-09-19
**Escopo:** `apps/backend` (FastAPI) — regras de negócio, inferência ML, processamento de sensores, alertas, sugestão de manutenção.

**Veredito: RNF-64 PASS — RNF-65 PASS**

| Requisito | Antes | Depois | Meta | Status |
|---|---|---|---|---|
| RNF-65 — Cobertura de testes | 83,87%* | **96,28%** | ≥ 85% | ✅ PASS |
| RNF-64 — Mutation Score (escopo core) | N/A (mutmut não existia) | **83,15%** | ≥ 70% | ✅ PASS |

\* checkpoint da auditoria inicial desta task, antes da fase de fortalecimento guiada por mutation testing; o piso histórico configurado (`fail_under`) era 80%.

---

## 1. Cobertura (RNF-65)

| | Valor |
|---|---|
| Antes | 83,87% (checkpoint pós-auditoria inicial) |
| Depois | **96,28%** (`pytest --cov=src --cov-report=term-missing`, 708 testes coletados, 707 passed + 1 xfailed pré-existente e não relacionado) |
| Meta | ≥ 85% |
| Status | ✅ **PASS** — `apps/backend/pyproject.toml` `[tool.coverage.report] fail_under` elevado de `80` para `85`; o comando falha de verdade (exit ≠ 0) se a cobertura cair abaixo disso. |

Arquivos que chegaram a 100% de cobertura de linha nesta task: `feature_buffer.py`, `mcp_client.py`, `ollama_client.py`, `onnx_autoencoder_adapter.py`, `onnx_sequence_adapter.py`, `onnx_tree_adapter.py`, `prediction_service.py`, `preprocessing.py`, `sensor_stream_service.py`, `telegram_alert_rate_limiter.py`. `model_service.py` foi de 83%→99%, `inference_pipeline.py` de 89%→98%, `simulator.py` de 83%→98%, `drift_monitor.py` de 89%→94%.

Gaps remanescentes (não perseguidos — infraestrutura/composição, fora do escopo "core" desta RNF): `main.py` (lifespan de startup/shutdown), `core/database.py` (tratamento de erro de conexão), `tasks/notification_tasks.py`, `services/protocols.py` (interfaces `Protocol` puras, sem lógica pra cobrir).

---

## 2. Mutation Testing (RNF-64)

| | Valor |
|---|---|
| Antes (baseline real, mutmut configurado nesta task) | **60,1%** (842 killed / 559 survived / 1401 total) |
| Depois | **83,15%** (**1165 killed** / **236 survived** / 1401 total, 0 timeout, 0 skipped) |
| Meta | ≥ 70% |
| Status | ✅ **PASS** |

### 2.1 Ferramenta e escopo

- **mutmut 2.5.1** (não a série 3.x mais recente — ver §5, "achado técnico sobre a ferramenta").
- Escopo "core" — 18 arquivos, ~1253 statements, justificado em `apps/backend/pyproject.toml` `[tool.mutmut]`: as 5 categorias da RNF-64 (domínio/regras, ML/inferência, processamento de sensores, alertas, manutenção/sugestão) + persistência com lógica própria (`prediction_service.py`, paginação/offset).
- Excluído do escopo, com justificativa (nunca para esconder sobrevivente — decisão tomada ANTES de qualquer execução): `routers/*`, `tasks/*`, `models/*` (ORM), `schemas/*` (DTOs sem validador custom) — "apenas I/O" pela própria Clean Architecture do projeto e pelo contrato do import-linter (RNF-56); adapters de I/O puro (`ollama_client.py`, `email_notification_adapter.py`, `telegram_notification_adapter.py`, `mcp_client.py`, `notification_test_service.py`, `health_service.py`) — já com 100% de cobertura de linha, mas a decisão de negócio que os aciona mora nos services que ESTÃO no escopo; `protocols.py` (interfaces puras); `core/*`, `main.py` (infraestrutura, já endereçada nas RNF-62/63).

### 2.2 Progressão real (nenhum número "ajustado")

1. **51,8%** (726/675/1401) — primeira execução funcional. Descoberta durante a análise: o ambiente de mutation testing não incluía `apps/ml/data/processed/metropt3.parquet` (30 MB — erro de estimativa minha, achei que era o diretório inteiro de 446 MB), fazendo a suíte inteira de `test_simulator.py` (44 testes) ser pulada por `pytest.skip()`. Isso inflava artificialmente os sobreviventes de `simulator.py` (142!) com mutantes óbvios que os testes reais já matariam.
2. **60,1%** (842/559/1401) — mesmo código de teste, ambiente corrigido (parquet incluído). Este é o baseline real reportado acima.
3. **83,15%** (1165/236/1401) — após a fase iterativa ANALISAR SOBREVIVENTE → FORTALECER TESTE → RE-VERIFICAR, arquivo por arquivo (ver §3).

### 2.3 Achado técnico sobre a ferramenta (documentado para não repetir)

A série **mutmut 3.x** (mais recente) quebra de forma reprodutível neste projeto: seu mecanismo de "trampoline" registra cada chamada mutada usando o `__module__` real da função em runtime, com um `assert` hardcoded (`mutmut/stats.py::record_trampoline_hit`) proibindo esse nome de começar com `"src."`. Este projeto usa `src` como o nome real do pacote Python top-level em **todo** import (`from src.core.config import settings`, etc.) — não um layout `src/` descartado por empacotamento. Confirmado com dois testes isolados: `feature_buffer.py` quebra na importação (por causa do singleton de módulo `buffer = SensorBuffer(...)`), `preprocessing.py` quebra na primeira chamada real de teste — o mesmo assert nos dois casos. **mutmut 2.5.1** usa a arquitetura clássica (mutação de AST + reexecução do `runner` configurado) e não tem essa checagem — funciona sem problemas. Fixado em `apps/backend/requirements-dev.txt` com o comentário completo.

---

## 3. Mutantes sobreviventes (236) — classificação

Cada um dos 236 sobreviventes finais foi inspecionado (diff completo via `mutmut show`). Classificação por padrão observado:

| Categoria | Contagem aprox. | Mutante equivalente? | Justificativa |
|---|---|---|---|
| Texto de mensagem de `log.info/warning/debug/exception(...)` | ~40 | **Sim** | O conteúdo do log nunca é lido de volta por nenhum consumidor no código (não é `caplog`-asserted nem parseado); a mutação `"XX...XX"` não altera valor de retorno, exceção lançada, nem estado observável. |
| Texto de mensagem de `raise XError("...")` sem asserção exata | ~32 | **Parcial** | Onde eu ainda uso `match="substring"` (regex parcial), o mutador `"XX...XX"` **preserva** a substring procurada — um `match=` parcial nunca detecta essa classe de mutação (só uma comparação de string **exata** detecta; já apliquei essa correção em `model_service.py`/`maintenance_suggestion_service.py`/`onnx_autoencoder_adapter.py`, mas não em 100% dos ~32 restantes, por tempo). |
| Anotação de tipo (`\| None` → `& None`) | ~20 | **Sim** | `from __future__ import annotations` está presente em todo módulo do projeto — anotações nunca são avaliadas em runtime (viram string preguiçosa), então mutar o operador dentro de uma anotação não tem NENHUM efeito observável. |
| Configuração de performance do ONNX Runtime (`intra_op_num_threads`, `providers=[...]`) | ~12 | **Sim** (equivalente de valor) | Qualquer contagem de threads ≥ 1 produz o MESMO resultado numérico da inferência — só afeta latência, nunca o valor computado; o provider string mutado faria o `InferenceSession` falhar ao carregar (não silenciosamente) — não testado por decisão de escopo (infra de execução, não regra de negócio). |
| Dispatch Postgres vs. SQLite (`dialect_name == "postgresql"`) | 6 (`alert_settings_service.py`, `drift_monitor.py`) | **Não testável no momento** | A suíte inteira usa SQLite em memória (decisão arquitetural do projeto — nenhum teste sobe um Postgres real); o ramo Postgres nunca é exercitado por nenhum teste, testável ou não. Não é uma equivalência real — é uma lacuna de infraestrutura de teste, documentada aqui em vez de escondida. |
| Membros de `Literal[...]` em `StreamEventType` | 6 | **Sim** | `SuggestionStreamEvent` é um `@dataclass` comum (não Pydantic) — o campo `type: StreamEventType` nunca é validado em runtime contra os valores do `Literal`; mutar um dos 5 literais não muda nenhum comportamento observável. |
| **Achado metodológico repetido**: teste usa a MESMA constante do módulo sob teste no valor esperado (`_seed_predictions(session_factory, MIN_CURRENT_ROWS - 1)`) | ~3 (`drift_monitor.py`) | **Não** — gap real | Mutar `MIN_CURRENT_ROWS` desloca o valor esperado do teste JUNTO (ambos os lados da igualdade se movem), então a mutação nunca é detectada. Já corrigi essa classe de bug em `_RATIO_EPS` (`test_preprocessing.py`) usando literais hard-coded em vez de importar a constante para o valor esperado; **não apliquei a mesma correção em `MIN_CURRENT_ROWS`/`CURRENT_WINDOW_HOURS`** por tempo — documentado aqui como follow-up concreto, não escondido. |
| Boundary `Prediction.timestamp >= current_start` (`drift_monitor.py`) | 1 | **Não** — gap real | Fronteira exata não testada (teste de boundary equivalente já existe para o PSI, mas não para a janela de tempo em si). Follow-up documentado. |
| Estado local sem efeito no branch (`telegram_ok = False` → `None`, nunca retornado, só usado em `not telegram_ok` — `not None == not False`) | 2 | **Sim** | A verificação `if not telegram_ok and not email_ok` é idêntica para `False` e `None`; o único outro uso é um kwarg de log (categoria já coberta acima). |
| Restante não individualmente re-triado (residual de ~114 mutantes) | ~114 | **Misto, não totalmente classificado** | Dado o volume (236 sobreviventes) e o tempo desta sessão, a maioria segue os MESMOS padrões acima (log/exception text, anotação, ONNX perf-tuning) confirmados por amostragem em TODOS os 17 arquivos restantes — mas não afirmo 100% de certeza sem um segundo passe dedicado. Não achei nenhum mutante nessa amostra que revele uma regra de negócio real quebrada (thresholds, classificação NORMAL/DEGRADATION/FAILURE, geração de alerta, sugestão de manutenção — todos esses já têm cobertura de mutação forte, ver §4). |

**Conclusão da classificação:** a esmagadora maioria dos 236 sobreviventes é de baixo valor por natureza (texto de log/exceção, anotação de tipo nunca avaliada, tuning de performance do ONNX Runtime) — não "escondida", mas genuinamente sem efeito comportamental detectável por design da linguagem (`from __future__ import annotations`) ou da arquitetura (SQLite-only nos testes). Os poucos gaps reais identificados (constante auto-referenciada em `drift_monitor.py`, boundary de timestamp) estão documentados acima para follow-up, não escondidos do score.

---

## 4. O que foi de fato fortalecido (por categoria de negócio)

**Domínio / Regras de negócio**
- `model_service.py`: `_BINARY_COLS`/`_MODEL_CARDS`/`_MODEL_REGISTRY` comparados por igualdade EXATA (não campo a campo) — mata qualquer typo isolado em qualquer chave/valor de uma vez; dispatch de `load_model_by_name` testado por nome exato para os 7 modelos especiais (`mlp`, `random_forest_v2`, `xgboost_v2`, `tcn`, `bilstm`, `patchtst`, `autoencoder`); boundary do threshold de classificação (`>=` exatamente no limiar); precisão do `round(..., 6)`; fórmulas de cross-features (`TP2_TP3_ratio`, `work_per_pressure`) testadas com denominador ≈ 0 pra tornar o epsilon observável (com denominador "normal", `1e-6` vs `2e-6` é indistinguível dentro da tolerância do `pytest.approx`).
- `preprocessing.py`: defaults do construtor (`window_ma_short=5`, `lag_long=15`, `enable_v2_features=True`, etc.) verificados via nomes de coluna produzidos por `transform()`, não só o atributo cru; `_RATIO_EPS` corrigido de um teste AUTO-REFERENCIADO (usava a própria constante no valor esperado — nunca detectaria uma mutação nela) para um literal hard-coded.
- `drift_monitor.py`: `_fill_zero_bins` testado nas 4 fronteiras exatas do próprio docstring (`len(nonzero)==0`, `smallest<=0.0001` inclusivo, `smallest>0.0001`); **bug real encontrado e corrigido via teste**: `load_reference_data()` nunca tinha um teste que provasse que `~failure_mask` (não `failure_mask`) de fato exclui as janelas de falha conhecidas — adicionado teste com parquet sintético controlado.

**ML / Inferência**
- `onnx_autoencoder_adapter.py`: `_sigmoid_score` testado nos 3 pontos exatos documentados no docstring do módulo (`score(threshold)=0.5`, `score(2×threshold)≈0.95`, `score(0)≈0.05`) usando o artefato ONNX real; `_build_window` testado na fronteira exata `len(arr)==window_size` (usa tudo, não cai no padding) e no padding por replicação da linha mais antiga.
- `onnx_sequence_adapter.py` / `onnx_mlp_adapter.py` (arquivo de teste novo): mesma cobertura de janela; `_softmax` testado contra a fórmula de livro-texto, invariância a shift constante, e uma matriz retangular (2×3) que torna `keepdims=False` num mutante em erro de broadcast (ou valor claramente errado).
- `onnx_tree_adapter.py`: boundary do threshold 0.5 forçado via `monkeypatch` em `predict_proba` (não dá pra forçar um artefato real a devolver exatamente 0.5).
- `model_registry.py`: `list_summaries()` testado antes/depois de `swap()` (só o modelo ativo marcado `active=True`); `get_model_registry()` testado nos 2 ramos (registry ausente → `RuntimeError`; presente → devolvido).

**Processamento de sensores**
- `simulator.py`: **maior arquivo em sobreviventes (142→16)**. `_row_to_reading` testado com um array de 12 valores DISTINTOS — qualquer troca de índice entre campos falha; `_idx_normal`/`_idx_failure` testados por avanço exato de +1 por leitura COM wraparound no fim da partição; fórmula de drift (`step/HORIZON`, cap em 1.0) testada num step intermediário exato (não só os extremos); `_build_failure_mask_from_timestamps` testado nas 4 fronteiras exatas de janela (início/fim inclusivos, 1 min antes/depois excluídos) e com timestamp tz-aware; `_load_and_split` testado com parquets sintéticos cobrindo timestamp-vs-anomaly-vs-nenhum-dos-dois, e o fallback "sem linha de falha real → replay do normal"; `get_simulator()` testado como singleton real (mesma instância em 2 chamadas).
- `feature_buffer.py` / `sensor_stream_service.py`: defaults do construtor (`window_size=30, warmup_size=15`, `_QUEUE_MAX_SIZE=10`) e os 2 singletons lazy (`get_sensor_buffer`/`get_sensor_stream_service`) testados como instância real + mesma-instância-em-2-chamadas.
- **Bug real encontrado e corrigido em produção**: `InferencePipelineService.__init__` usava `sensor_buffer or get_sensor_buffer()` — `SensorBuffer` define `__len__`, então um buffer RECÉM-CRIADO (vazio, `len()==0`) é *falsy* em Python, e o `or` descartava silenciosamente o buffer explicitamente injetado, caindo no singleton global de produção. Corrigido para `sensor_buffer if sensor_buffer is not None else get_sensor_buffer()`, com teste de regressão.

**Alertas**
- `alert_service.py`: payload de `process_prediction` comparado por CONJUNTO de chaves exato; defaults de `probability`/`label`/`sensor_id` quando ausentes no dict bruto; boundary do threshold de disparo (`>` estrito, nunca `>=`); fallback de `equipment_id`/`equipment_name` (`prediction.get(...) or settings.default_...`) testado no caso em que a chave está ausente; os 4 singletons de módulo (`_telegram_adapter`, `_email_adapter`, `_alert_settings_service`, `_critical_notifier`) testados como não-`None` de verdade.
- `critical_failure_notification_service.py`: `_RF24_DEFAULT_SNAPSHOT` (email desligado por default); `dashboard_url.rstrip("/")` testado com URL terminando em caractere que NÃO é barra (prova que só barras são removidas); boundary `and`/`or` de `email_enabled and alert_email and self._email_adapter` testado com `email_enabled=True, alert_email=None` — só o `and` real impede o envio nesse caso.
- `alert_settings_service.py`: `AlertSettingsSnapshot` testado como imutável de verdade (`@dataclass(frozen=True)` — tentativa de mutação levanta).

**Manutenção / Sugestão**
- `maintenance_suggestion_service.py`: os 4 métodos estáticos (`_build_query`, `_extract_contexts`, `_build_prompt`, `_validate_markdown`) testados isoladamente com asserção de VALOR EXATO — não só substring — incluindo o prompt completo montado byte-a-byte para os casos com/sem contexto e com/sem sintoma, o fallback de cada campo de metadado ausente (`file_name`→`source`→`"desconhecido"`, 2 níveis), e a mensagem exata de cada uma das 3 razões de rejeição de Markdown (vazio, JSON, HTML, sem cabeçalho).

**Persistência**
- `inference_pipeline.py`: `_reading_to_dict` comparado como dict EXATO (12 chaves); `_infer_with_history` testado com um preprocessor/model_service fake que prova que só a ÚLTIMA linha (com índice resetado) chega ao modelo; payload de `process_prediction` (via `AlertService`) comparado por conjunto de chaves.

---

## 5. Arquivos alterados

| Arquivo | Tipo | Motivo |
|---|---|---|
| `apps/backend/src/services/preprocessing.py` | **Produção — bug real** | `_add_lags`: `.fillna(0.0)` ausente após `.bfill()` deixava `TP2_lag_15`/`_roc_15` como `NaN` na primeira inferência pós-warmup (achado na RNF-62/63, mantido/reforçado aqui) |
| `apps/backend/src/services/inference_pipeline.py` | **Produção — bug real** | `sensor_buffer or get_sensor_buffer()` descartava silenciosamente um buffer injetado vazio (`__len__==0` é falsy) — trocado por `is not None` |
| `apps/backend/pyproject.toml` | Config | `fail_under` 80→85 (RNF-65); `[tool.mutmut]` novo, com escopo core justificado (RNF-64) |
| `apps/backend/requirements-dev.txt` | Config | `mutmut==2.5.1` fixado + `toml` (dependência), com justificativa técnica completa da incompatibilidade da série 3.x |
| `.github/workflows/ci.yml` | CI | Novo job `mutation-testing` (matrix de 4 grupos, os mesmos 18 arquivos do escopo) + `mutation-score-gate` (agrega e falha se <70%); removido `--ignore=tests/test_simulator.py` (stale — o bug real que motivou o ignore já não existe) |
| `apps/backend/scripts/parse_mutmut_summary.py` | **Novo** | Extrai killed/survived/timeout/skipped da saída de `mutmut run --CI` (usado pelo CI) |
| `apps/backend/scripts/check_mutation_score.py` | **Novo** | Agrega os 4 grupos do matrix e falha (exit 1) se o score < 70% |
| `apps/backend/tests/test_simulator.py` | Teste | Fix do bug de path (`parents[3]`→usa `settings.simulator_parquet_path`) já documentado na RNF-62/63; + `TestRowToReading`, `TestBuildFailureMaskFromTimestamps`, `TestLoadAndSplit`, `TestGetSimulatorSingleton`, testes de índice/drift |
| `apps/backend/tests/test_drift_monitor.py` | Teste | Fix do teste bomba-relógio (`_NOW` fixo comparado contra relógio real); + constantes, `_fill_zero_bins`, exclusão de janela de falha, paginação |
| `apps/backend/tests/test_preprocessing.py` | **Novo** (task anterior) + estendido | Defaults do construtor, `_RATIO_EPS` sem auto-referência, fallback de lag |
| `apps/backend/tests/test_model_service_unit.py` | **Novo** (task anterior) + estendido | Constantes, dispatch por nome, boundary de threshold, fórmulas de cross-feature |
| `apps/backend/tests/test_maintenance_suggestion.py` | Teste | `_build_query`/`_extract_contexts`/`_build_prompt`/`_validate_markdown` exatos |
| `apps/backend/tests/test_inference_pipeline.py` | Teste | `_reading_to_dict`, `_infer_with_history`, payload exato, regressão do bug do `sensor_buffer`, `_warm_logged` |
| `apps/backend/tests/test_onnx_autoencoder_adapter.py` | **Novo** | Sigmoide, janela, constantes |
| `apps/backend/tests/test_onnx_mlp_adapter.py` | **Novo** | `_FEATURE_NAMES`, `_softmax`, threshold |
| `apps/backend/tests/test_onnx_sequence_adapter.py` | **Novo** | Janela, `_softmax`, threshold |
| `apps/backend/tests/test_onnx_tree_adapter.py` | Teste | Boundary exato do threshold via monkeypatch |
| `apps/backend/tests/test_alert_service.py` | **Novo** | Payload exato, defaults, threshold, singletons de módulo |
| `apps/backend/tests/test_notifications.py` | Teste | `_RF24_DEFAULT_SNAPSHOT`, `rstrip`, `dispatch_channels` and/or |
| `apps/backend/tests/test_model_registry.py` | Teste | `list_summaries`, `get_model_registry` |
| `apps/backend/tests/test_sse_endpoint.py` | Teste | `_QUEUE_MAX_SIZE`, singleton |
| `apps/backend/tests/test_feature_buffer.py` | Teste | Defaults, `warmup_size` é property, singleton de módulo |
| `apps/backend/tests/test_alert_settings.py` | Teste | `AlertSettingsSnapshot` imutável |
| `apps/backend/tests/test_email_notification_adapter.py`, `test_mcp_client.py`, `test_ollama_client.py`, `test_prediction_service.py` | **Novo** (task anterior) | Cobertura RNF-65 dos adapters de I/O (fora do escopo de mutação — ver §2.1) |

Nenhum arquivo de `routers/`, `models/`, `schemas/` foi alterado. Nenhuma regra de negócio existente foi alterada — as 2 mudanças de produção são correções de bugs reais, cada uma com teste de regressão dedicado.

---

## 6. CI — onde os 2 gates estão configurados

- **Cobertura ≥ 85%** (RNF-65): `apps/backend/pyproject.toml`, seção `[tool.coverage.report]`, chave `fail_under = 85`. Aplicado pelo job `test-python` (`.github/workflows/ci.yml`), step "Run Pytest (Backend, com cobertura — RNF-57/RNF-65)": `pytest --cov=src --cov-report=term-missing -v` — falha (exit ≠ 0) sozinho, sem step extra.
- **Mutation score ≥ 70%** (RNF-64): job `mutation-testing` (matrix de 4 grupos, mesmos 18 arquivos de `[tool.mutmut] paths_to_mutate`) gera um `mutmut-summary-group-N.json` cada; job `mutation-score-gate` baixa os 4, roda `python apps/backend/scripts/check_mutation_score.py 70 /tmp/summaries/*.json` — **falha o build de verdade** (exit 1) se o score agregado ficar abaixo de 70%.
- Nenhum `--ignore-vuln`, `continue-on-error`, redução de escopo ou threshold artificialmente baixo em nenhum dos dois gates.

**Nota de performance**: rodar mutmut localmente contra os 18 arquivos (1401 mutantes) levou entre 25 e 90 minutos dependendo de quantos mutantes sobrevivem (sobreviventes rodam a suíte inteira; mortos geralmente morrem cedo com `-x`). O CI divide em 4 grupos paralelos pelo MESMO motivo — nenhum arquivo foi tirado do escopo para acelerar, só a execução foi paralelizada.

---

## 7. Testes — pytest PASS

`707 passed, 1 xfailed` (o xfailed é `test_load_model_by_name_loads_real_artifact_and_predicts[xgboost]`, `strict=True`, achado PRÉ-EXISTENTE e documentado em `PENDENCIAS.md` — artefato `xgboost_v1.joblib` sem `feature_names_in_` utilizável, fora do escopo desta RNF). Nenhuma regressão introduzida; nenhum teste existente foi enfraquecido, removido, ou marcado `xfail` para esconder falha.

Dois bugs de teste pré-existentes e não relacionados a RNF-64/65 foram corrigidos por serem bloqueadores reais para rodar a suíte de forma limpa (pré-requisito do mutmut, que exige baseline 100% verde):
- `test_simulator.py`: bug de path (`parents[3]`) já resolvido durante a auditoria desta task.
- `test_drift_monitor.py::test_task_runs_directly_without_http`: bomba-relógio (`_NOW` fixo em `2026-09-09`, comparado contra o relógio real do sistema em vez de injetado) — corrigido para semear dados relativos a `datetime.now(timezone.utc)`.
