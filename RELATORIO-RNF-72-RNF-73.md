# RELATÓRIO — RNF-72 / RNF-73: Inferência ONNX Otimizada + Batch Prediction

## 1. Objetivo

- **RNF-72** — inferência ONNX pelo menos **2x mais rápida** que a Joblib equivalente.
- **RNF-73** — suportar **batch prediction de 100 amostras** via `POST /predict/batch`, com inferência vetorizada real (não um loop de N chamadas).
- Paridade funcional Joblib × ONNX comprovada com tolerância numérica documentada.

Regra seguida do início ao fim: **auditar e medir o baseline antes de qualquer alteração**. Nada foi assumido — cada número abaixo tem o comando que o produziu.

---

## 2. Auditoria inicial — achado central

Antes de escrever qualquer código, a auditoria revelou que **a infraestrutura ONNX já existia no projeto**, de forma madura e versionada:

| Item pedido pela task | Estado encontrado |
|---|---|
| Onde os modelos Joblib são carregados | `ModelService`/`ModelRegistry` (`src/services/model_service.py`), factory `load_model_by_name` |
| Onde ocorre preprocessing | `ModelService._build_feature_row` (stateless, usado por `POST /predict/`) + `MetroPTPreprocessor` (pipeline contínuo) |
| Onde ocorre inferência | `ModelService.predict_from_features` → `self._model.predict_proba(X)` |
| Modelos Random Forest / XGBoost | `random_forest` (joblib) / `random_forest_v2` (onnx); `xgboost` (joblib) / `xgboost_v2` (onnx) |
| Seleção do modelo ativo | `RF-10` — `ACTIVE_MODEL` env var + `PUT /models/active` (hot-swap), `ModelRegistry.KNOWN_MODELS` |
| Contrato atual de `/predict` | `PredictRequest` (12 sensores) → `PredictResponse` (predicted_class, failure_probability, timestamp) — **inalterado nesta task** |
| Infra ONNX prévia | **SIM** — `OnnxTreeAdapter` (`src/services/onnx_tree_adapter.py`), sessão ONNX Runtime reutilizada, CPU execution provider, já **batch-capaz nativamente** (`predict_proba` aceita DataFrame de N linhas) |

### 2.1. Achado #1 — `random_forest_v2.onnx`/`xgboost_v2.onnx` NÃO são modelos diferentes

`train_random_forest.py`/`train_xgboost.py` treinam **um único modelo** e exportam os **dois formatos na MESMA execução**:

```python
# train_random_forest.py (linha ~572-579)
joblib.dump(model, MODELS_DIR / "random_forest_final.joblib")
export_to_onnx(model, feature_names=list(X.columns), onnx_path=MODELS_DIR / "random_forest_v2.onnx")
```

Confirmado por `model_card.json` (compartilhado pelos dois nomes no registry): `feature_count: 80` para RF, `feature_count: 34` para XGB — idêntico nos dois formatos. **Nenhuma conversão em runtime foi necessária nesta task** — os artefatos `.onnx` já existem, versionados em `apps/ml/models/` (mesmo padrão do `.joblib`, ver `.gitignore`).

### 2.2. Achado #2 — XGBoost V1 (joblib) tem uma limitação pré-existente

`xgboost_v1.joblib` foi treinado em **ndarray puro, sem nomes de coluna** — exigência do conversor ONNX `onnxmltools` (`train_xgboost.py`, linha ~337). Consequência: `XGBClassifier.feature_names_in_` não existe nesse artefato, e `ModelService.__init__` (`list(self._model.feature_names_in_)`) levanta `AttributeError`. **Achado pré-existente, não introduzido por esta task** — já documentado em `tests/test_model_service_real_artifacts.py` (`xfail(strict=True)`) antes desta task. Reconfirmado aqui; contornado *somente* para benchmark/paridade (`joblib.load` direto + `xgboost_v1_card.json::feature_names` para ordenar colunas) — **nenhuma correção do bug em si**, fora do escopo.

### 2.3. Testes/fixtures existentes

`tests/test_onnx_tree_adapter.py`, `tests/test_model_service_real_artifacts.py`, `tests/test_model_service_unit.py` já cobriam boa parte do `OnnxTreeAdapter`/`ModelService`. `benchmark_models.py` (pré-existente, RF-18/RNF-36) já comparava os 9 modelos sob condições reais — reaproveitado como referência cruzada (ver §3.3), mas **não** substitui o benchmark dedicado desta task (mede o pipeline contínuo, não o caminho single-sample exato de `POST /predict/`).

---

## 3. Baseline (Fase 1/7 — antes de qualquer alteração de código)

### 3.1. Metodologia

`benchmark_onnx_vs_joblib.py` (novo): 300 amostras **reais** do dataset MetroPT-3 (`apps/ml/data/processed/metropt3.parquet`, igualmente espaçadas — sem embaralhar, sem sintético), mesmo preprocessing de produção (`ModelService._build_feature_row`), warm-up de 20 execuções (não contadas), mesma máquina, mesmo processo, execuções sequenciais para os 4 caminhos.

```bash
cd apps/backend
python benchmark_onnx_vs_joblib.py --n-samples 300 --warmup 20
```

**Ambiente:** container `api` (Docker), AMD Ryzen 7 7700X (16 threads visíveis), `onnxruntime==1.30.0`, `scikit-learn==1.8.0`, `xgboost==3.2.0`, Python 3.11.16.

### 3.2. Resultado — single prediction (`benchmark_onnx_vs_joblib_results.json`)

| Modelo | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (/s) |
|---|---:|---:|---:|---:|---:|
| `random_forest` (joblib) | 49.900 | 49.138 | 54.590 | 62.318 | 20.0 |
| `random_forest_v2` (onnx) | 5.389 | 5.350 | 5.769 | 5.892 | 185.6 |
| `xgboost` (joblib, contorno §2.2) | 1.584 | 1.476 | 1.949 | 4.030 | 631.2 |
| `xgboost_v2` (onnx) | 0.621 | 0.614 | 0.683 | 0.736 | 1609.6 |

**Speedup (Joblib latência / ONNX latência):**

| Modelo | Speedup mean | Speedup p50 | Speedup p95 |
|---|---:|---:|---:|
| Random Forest | **9.26x** | 9.19x | **9.46x** |
| XGBoost | **2.55x** | 2.40x | **2.85x** |

Ambos **muito acima** do piso de 2.0x exigido pelo RNF-72 — não é um resultado marginal.

### 3.3. Validação cruzada com o benchmark pré-existente

`benchmark_models.py` (RF-18/RNF-36, não alterado nesta task) já media, sob o pipeline contínuo real (`_infer_with_history`): `random_forest` p95=48.85ms vs `random_forest_v2` p95=21.03ms (≈2.3x — número menor porque aquele benchmark inclui overhead do pipeline/buffer, não só a inferência). A DIREÇÃO do resultado (ONNX bem mais rápido) é consistente entre os dois benchmarks, medidos por métodos independentes — reforça que o achado é real, não um artefato de metodologia.

### 3.4. Batch baseline (Fase 1 §3 — comportamento atual para 100 amostras)

**Antes desta task, não existia `POST /predict/batch`.** O único jeito de obter 100 predições era 100 chamadas sequenciais a `POST /predict/` — medido em §6.2/§7 como parte da comparação (não estimado, medido).

---

## 4. Conversão dos modelos (Fase 2)

**Nenhuma conversão nova foi necessária** (ver §2.1) — os artefatos `random_forest_v2.onnx`/`xgboost_v2.onnx` já existem, gerados por `train_random_forest.py`/`train_xgboost.py` via `skl2onnx>=1.17.0`/`onnxmltools>=1.12.0` (`apps/ml/requirements.txt`), reproduzíveis com:

```bash
cd apps/ml
python -m src.train_random_forest   # gera random_forest_final.joblib + random_forest_v2.onnx
python -m src.train_xgboost          # gera xgboost_v1.joblib + xgboost_v2.onnx
```

Requisitos dos artefatos (checklist da task) — todos já satisfeitos pelos artefatos existentes:
- ✅ Carregam com ONNX Runtime (`OnnxTreeAdapter`, já testado em `test_onnx_tree_adapter.py`).
- ✅ Input/output names determinísticos (`self._input_name`/`self._output_names`, lidos da sessão).
- ✅ Shape preservada (`FloatTensorType([None, len(feature_names)])` no export — batch dimension explícita).
- ✅ CPU (`providers=["CPUExecutionProvider"]`).
- ✅ Reproduzíveis (comandos acima).
- ✅ Metadados/versionamento — `model_card.json`/`xgboost_v1_card.json` (`feature_count`, `feature_names`, `decision_threshold`, `trained_at`).

---

## 5. `OnnxInferenceEngine` (Fase 3)

**Decisão de design:** não foi criada uma nova classe `OnnxInferenceEngine` duplicando o que já existe. `OnnxTreeAdapter` (`src/services/onnx_tree_adapter.py`, pré-existente) **já satisfaz integralmente** o papel pedido:

- Carrega a sessão ONNX **uma vez** no construtor (`ort.InferenceSession`), nunca por request — reutilizada via `ModelRegistry`/`ModelService` (singleton no lifespan, mesmo padrão do restante do projeto).
- CPU execution provider explícito (`providers=["CPUExecutionProvider"]`).
- `intra_op_num_threads=1`/`inter_op_num_threads=1` já configurados (decisão pré-existente, documentada no próprio adapter — não alterada nesta task, RNF-72 não pediu re-tuning e o baseline já bate a meta com folga).
- Single **e** batch prediction: `predict_proba(X: pd.DataFrame)` já aceita e retorna N linhas — confirmado por `test_extract_probabilities_pass1_ndarray_shape_n_by_2`/`pass2` (pré-existentes) e reexercitado por esta task com N=100 reais.
- Validação de shape/dtype: `np.ascontiguousarray(X.to_numpy(dtype=np.float32))` + `_extract_probabilities` valida o shape da saída antes de aceitar.

Criar uma segunda abstração paralela violaria o próprio princípio da task (reaproveitar, não duplicar) — documentado aqui em vez de escondido.

---

## 6. Paridade Joblib × ONNX (Fase 4)

### 6.1. Metodologia e tolerância

`tests/test_onnx_joblib_parity.py` (novo) — 500 amostras **reais** do dataset (mesma técnica de espaçamento do §3.1) + 5 casos de borda (todos os sensores digitais em 0, todos em 1, pressão alta, pressão baixa, corrente zero). NaN/inf não testados aqui — já cobertos pela validação Pydantic existente (`test_predict_endpoint.py::test_predict_non_numeric_value_returns_422`), que rejeita esses valores antes de chegar ao modelo — nenhuma duplicação.

**Tolerância: `rtol=1e-3`, `atol=1e-4`.** Justificativa (medida empiricamente, não escolhida a priori): ONNX Runtime computa em float32 (`_extract_probabilities`, `dtype=np.float32`); sklearn/xgboost computam em float64. Diferença máxima absoluta observada nas 500 amostras reais:

| Modelo | Max diff observado | Consistente com |
|---|---:|---|
| Random Forest | **1.31e-7** | epsilon de máquina float32 (~1.19e-7) |
| XGBoost | **6.61e-8** | epsilon de máquina float32 |

A tolerância documentada dá **~1000x de margem** sobre o que foi observado — generosa o bastante para não quebrar por ruído de plataforma, apertada o bastante para pegar uma divergência estrutural real de conversão.

### 6.2. Resultado

```bash
cd apps/backend
python -m pytest tests/test_onnx_joblib_parity.py -v
```

```
7 passed in 87.98s
```

- **Random Forest:** 0/500 mismatches de classe, `np.allclose` OK, max diff 1.31e-7, 5/5 casos de borda com classe e probabilidade equivalentes.
- **XGBoost:** probabilidades `allclose` OK (500 amostras + max diff), max diff 6.61e-8. (Comparação de `predicted_class` omitida deliberadamente para XGBoost — o joblib V1 bruto não tem o `decision_threshold` tunado do model card, que só `ModelService` aplica; comparar classes com thresholds diferentes seria uma comparação inválida, não paridade real — só as probabilidades brutas, que independem de threshold, são comparadas.)

**Resultado: PASS.** Nenhuma divergência de classe encontrada; divergência numérica 3-4 ordens de grandeza abaixo da tolerância documentada.

---

## 7. `POST /predict/batch` (Fase 5)

### 7.1. Arquitetura

```
BatchPredictRequest (1-100 samples)
       ↓
router: predict_batch() — I/O puro, delega tudo
       ↓
ModelService.predict_batch(requests)
       ↓
N × _build_feature_row(request)   — preprocessing por amostra (aceitável, RNF-73 §"processamento individual")
       ↓
pd.concat([...]) → matriz [N, features]
       ↓
_align_columns(X)                 — MESMA lógica de predict_from_features (extraída, DRY)
       ↓
self._model.predict_proba(X)      — UMA ÚNICA chamada vetorizada (RandomForestClassifier/OnnxTreeAdapter
                                     já aceitam N linhas nativamente — nenhuma mudança nos adapters)
       ↓
N × PredictResponse (mesma ordem)
       ↓
BatchPredictResponse(predictions=[...], count=N)
```

**Prova de vetorização (não um loop disfarçado):** `test_batch_never_calls_model_predict_proba_more_than_once` — mocka `predict_proba` mantendo `feature_names_in_` real, roda 100 amostras, assevera `call_count == 1`.

### 7.2. Contrato

- `POST /predict/batch` — `{"samples": [PredictRequest, ...]}` (1 a 100), retorna `{"predictions": [PredictResponse, ...], "count": N}`.
- `predictions[i]` corresponde exatamente a `samples[i]` — provado por `test_output_order_matches_input_order` (sensores extremos e distinguíveis em posições diferentes).
- Rate limit próprio: **20/min** (`PREDICT_BATCH_RATE_LIMIT`, `src/core/rate_limit.py`) — mais restritivo que o single (100/min, RNF-19) de propósito: cada requisição de batch pode custar até 100x mais inferência. 20/min no pior caso ainda permite até 2000 inferências/min — proporcional ao orçamento do single.
- `PredictRequest`/`PredictResponse`/contrato de `POST /predict/` **inalterados** (RNF-72/73 §15).

### 7.3. Decisões de escopo deliberadas (documentadas, não escondidas)

- **Sem persistência em `predictions`** — confirmado empiricamente (contagem de linhas antes/depois de uma chamada batch: inalterada). RF-09 ("toda predição bem-sucedida é persistida") foi escopado ao `POST /predict/` de amostra única quando escrito; estender para batch é uma decisão de produto fora do critério de aceite do RNF-73 (que é sobre latência/throughput de inferência vetorizada, não persistência). Inventar gravação em lote não pedida adicionaria complexidade (transação, teste, mutation score) sem requisito correspondente.
- **Sem cache (RNF-70/71) nem alerta (RF-14/RF-24)** — o cache de inferência é escopado a chaves de amostra única; estender para combinações arbitrárias de até 100 amostras multiplicaria a complexidade da chave sem benefício comprovado (um cliente que already sabe que quer um lote normalmente não está repetindo o MESMO lote). Alertas por amostra em lote também não fazem parte do contrato pedido.

---

## 8. Validação do batch (Fase 6)

`tests/test_predict_batch.py` (novo, 20 testes) — duas camadas:

**`TestModelServicePredictBatch`** (artefato REAL `random_forest`, joblib):
- Lote vazio → `[]` (guard explícito, sem depender do caller validar primeiro).
- 1, 2, 10, 100 amostras → `len(output) == len(input)`.
- **Equivalência exata** — `predict_batch([s1..s100])` produz `predicted_class`/`failure_probability` **idênticos** (não só próximos) a `[predict(s1), ..., predict(s100)]`, amostra a amostra.
- Ordem preservada — 3 amostras (extremo-baixo, extremo-alto, extremo-baixo de novo): posições 0 e 2 idênticas entre si, ambas diferentes da posição 1.
- Vetorização real — `predict_proba` chamado exatamente 1x para 100 amostras (mock `wraps` do modelo real).
- Erro do modelo propaga (nunca é engolido) — mesma garantia de `predict_from_features`.

**`TestPredictBatchEndpoint`** (HTTP, modelo mockado, mesmo padrão de `test_predict_endpoint.py`):
- 1, 2, 10, 100 amostras → 200, `count`/`len(predictions)` corretos.
- Lote vazio → 422. 101 amostras → 422. Campo faltando numa amostra → 422. dtype inválido → 422. `samples` ausente → 422.
- Ordem da resposta bate com a ordem do request (mock cuja probabilidade varia com `TP2`).
- Modelo não carregado → 503 (mesmo tratamento do `/predict/` single).

```
20 passed in 8.29s
```

---

## 9. Benchmark RNF-72 — resultado final

Ver §3.2 (medido ANTES de qualquer alteração de código — os caminhos Joblib e ONNX já existiam; nenhuma mudança nesta task alterou a lógica de inferência single-sample, só extraiu `_align_columns` para reaproveitar em `predict_batch`, sem efeito em latência). Re-confirmado após as alterações: `test_onnx_joblib_parity.py` (500 amostras, PASS) e a suíte completa (§11) validam que o comportamento não regrediu.

| Modelo | Speedup mean | Speedup p50 | Speedup p95 | RNF-72 (≥2.0x) |
|---|---:|---:|---:|---|
| Random Forest | 9.26x | 9.19x | 9.46x | ✅ PASS |
| XGBoost | 2.55x | 2.40x | 2.85x | ✅ PASS |

**RNF-72: PASS** para os dois modelos, com margem ampla (RF quase 5x acima do piso; XGB acima do piso mesmo no p50, o número mais conservador).

---

## 10. Benchmark RNF-73 — resultado final

### 10.1. `POST /predict/batch` com 100 amostras (HTTP real, `benchmark_predict_batch.py`)

```bash
python benchmark_predict_batch.py --host http://localhost:8000 --n-batch-requests 30
```

| Métrica | Valor |
|---|---:|
| Requests | 30 |
| Failures | 0 |
| Mean | 37.39 ms |
| p50 | 34.36 ms |
| p95 | 57.47 ms |
| p99 | 75.37 ms |
| Min / Max | 33.28 ms / 76.99 ms |

### 10.2. Comparação — 100 predições: sequencial vs. batch

| Abordagem | Tempo total (parede) |
|---|---:|
| 100x `POST /predict/` sequencial (respeitando 100/min, RNF-19) | 63.62 s |
| 1x `POST /predict/batch` (100 amostras, respeitando 20/min) | 0.035 s |

**Nota de honestidade metodológica:** essa razão (~1805x) é dominada pelo *pacing* deliberado entre as 100 chamadas sequenciais (necessário para respeitar o rate limit real de `/predict/`, nunca contornado) — não é uma comparação pura de custo de inferência. A comparação de custo de inferência pura está em §9 (single) e §10.1 (batch): **37ms para 100 amostras em batch ≈ 0.37ms/amostra**, contra **~50ms/amostra** via `/predict/` sequencial sem espera artificial — ainda assim uma demonstração real e honesta de que o endpoint aproveita a vetorização (não é um loop disfarçado, ver prova em §7.1) **e** de que, na prática, um cliente que precisa de 100 predições se beneficia enormemente do batch (menos overhead de rede/HTTP por amostra, e não fica sujeito ao rate limit mais restritivo do single).

### 10.3. Teste de carga — Locust (`locust_predict_batch.py`)

```bash
locust -f locust_predict_batch.py --host http://localhost:8000 \
       --headless -u 1 -r 1 --run-time 150s --csv=loadtest_results/predict_batch
```

Respeitando o rate limit real do endpoint (20/min, `PREDICT_BATCH_RATE_LIMIT`) — nunca contornado.

| Métrica | Valor |
|---|---:|
| Requests (batches) | 48 |
| Amostras processadas | 4800 |
| Failures | 0 |
| p50 | 77.0 ms |
| p95 | 110.0 ms |
| p99 | 150.0 ms |

Números um pouco mais altos que o benchmark httpx direto (§10.1) — mesmo overhead de plataforma (Docker Desktop/Windows, proxy de porta) já documentado no RELATORIO-RNF-70-RNF-71.md, consistente entre os dois benchmarks desta task também.

**RNF-73: PASS** — 100 amostras processadas corretamente numa única requisição, inferência vetorizada comprovada (§7.1), baixa latência (dezenas de ms, não segundos), 0 falhas em 4800 amostras processadas via carga real.

---

## 11. Regressão (Fase 10)

```bash
cd apps/backend
TEST_REDIS_URL=redis://redis:6379/15 python -m pytest --cov=src --cov-report=term-missing
ruff check . ; black --check . ; mypy . --ignore-missing-imports ; lint-imports
```

```
763 passed, 1 xfailed, 11 warnings in 203.07s
Required test coverage of 85.0% reached. Total coverage: 96.26%
ruff: All checks passed!
black: All done, 113 files unchanged.
mypy: Success: no issues found in 115 source files.
import-linter: Contracts: 1 kept, 0 broken.
```

`src/services/model_service.py` (arquivo com a lógica nova, `predict_batch`/`_align_columns`): **99% de cobertura** (só a linha 361, pré-existente e não relacionada — `load_active_model()`, thin wrapper trivial — segue descoberta).

**Achado real durante a regressão (corrigido):** `ModelServiceProtocol` cresceu para incluir `predict_batch` (RNF-56, mesma fronteira estrutural do `predict`) — o fake de teste em `tests/test_dependency_injection.py::_FakeModelService` não implementava o novo método e deixou de satisfazer o Protocol estruturalmente (`isinstance(fake, ModelServiceProtocol) == False`), quebrando `test_predict_router_uses_fake_model_and_alert_service`. Corrigido adicionando `predict_batch` ao fake (delega para `predict` por amostra — o fake em si não precisa ser vetorizado, só satisfazer o contrato). Reportado aqui porque é exatamente o tipo de achado que "rodar a suíte inteira antes de declarar pronto" deveria capturar.

**Mutation score:** `model_service.py` já está no escopo do `mutmut` (grupo 10, `apps/backend/pyproject.toml [tool.mutmut]`, inalterado nesta task — nenhum grupo novo necessário). Não foi rodada uma passada de mutation testing dedicada e completa deste arquivo (189 statements) nesta sessão — decisão consciente de tempo/risco (ver limitações do RELATORIO-RNF-70-RNF-71.md sobre os riscos operacionais reais de rodar `mutmut` repetidamente contra um ambiente compartilhado). Evidência indireta de robustez: cobertura de linha 99%, e os novos testes usam asserções fortes contra mutação (igualdade exata amostra-a-amostra, contagem de chamadas, preservação de ordem) — historicamente o tipo de asserção que mata a maioria das mutações de lógica. O score AGREGADO real (piso 70%, RNF-64) é decidido pelo `mutation-score-gate` do CI.

---

## 12. Artefatos ONNX — documentação (Fase 11)

| Artefato | Origem | Ferramenta | Versão |
|---|---|---|---|
| `apps/ml/models/random_forest_v2.onnx` | `train_random_forest.py::export_to_onnx` | `skl2onnx.convert_sklearn` | skl2onnx>=1.17.0 |
| `apps/ml/models/xgboost_v2.onnx` | `train_xgboost.py::_export_to_onnx` | `onnxmltools.convert.convert_xgboost` | onnxmltools>=1.12.0 |

Runtime de inferência: `onnxruntime==1.30.0` (medido no container `api`; `requirements.txt` pin `onnxruntime>=1.18.0`). Conversão: `onnx>=1.16.1`, `onnxscript>=0.1.0` (`apps/ml/requirements.txt`).

**Nenhuma dependência nova foi adicionada** — `onnxruntime` (backend), `skl2onnx`/`onnxmltools`/`onnx` (ml) já estavam presentes antes desta task.

**Procedimento para regenerar:**
```bash
cd apps/ml
python -m src.train_random_forest
python -m src.train_xgboost
```

---

## 13. Comparação — antes / depois

| Métrica | Antes (Joblib) | Depois (ONNX) | Resultado |
|---|---:|---:|---|
| RF single p50 | 49.14 ms | 5.35 ms | speedup 9.19x |
| RF single p95 | 54.59 ms | 5.77 ms | **speedup 9.46x** |
| XGB single p50 | 1.48 ms | 0.61 ms | speedup 2.40x |
| XGB single p95 | 1.95 ms | 0.68 ms | **speedup 2.85x** |
| Batch de 100 (endpoint) | N/A (não existia) | p95=57.5ms (httpx) / 110ms (Locust) | RNF-73 PASS |
| Paridade (max diff probabilidade) | — | RF: 1.31e-7 / XGB: 6.61e-8 | dentro de rtol=1e-3/atol=1e-4 |

---

## 14. Resultado RNF-72

**PASS** para Random Forest e XGBoost.

- Medido contra o caminho Joblib REAL (`random_forest`/`xgboost` V1, mesmo preprocessing de produção), não um Joblib artificialmente pior.
- Speedup calculado em `mean`, `p50` **e** `p95` (não só um número isolado) — todos acima de 2.0x.
- Warm-up separado da medição (20 execuções descartadas).
- Amostras reais do dataset MetroPT-3, mesma máquina, mesmo processo, mesmo preprocessing para os dois lados.
- Nenhum request lento excluído, nenhuma média usada isoladamente, threshold/preprocessing não alterados para o benchmark.

## 15. Resultado RNF-73

**PASS.**

- `POST /predict/batch` processa 1 a 100 amostras numa única requisição.
- Inferência genuinamente vetorizada — provado por contagem de chamadas ao modelo (1 chamada para 100 amostras), não um loop disfarçado.
- Medido via endpoint HTTP real (`benchmark_predict_batch.py`) **e** via Locust (`locust_predict_batch.py`) — não só chamada Python interna.
- p95 = 57.5ms (httpx) / 110ms (Locust, com overhead de plataforma) para 100 amostras — latência baixa.
- 0 falhas em 4800 amostras processadas sob carga real, rate limit próprio (20/min) respeitado, nunca contornado.

## 16. Resultado Paridade

**PASS.**

- 500 amostras reais + 5 casos de borda testados para Random Forest (classe E probabilidade).
- 500 amostras reais testadas para XGBoost (probabilidade — comparação de classe evitada por diferença de threshold entre o artefato bruto e o `ModelService`, ver §6.2).
- Tolerância `rtol=1e-3`/`atol=1e-4`, justificada empiricamente (float32 ONNX vs float64 sklearn/xgboost) — 3-4 ordens de grandeza de margem sobre a divergência real observada.
- Zero mismatches de classe encontrados.

---

## 17. Limitações

1. **XGBoost V1 (joblib) tem uma limitação estrutural pré-existente** (§2.2) — não instancia via `ModelService` normal (`ACTIVE_MODEL=xgboost` quebraria a API hoje). Não corrigido nesta task (fora do escopo RNF-72/73, que não pede correção de bugs pré-existentes de outro requisito); contornado apenas para benchmark/paridade via carregamento direto do joblib.
2. **Mutation testing dedicado de `model_service.py` não foi re-executado nesta sessão** (§11) — decisão de tempo/risco, documentada; cobertura de linha 99% e testes com asserções fortes (equivalência exata, contagem de chamadas) são evidência indireta; o score agregado real é decidido pelo CI.
3. **A comparação de "tempo de parede" 100x-sequencial-vs-1x-batch (§10.2) é dominada pelo rate limit**, não pelo custo de inferência puro — documentado explicitamente para não ser lido como "ONNX/batch é 1800x mais rápido que inferência single" (seria enganoso); o número de custo de inferência puro está em §9/§10.1.
4. **Latência medida em Docker Desktop/Windows**, mesmo overhead de plataforma já documentado no RELATORIO-RNF-70-RNF-71.md — não invalida a comparação relativa (mesma máquina para os dois lados de cada comparação), mas os números absolutos em produção Linux tendem a ser menores.
5. **`POST /predict/batch` não persiste nem aciona alertas/cache** (§7.3) — decisão de escopo deliberada e documentada, não uma omissão.

---

## 18. Arquivos alterados/criados

**Novos:**
- `apps/backend/benchmark_onnx_vs_joblib.py` — benchmark RNF-72 (single, Joblib vs ONNX).
- `apps/backend/benchmark_predict_batch.py` — benchmark RNF-73 (HTTP real, batch de 100).
- `apps/backend/tests/test_onnx_joblib_parity.py` — 7 testes de paridade.
- `apps/backend/tests/test_predict_batch.py` — 20 testes de batch (unit + endpoint).
- `locust_predict_batch.py` — carga para `POST /predict/batch`.
- `RELATORIO-RNF-72-RNF-73.md` — este relatório.

**Modificados:**
- `apps/backend/src/services/model_service.py` — `predict_batch()`, `_align_columns()` (extraído de `predict_from_features`, DRY).
- `apps/backend/src/schemas/predict.py` — `BatchPredictRequest`/`BatchPredictResponse`.
- `apps/backend/src/routers/predict.py` — `POST /predict/batch`.
- `apps/backend/src/services/protocols.py` — `ModelServiceProtocol.predict_batch`.
- `apps/backend/src/core/rate_limit.py` — `PREDICT_BATCH_RATE_LIMIT`.
- `apps/backend/tests/test_dependency_injection.py` — `_FakeModelService.predict_batch` (achado real, ver §11).

**Nenhuma migration/índice criado** (fora de escopo — RNF-72/73 não envolve banco). **Nenhuma dependência nova** (`onnxruntime`/`skl2onnx`/`onnxmltools` já presentes). **Nenhum artefato ONNX novo** (já existiam).

---

## 19. Comandos para reproduzir

```bash
# Stack real
docker compose -f docker-compose.yml -f docker-compose.ci-load-smoke.yml up -d --build db redis api

# RNF-72 — baseline/benchmark single (Joblib vs ONNX)
docker exec api python benchmark_onnx_vs_joblib.py --n-samples 300 --warmup 20

# Paridade
docker exec api python -m pytest tests/test_onnx_joblib_parity.py -v

# Batch — testes
docker exec api python -m pytest tests/test_predict_batch.py -v

# RNF-73 — benchmark HTTP real
docker exec api python benchmark_predict_batch.py --host http://localhost:8000 --n-batch-requests 30

# RNF-73 — carga (Locust)
locust -f locust_predict_batch.py --host http://localhost:8000 \
       --headless -u 1 -r 1 --run-time 150s --csv=loadtest_results/predict_batch

# Regressão completa
docker exec -e TEST_REDIS_URL=redis://redis:6379/15 api python -m pytest --cov=src --cov-report=term-missing
docker exec api sh -c "ruff check . && black --check . && mypy . --ignore-missing-imports && lint-imports"
```

---

## 20. Recomendação sobre commit

Estado pronto para revisão, com as ressalvas do §17 registradas (não escondidas). **Nenhum commit foi feito** — aguardando apresentação dos resultados e autorização explícita, conforme instrução da task (§16 "AO FINAL... Não faça commit ainda") e CLAUDE.md §6.
