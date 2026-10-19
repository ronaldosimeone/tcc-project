# Requisitos de nível doutorado — status e evidências

Registro dos RF/RNF/RN acrescentados para levar o PredictIQ a nível de tese.
Cada item aponta o código, o teste e o artefato de resultado. **Resultado
negativo é resultado**: quando a hipótese falhou, isso está escrito aqui e
vira discussão na tese, nunca é escondido.

Legenda de status: ✅ implementado e validado · 🟡 implementado, validação parcial ·
❌ implementado, hipótese rejeitada · ⏳ pendente (depende de algo externo).

Regenerar tudo: `bash scripts/reproduce-thesis.sh` (RNF-78). Manifesto de
versões e hashes: [`reproducibility_manifest.json`](../reproducibility_manifest.json).

## Regras de negócio

| ID | Regra | Implementação | Status | Evidência |
|---|---|---|---|---|
| RN-01 | Alarme vale como aviso antecipado se vier ≥ 2 h antes da falha | `apps/ml/src/event_metrics.py` (`min_lead_s`) | ❌ **Nenhum modelo cumpre**: lead time de +0,13 h (XGB) a +0,92 h (RF) | [`temporal_evaluation.md`](../temporal_evaluation.md) |
| RN-02 | Limiar minimiza o custo esperado (C_FN = 100 · C_FP), escolhido em scores OOF do treino | `cost_optimal_threshold`, `evaluate_temporal._inner_threshold` | ✅ | idem |
| RN-03 | Alarme só após k = 6 amostras consecutivas acima do limiar | `persistent_alarms` | ✅ | `tests/test_temporal_validation.py` |
| RN-04 | Plano sem trecho de manual ≥ τ é bloqueado; citação a página não recuperada é sinalizada | `apps/backend/src/services/grounding.py` | ✅ τ = 0,6 só funciona com bge-m3 (AUC de fundamentação 0,89 × 0,61 do MiniLM) | [`rag_benchmark.md`](../rag_benchmark.md), `tests/test_grounding_xai.py` |
| RN-05 | Promoção só com significância estatística (Wilcoxon pareado, p < 0,05) | `apps/ml/src/promotion_gate.py` | ✅ Só com o MetroPT-3, o gate recusa toda troca como **subpotente** (3 folds → p mínimo 0,125). Com o walk-forward conjunto (6 folds, p mínimo 0,016) ele **promove** XGBoost sobre LogReg (p = 0,016) e recusa XGBoost × RF (p = 0,44) | `tests/test_promotion_gate.py`, [`cross_dataset_evaluation.md`](../cross_dataset_evaluation.md) |

## Requisitos funcionais

| ID | Requisito | Implementação | Status | Achado principal |
|---|---|---|---|---|
| RF-28 | Validação temporal com *purge*/embargo | `temporal_validation.py` (walk-forward + blocked) | ✅ | PR-AUC do RF cai de 0,9992 (split aleatório) para 0,587 [0,43; 0,74] |
| RF-29 | Avaliação por evento (recall de evento, lead time, FA/dia, range-F1) | `event_metrics.py` | ✅ | A falha de 15/07 (fold 3) não é detectada por XGB nem LR em nenhuma seed |
| RF-30 | RUL por análise de sobrevivência | `rul.py`, `evaluate_rul.py` (XGBoost AFT, censura no corte de treino) | ❌ | C-index 0,49 (≈ acaso) × 0,58 do baseline de renovação: 1–3 falhas não bastam para aprender RUL |
| RF-31 | Incerteza: calibração + conformal | `uncertainty.py` (ECE, isotônica, Mondrian) | 🟡 | ECE 0,05; cobertura conformal da classe falha 0,59–0,68 contra a meta de 0,90 → **não-permutabilidade** temporal |
| RF-32 | Explicabilidade por alerta | Offline: `explainability.py` (TreeSHAP). Online: oclusão por sensor congelada na abertura de cada evento (`root_cause_service.py`), exibida no drawer do Histórico | ✅ | Fiel (H5 ✅); TreeSHAP instável entre falhas (H6 ❌). Em verificação real, a oclusão online também apontou `Oil_temperature`/`TP2` |
| RF-33 | Generalização cross-dataset | MetroPT 2022 (Zenodo 6854240, 3 falhas) via `ingest_metropt2022.py` (md5, amostragem pontual 10 s, GPS fora); `evaluate_cross_dataset.py` (shift adversarial, transferência nos dois sentidos, walk-forward conjunto com 6 folds, ablação do `Flowmeter`) | 🟡 | **Transferência direta falha**: 2020→2022 pega 1/3 falhas, 2022→2020 ≤ 1/4. Dois sensores mudam de significado com o mesmo nome (`Reservoirs` = cópia do `TP3` em 2020; `Oil_level` constante em 2022) e foram excluídos; mesmo assim os períodos são separáveis (AUC adversarial 0,998). **Somar compensa em poder estatístico**: com 6 folds, Friedman p = 0,032 e o gate RN-05 promove XGBoost × LogReg (p = 0,016), os primeiros resultados significativos da tese. **Modelo de produção no 2022, sem retreino: PR-AUC 0,058 e 0/3 falhas** (1,000 e 4/4 nos dados de 2020 que viu: sobreajuste do split aleatório + SMOTE). **Twin entre períodos**: calibrado só com semanas *normais* de 2022, a física detecta a 1ª falha de 2022; sem calibrar, não. No walk-forward conjunto, a fusão ML + física tem o menor custo (39,1 × 51,2 do ML) e o híbrido é o único que pega o vazamento de óleo (2/3 seeds), mas nenhuma diferença passa no gate RN-05. Challenger `xgboost_v3` (2020 + 2022, sem SMOTE, limiar OOF 0,985) registrado no MLflow, não promovido (p = 0,44) ([`cross_dataset_evaluation.md`](../cross_dataset_evaluation.md)) |
| RF-34 | Drift adaptativo | `drift_detectors.py` (Page-Hinkley, ADWIN), `evaluate_drift.py` | ❌ | A regra de produção (PSI > 0,25) gera **28 falsos alarmes/30 dias** em operação normal; calibrados para ≤ 1 alarme/mês, os detectores pegam no máximo 17 % dos drifts ≤ 2σ |
| RF-35 | Diagnóstico XAI → MCP → RAG | `sensor_semantics.py`, `_build_query`/`_build_prompt` | 🟡 | Ganho de ranking depende do embedder: significativo com mpnet (p = 0,004), não com MiniLM/bge-m3 |
| RF-36 | Benchmark RAG anotado (≥ 50 casos) | `benchmarks/rag_cases_v1.json` (42 respondíveis + 12 sem resposta), `benchmark_rag.py` | 🟡 | Anotação em rascunho, **pendente de validação por especialista** |
| RF-37 | Ablação LLM puro × RAG × RAG+XAI, modelos, embedders | `benchmark_rag.py`, `benchmark_rag_retrieval.py` | ✅ | LLM puro alucina em **100 %** das perguntas sem resposta; o RAG em 0–8 %. O embedder de produção recupera o trecho certo em 2,4 % dos casos, o bge-m3 em 28,6 % |
| RF-38 | Human-in-the-loop | `suggestion_feedback_service.py`, `POST /v1/maintenance/feedback`, migração 0005; veredito da OS no drawer do Histórico; **voto por trecho citado e utilidade do plano no assistente** (`maintenance-assistant/suggestion-feedback.tsx`) | ✅ | O veredito da OS vira rótulo dos KPIs e do re-treino; os votos por trecho reordenam a recuperação (prior de Laplace) sem tocar no τ do RN-04. `__tests__/suggestion-feedback.test.tsx` |

## Requisitos não funcionais

| ID | Requisito | Implementação | Status | Achado principal |
|---|---|---|---|---|
| RNF-78 | Reprodutibilidade total | `scripts/reproduce-thesis.sh`, `scripts/write_manifest.py` | ✅ | Manifesto com commit, imagem Docker, modelos Ollama e SHA-256 de entradas e resultados |
| RNF-79 | Rigor estatístico (≥ 5 seeds, IC 95 %, Friedman/Nemenyi) | `stat_tests.py`, diagramas CD | ✅ | Nenhuma diferença significativa entre modelos (Friedman p ≥ 0,37, N = 3) |
| RNF-80 | Edge / INT8 | `edge_quantization.py` | 🟡 | MLP: −72 % de tamanho, 100 % de concordância. TCN: 1,5× mais rápida. Medido por *proxy* (1 thread), sem hardware real |
| RNF-81 | Robustez (ruído, sensor faltante, adversarial) | `evaluate_robustness.py` | ❌ | RF: ruído de 5 % σ derruba PR-AUC de 0,94 → 0,23. `DV_pressure` travado zera a detecção nos dois modelos. Viés de 0,25σ esconde a falha do RF |
| RNF-82 | Alucinação ≤ X % | métricas de `benchmark_rag.py` | 🟡 | Sem resposta: LLM puro 100 %, RAG (bge-m3, τ = 0,6) 8,3 %. X a definir com o orientador |
| RNF-83 | Energia/CO₂ | `apps/ml/src/evaluate_energy.py` (CodeCarbon 3.3.1 offline, grade BRA ≈ 98 gCO₂/kWh; CPU por Windows EMI e GPU por NVML **medidos**, RAM estimada e excluída; potência ociosa subtraída) | ✅ | Treino: RF 0,15 Wh, XGB 0,11 Wh, LogReg 0,02 Wh. Inferência ONNX: ≈ 0,09 mWh por 1.000 predições (≈ 0,3 Wh por ativo-ano a 0,1 Hz). **Uma sugestão do LLM (0,072 Wh, 0,007 gCO₂) custa o mesmo que ~800 mil predições** — o gasto do sistema está no RAG, não no modelo preditivo (`models/eval_energy.json`) |
| RNF-84 | Segurança do MCP (prompt injection, red-team) | `safety_guard.py`, `benchmark_redteam.py` | ✅ | Sem guard: o LLM mandou desativar a válvula de segurança em 2/2 casos. Com guard: 0/2, e trechos injetados descartados em 100 % |
| RNF-85 | Conformidade ISO 13374 / ISO 17359 / IEC 62443 | [`conformidade-normas.md`](conformidade-normas.md) | ✅ (documental) | Maior lacuna: autenticação/RBAC (FR1/FR2) |

## Engenharia e produto (segunda rodada)

| Item | Implementação | Status | Evidência |
|---|---|---|---|
| Histórico real (sem mock) | Tabelas `assets`, `failure_events`, `work_orders`, `root_cause_analyses` (migração 0006); `EventTracker` online com RN-03; `/v1/events` (cursor), `/v1/events/{id}` (leituras reais −2 h…+30 min), `/v1/kpis`, `/v1/work-orders`; frontend sem `history-mock.ts` | ✅ | `tests/test_history.py`, `__tests__/history-dashboard.test.tsx`, verificado no navegador |
| KPIs de confiabilidade | MTBF, MTTR, disponibilidade e precisão de alarmes **só com falhas confirmadas na OS**; OEE devolvido como não computável, com justificativa | ✅ | `history_service.kpis` |
| Frota multi-ativo | `FleetService`: N réplicas do MetroPT com offset, ruído, seed e agenda de modos próprios, dados compartilhados em memória | ✅ | `tests/test_fleet_and_rca.py` |
| Auth + RBAC + auditoria | JWT + argon2id, papéis operador < técnico < engenheiro < admin, trilha **encadeada por SHA-256** + trigger Postgres que bloqueia UPDATE/DELETE (migração 0007) | ✅ | `tests/test_auth_rbac_audit.py` (adulteração detectada), trigger verificado em Postgres 17 |
| Ingestão MQTT | Mosquitto com senha; `predictiq/fleet/<ativo>/telemetry` versionado, anti-spoofing tópico×payload, só ativos cadastrados | ✅ | `tests/test_mqtt_telemetry.py`, ponta a ponta com broker real em Linux |
| Digital twin + fusão | `digital_twin.py` (balanço de massa, trabalho elétrico, ciclo de trabalho); `evaluate_twin.py` | ✅ | **Fusão ML ou física detecta 3/3 falhas**, custo 35,0 → 11,3, lead 2,6 h na falha de julho (`models/eval_twin.json`) |
| Model registry + shadow + canary | `register_models.py` (MLflow, aliases champion/challenger, tags do gate RN-05); `shadow_predictions` (migração 0008), `GET /v1/models/shadow`, canary determinístico por hash | ✅ | `tests/test_shadow_canary.py`; registro verificado em MLflow local |
| Tracing distribuído | OpenTelemetry (FastAPI + httpx) → Jaeger, spans de domínio `rag.retrieve`/`rag.generate` com atributos RN-04/RNF-84 | ✅ | `tests/test_tracing.py`; trace recebido por Jaeger real |
| Testes de propriedade e contrato | Hypothesis (invariantes do RN-03, fuzz dos parsers, hash da auditoria); Schemathesis no OpenAPI do histórico | ✅ | Achou 2 bugs reais: status não documentados e **overflow de ID (crash)**, ambos corrigidos |
| Estudo com usuários | Protocolo intra-sujeitos (SUS, NASA-TLX, tempo, caso-armadilha) | ⏳ | [`protocolo-estudo-usuarios.md`](protocolo-estudo-usuarios.md) — exige aprovação do CEP |
| Pacote de replicação | `CITATION.cff`, `.zenodo.json`, script único + manifesto | 🟡 | Falta definir licença, ORCIDs e publicar a release (DOI) |
| Artigo | [`../artigo/rascunho-artigo.md`](../artigo/rascunho-artigo.md) com os números reais | 🟡 | Faltam trabalhos relacionados (PRISMA) e a conclusão |
| TimescaleDB | Migração 0009 (`src/core/timescale.py`): `predictions` vira hypertable (chunks de 1 dia, PK `(id, timestamp)`), compressão segmentada por ativo após 7 dias, agregado contínuo `predictions_hourly`. No-op em Postgres puro/SQLite | ✅ | Benchmark real, 2,59 M linhas (`benchmarks/timescale_benchmark.json`): série horária 30 d **77× mais rápida**, alarmes/dia da frota **398×**, disco **−49 %**; custo: ingestão 3,7× mais lenta (35 k linhas/s ≈ 350 mil ativos a 0,1 Hz). Upgrade → downgrade → upgrade verificado com dados preservados |
| OPC UA (IEC 62541) | `services/opcua_telemetry.py`: gateway simulado (variáveis tipadas por sensor + nó `Sequence` como gatilho) e ingestor por *subscription* que lê a amostra inteira num único `Read`; `TELEMETRY_TRANSPORT=opcua` | 🟡 | `tests/test_opcua_telemetry.py` com servidor e cliente reais (amostra completa, ativo não cadastrado ignorado, `Sequence` repetido descartado, senha errada recusada). Sem criptografia: Basic256Sha256 + certificados fica como lacuna (IEC 62443 FR4) |
| Painel Grafana dos sinais novos | `infra/grafana/dashboards/predictiq-reliability.json` + 7 contadores novos em `core/metrics.py` (RN-04, RNF-84, RF-38, eventos, telemetria, shadow) | ✅ | `tests/test_metrics_second_round.py` garante que o painel só consulta métricas existentes e sem label de alta cardinalidade |

## Auditoria de vazamento e de serving (terceira rodada)

| Achado | Onde | Correção | Evidência |
|---|---|---|---|
| Os 7 modelos legados (RF, XGB, MLP, TCN, BiLSTM, PatchTST, AE) usavam split aleatório estratificado e escolhiam o limiar por F2 **no próprio teste**; RF ainda com SMOTE | `train_*.py`, cards `*_v1` | Todos retreinados pelo protocolo aninhado: `train_production.py` (árvores → `random_forest_v3`, `xgboost_v3`; redes → `*_v2`); champion passa a ser `random_forest_v3` | [`model_evaluation_leakfree.md`](../model_evaluation_leakfree.md) |
| Limiar de evento fixo em 0,7 para qualquer modelo | `EVENT_THRESHOLD` | Evento = alarme persistido **no limiar do próprio modelo** (card); `EVENT_THRESHOLD` vira override opcional | `tests/test_history.py` |
| Modelos de sequência recebiam **uma única linha** e o adapter repetia essa linha 60 vezes: viam uma janela constante | `inference_pipeline._infer_with_history` | Modelos com janela recebem o buffer inteiro | `tests/test_inference_pipeline.py` |
| Buffer de 30 leituras para modelos treinados com janelas de 60 | `feature_buffer.SensorBuffer` | Buffer de 60 (as features das árvores olham no máximo 15 para trás) | `tests/test_feature_buffer.py` |
| Seletor de modelos escondia só a versão base quando havia `_v2` | `frontend/lib/model-name.ts` | Mostra só a maior versão de cada modelo | `__tests__/model-name.test.ts` |

## Protocolo aninhado (quarta rodada) — todos os modelos sob o mesmo rigor

| Lacuna fechada | Implementação | Resultado | Evidência |
|---|---|---|---|
| Hiperparâmetros do XGBoost vinham do Optuna num **split aleatório** (vazamento na seleção) | `nested_protocol.py`: busca dentro do treino de cada fold (leave-one-failure-out), objetivo = custo RN-02 out-of-fold | O RF cai de 3 para 2 falhas sem os hiperparâmetros vazados; o XGBoost escolhe árvores rasas (profundidade 4, 600 árvores) | `models/nested/*.json` |
| Redes com 1 seed e limiar por método diferente do das árvores | Mesmo protocolo para os 8 modelos, 3 seeds; redes em GPU (`apps/ml/.venv-gpu`, torch CUDA) | **XGBoost: 3,3/6 falhas, 12 episódios de falso alarme, custo 46,5** (melhor); RF 2,0/6, custo 71,6 | [`model_evaluation_leakfree.md`](../model_evaluation_leakfree.md) |
| Gate RN-05 com 6 folds | Wilcoxon pareado por fold, custo médio das seeds | XGBoost × RF: melhora mediana +12,8, **p = 0,125 → não promove**; o RF segue champion e o XGBoost, challenger em shadow | idem |
| Redes superconfiantes, probabilidade sem significado | Calibração Platt nas predições out-of-fold; nas redes, embutida no ONNX | ECE já baixo (0,02–0,03) e a calibração **não melhora no teste** (0,028 → 0,033 no RF): o shift entre períodos impede a calibração de transferir | idem |
| Cobertura conformal 0,59–0,68 (não-permutabilidade) | `evaluate_conformal_aci.py`: conformal adaptativo (ACI, Gibbs & Candès 2021), por classe | Cobertura da classe falha **0,12–0,54 (estático) → 0,88–0,91 (ACI)** nos 8 modelos, conjuntos de tamanho médio 1,03–1,26 | `models/eval_conformal_aci.json` |
| RUL avaliado com só 4 falhas | `evaluate_rul.py --pooled`, rótulos de sobrevivência censurados por dataset | XGBoost AFT C-index **0,48** × 0,65 do baseline de calendário: RUL não aprendível com 7 falhas | `models/eval_rul_pooled.json` |
| Conclusões dependem de k, horizonte e razão de custo? | `evaluate_sensitivity.py`: k ∈ {3, 6, 12} × horizonte ∈ {2, 6, 12} h × C_FN/C_FP ∈ {10, 100, 1000}, limiar re-escolhido nas predições out-of-fold | **XGBoost é o melhor em 13 de 27 configurações** (BiLSTM e MLP em 6 cada); ranking moderadamente estável (τ de Kendall mediano 0,50, mínimo 0,21). O número de falhas detectadas depende bastante da configuração (XGBoost 1,7–4,3), por isso RN-01/02/03 precisam ficar declarados | `models/eval_sensitivity.json` |

## Falhas inéditas e diagnóstico real (quinta rodada)

| Pergunta | Implementação | Resultado | Evidência |
|---|---|---|---|
| O sistema prevê uma falha que **nunca viu**? | MetroPT2 (Zenodo 7766691, md5 verificado): 2 falhas **depois** de todo o treino. `ingest_metropt2.py` (mesma harmonização), `evaluate_metropt2.py` pontua os ONNX servidos exatamente como o backend | O MetroPT2 é **outra APU** (TP3 com r ≈ 0 no período comum) → teste duplamente fora da amostra. **Vazamento de ar: os 7 modelos detectam**, sem nenhum falso alarme nas árvores. **Vazamento de óleo: só MLP, TCN, BiLSTM e PatchTST (e o twin físico) detectam, ~57 h antes do alarme LPS do trem**; RF (modelo ativo), XGBoost e autoencoder não | [`metropt2_unseen_failures.md`](../metropt2_unseen_failures.md) |
| A antecedência é real ou um alarme solto? | Métrica "sustentada" pré-declarada: 1º episódio de alarme ≥ 10 min | O "+119 min" da MLP/TCN no vazamento de ar era um alarme isolado de 1–5 min 2 h antes, seguido de 2,5 h de silêncio: não conta como aviso | idem |
| Uma falha do mesmo tipo vista uma vez basta? | `--no-oil-models`: mesma receita aninhada, treinada só antes do vazamento de óleo de 2022 | **Sim, para os modelos temporais.** Sem nenhum exemplo de óleo no treino, **0/7** modelos detectam o vazamento de óleo inédito; com **um** exemplo (produção), MLP/TCN/BiLSTM/PatchTST detectam, alarme **sustentado ~55 h antes do LPS**, em outra APU. Árvores e autoencoder não detectam nem com o exemplo. O vazamento de ar é detectado por 7/7 nas duas variantes | `models/eval_metropt2.json` |
| O que o modelo decide offline é o que a tela mostra ao vivo? | Harness que passa os cenários reais pelo **código exato** do pipeline (simulador → buffer → pré-processador → ONNX), 6 cenários × 7 modelos | **Não era.** O ML detectava ao vivo, mas (1) a tela e o alerta push usavam faixas fixas 0,35/0,65/0,70 para todos os modelos, enquanto os limiares calibrados vão de 0,08 (MLP) a 0,905 (XGBoost): detecções com P = 0,2–0,5 apareciam como NORMAL; (2) modelos de sequência eram pontuados com janela incompleta (padding repetindo a linha mais antiga → falso alarme 19 s após cada início); (3) trocar de cenário emendava duas gravações no buffer. **Corrigido**: risco e alerta relativos ao limiar do modelo (`alert_fires`, `active_threshold`, `getRiskLevel(p, limiar)`), janela real obrigatória (`has_full_window`), buffer iniciado com o histórico real da gravação (`start_scenario`). Com as correções, **32/42** combinações detectadas pelo caminho ao vivo (TCN 6/6, a única que pega também o secador; RF 5/6; MLP, BiLSTM, PatchTST 5/6; XGBoost 4/6; autoencoder 2/6). Verificado no app: BiLSTM no vazamento de ar do MetroPT2 → CRÍTICO no passo 311 (harness previa 307) | `benchmarks/live_replay_matrix.json`, `tests/test_live_serving_parity.py`, `tests/test_model_relative_alerts.py`, `__tests__/model-relative-risk.test.ts` |
| A apresentação mostra isso honestamente? | Simulador reproduz a sequência real de 6 falhas com selo **"vista no treino · demonstração"** × **"inédita · teste honesto"** (`failure_scenarios.py`, `PUT /simulator/scenario`, `RealFailureSection`) | Só as 2 do MetroPT2 são marcadas inéditas (teste automatizado garante) | `apps/backend/tests/test_failure_scenarios.py` |
| O RAG identifica **qual** falha e traz a solução? | `build_diagnosis_cases.py` (evidência = RCA por oclusão do modelo ativo nos 30 min iniciais de cada uma das 9 falhas reais) + `benchmark_rag_diagnosis.py` (relevância por código de tarefa WAP-5, provisória) | **Não.** Só com sensores, secador e óleo identificados 0/3 (acurácia balanceada = acaso); o 0,67 bruto é a taxa da classe majoritária. Em produção (MiniLM, τ = 0,6) o RN-04 **abstém em 100 %**, sem citação inventada; com bge-m3 o LLM responde, mas nomeia o componente errado em 56 %. Causa: sensores compartilhados por todos os componentes + corpus de manutenção *programada*, sem tabela de *fault finding* | [`rag_diagnosis_benchmark.md`](../rag_diagnosis_benchmark.md) |

## Itens refeitos

| Item | O que mudou |
|---|---|
| `docs/model_comparison.md` | Banner com as métricas sem vazamento; critério do RF-10 passa a ser o custo esperado + gate RN-05 |
| RNF-46 | Passa a medir qualidade (RF-36/37), não só latência |
| RF-22 | Gate de fundamentação (RN-04) e *safety guard* (RNF-84) antes e depois do LLM |
| Corpus RAG | Sidecar de curadoria por manual; WAP-5 restrito à seção 6.6 ([`CORPUS-CURATION.md`](../../apps/mcp-server/CORPUS-CURATION.md)) |

## Decisões que dependem do autor/orientador

1. **Trocar o embedder de produção para bge-m3**: melhora muito a recuperação (gabarito em 28,6 % × 2,4 %). RNF-45 **re-medido** na imagem do mcp-server, CPU com 8 threads (`apps/mcp-server/embedder_latency_benchmark.json`): MiniLM p95 20,7 ms · mpnet 41,6 ms · **bge-m3 111,8 ms — passa** (< 500 ms), ~5× mais lento que o MiniLM. Custos que sobram: ~1,5 GB a mais de RAM no container, reindexação de ~4 min (524 trechos) e embeddings de 1024 dimensões. Tecnicamente viável; a troca continua sendo decisão sua.
2. **Validação dos 54 casos do RF-36 por especialista em manutenção.**
3. **Definir o X do RNF-82** (alucinação máxima aceitável).
4. **Conseguir um manual de *fault finding* do compressor real.** É o maior limitador do RAG hoje.
