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
| RN-05 | Promoção só com significância estatística (Wilcoxon pareado, p < 0,05) | `apps/ml/src/promotion_gate.py` | ✅ Gate recusa toda troca como **subpotente** (3 folds → p mínimo 0,125) | `tests/test_promotion_gate.py` |

## Requisitos funcionais

| ID | Requisito | Implementação | Status | Achado principal |
|---|---|---|---|---|
| RF-28 | Validação temporal com *purge*/embargo | `temporal_validation.py` (walk-forward + blocked) | ✅ | PR-AUC do RF cai de 0,9992 (split aleatório) para 0,587 [0,43; 0,74] |
| RF-29 | Avaliação por evento (recall de evento, lead time, FA/dia, range-F1) | `event_metrics.py` | ✅ | A falha de 15/07 (fold 3) não é detectada por XGB nem LR em nenhuma seed |
| RF-30 | RUL por análise de sobrevivência | `rul.py`, `evaluate_rul.py` (XGBoost AFT, censura no corte de treino) | ❌ | C-index 0,49 (≈ acaso) × 0,58 do baseline de renovação: 1–3 falhas não bastam para aprender RUL |
| RF-31 | Incerteza: calibração + conformal | `uncertainty.py` (ECE, isotônica, Mondrian) | 🟡 | ECE 0,05; cobertura conformal da classe falha 0,59–0,68 contra a meta de 0,90 → **não-permutabilidade** temporal |
| RF-32 | Explicabilidade por alerta | Offline: `explainability.py` (TreeSHAP). Online: oclusão por sensor congelada na abertura de cada evento (`root_cause_service.py`), exibida no drawer do Histórico | ✅ | Fiel (H5 ✅); TreeSHAP instável entre falhas (H6 ❌). Em verificação real, a oclusão online também apontou `Oil_temperature`/`TP2` |
| RF-33 | Generalização cross-dataset | — | ⏳ | Requer download do C-MAPSS / MetroPT-2 (autorização pendente) |
| RF-34 | Drift adaptativo | `drift_detectors.py` (Page-Hinkley, ADWIN), `evaluate_drift.py` | ❌ | A regra de produção (PSI > 0,25) gera **28 falsos alarmes/30 dias** em operação normal; calibrados para ≤ 1 alarme/mês, os detectores pegam no máximo 17 % dos drifts ≤ 2σ |
| RF-35 | Diagnóstico XAI → MCP → RAG | `sensor_semantics.py`, `_build_query`/`_build_prompt` | 🟡 | Ganho de ranking depende do embedder: significativo com mpnet (p = 0,004), não com MiniLM/bge-m3 |
| RF-36 | Benchmark RAG anotado (≥ 50 casos) | `benchmarks/rag_cases_v1.json` (42 respondíveis + 12 sem resposta), `benchmark_rag.py` | 🟡 | Anotação em rascunho, **pendente de validação por especialista** |
| RF-37 | Ablação LLM puro × RAG × RAG+XAI, modelos, embedders | `benchmark_rag.py`, `benchmark_rag_retrieval.py` | ✅ | LLM puro alucina em **100 %** das perguntas sem resposta; o RAG em 0–8 %. O embedder de produção recupera o trecho certo em 2,4 % dos casos, o bge-m3 em 28,6 % |
| RF-38 | Human-in-the-loop | `suggestion_feedback_service.py`, `POST /v1/maintenance/feedback`, migração 0005; veredito da OS (causa confirmada / falso alarme) no drawer do Histórico | 🟡 | O veredito da OS vira rótulo dos KPIs e do re-treino. Falta o voto por trecho citado na UI do assistente |

## Requisitos não funcionais

| ID | Requisito | Implementação | Status | Achado principal |
|---|---|---|---|---|
| RNF-78 | Reprodutibilidade total | `scripts/reproduce-thesis.sh`, `scripts/write_manifest.py` | ✅ | Manifesto com commit, imagem Docker, modelos Ollama e SHA-256 de entradas e resultados |
| RNF-79 | Rigor estatístico (≥ 5 seeds, IC 95 %, Friedman/Nemenyi) | `stat_tests.py`, diagramas CD | ✅ | Nenhuma diferença significativa entre modelos (Friedman p ≥ 0,37, N = 3) |
| RNF-80 | Edge / INT8 | `edge_quantization.py` | 🟡 | MLP: −72 % de tamanho, 100 % de concordância. TCN: 1,5× mais rápida. Medido por *proxy* (1 thread), sem hardware real |
| RNF-81 | Robustez (ruído, sensor faltante, adversarial) | `evaluate_robustness.py` | ❌ | RF: ruído de 5 % σ derruba PR-AUC de 0,94 → 0,23. `DV_pressure` travado zera a detecção nos dois modelos. Viés de 0,25σ esconde a falha do RF |
| RNF-82 | Alucinação ≤ X % | métricas de `benchmark_rag.py` | 🟡 | Sem resposta: LLM puro 100 %, RAG (bge-m3, τ = 0,6) 8,3 %. X a definir com o orientador |
| RNF-83 | Energia/CO₂ | — | ⏳ | Requer instalar o CodeCarbon (autorização pendente) |
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

## Itens refeitos

| Item | O que mudou |
|---|---|
| `docs/model_comparison.md` | Banner com as métricas sem vazamento; critério do RF-10 passa a ser o custo esperado + gate RN-05 |
| RNF-46 | Passa a medir qualidade (RF-36/37), não só latência |
| RF-22 | Gate de fundamentação (RN-04) e *safety guard* (RNF-84) antes e depois do LLM |
| Corpus RAG | Sidecar de curadoria por manual; WAP-5 restrito à seção 6.6 ([`CORPUS-CURATION.md`](../../apps/mcp-server/CORPUS-CURATION.md)) |

## Decisões que dependem do autor/orientador

1. **Trocar o embedder de produção para bge-m3**: melhora muito a recuperação, mas o modelo tem 568 M parâmetros (≈ 2,2 GB), contra 118 M do MiniLM. Precisa re-medir o RNF-45 (p95 < 500 ms).
2. **Validação dos 54 casos do RF-36 por especialista em manutenção.**
3. **Autorizar o download dos datasets do RF-33 e a instalação do CodeCarbon (RNF-83).**
4. **Definir o X do RNF-82** (alucinação máxima aceitável).
5. **Conseguir um manual de *fault finding* do compressor real.** É o maior limitador do RAG hoje.
