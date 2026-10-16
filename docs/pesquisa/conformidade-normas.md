# Mapeamento de conformidade — RNF-85

Mapeia os componentes do PredictIQ para três normas de referência. O objetivo
**não** é declarar certificação (isso exige auditoria formal), e sim mostrar
à banca **onde** a arquitetura se alinha a cada norma e **onde há lacunas
conhecidas**. Cada linha aponta o artefato concreto no repositório.

Legenda: ✅ atende · 🟡 atende parcialmente · ❌ lacuna declarada.

## 1. ISO 13374 — Condition monitoring and diagnostics of machines (data processing)

A ISO 13374-1/-2 organiza um sistema de CBM em seis blocos de processamento.
O PredictIQ cobre todos os seis:

| Bloco ISO 13374 | Função | Implementação | Status |
|---|---|---|---|
| DA — Data Acquisition | Coletar sinais dos sensores | Replay do MetroPT-3 e frota simulada; **ingestão MQTT** com broker autenticado e validação de tópico/payload (`mqtt_telemetry.py`) | 🟡 Transporte industrial real (MQTT); a fonte ainda é simulada. OPC UA não implementado |
| DM — Data Manipulation | Filtrar, extrair *features* | `MetroPTPreprocessor` (80 features: janelas, *lags*, razões entre sensores), `feature_buffer.py` | ✅ |
| SD — State Detection | Normal × anormal | Classificadores ONNX (RF, XGB, DL), autoencoder não supervisionado; persistência k = 6 (RN-03, `event_metrics.persistent_alarms`) | ✅ |
| HA — Health Assessment | Diagnóstico: qual componente e por quê | SHAP agregado por sensor (`apps/ml/src/explainability.py`) + vocabulário de componente (`sensor_semantics.py`) | 🟡 Explicação validada offline (fidelidade H5); ainda não exibida por alerta no dashboard |
| PA — Prognostic Assessment | Estimar o tempo até a falha | RUL por análise de sobrevivência (`apps/ml/src/rul.py`, XGBoost AFT) | ❌ Implementado mas **não aprovado**: C-index ≈ 0,5 com 1–3 falhas de treino (`models/eval_rul.json`) |
| AG — Advisory Generation | Recomendar a ação de manutenção | RAG via MCP + LLM local com gate de fundamentação (RN-04) e checagem de citações | 🟡 Funciona, mas o corpus atual cobre pouco (cobertura gold 2,4 % com o embedder de produção — `docs/rag_benchmark.md`) |

Ligado à ISO 13374-2 (comunicação): os blocos conversam por interfaces
explícitas (REST/SSE/WebSocket documentados em OpenAPI, MCP para a
recuperação). Não há implementação do esquema MIMOSA OSA-CBM.

## 2. ISO 17359 — Condition monitoring: general guidelines

A ISO 17359 descreve o **procedimento** para implantar monitoramento de
condição. Mapa das etapas para as evidências do projeto:

| Etapa ISO 17359 | Evidência | Status |
|---|---|---|
| Análise de custo-benefício | Custo assimétrico explícito C_FN = 100 · C_FP (RN-02) e custo esperado por modelo (`docs/temporal_evaluation.md`) | 🟡 A razão de custos é suposição, sem dado financeiro real |
| Auditoria do equipamento e criticidade | Compressor da APU: falha de *air leak* imobiliza o trem (Veloso et al., 2022) | 🟡 Não há FMECA formal |
| Seleção de modos de falha e parâmetros a monitorar | 12 sensores do MetroPT-3; ranking SHAP por falha mostra `Oil_temperature` e `DV_pressure` recorrentes (`models/explain_events.json`) | ✅ |
| Seleção de técnica de medição e intervalo | Amostragem de 10 s (dataset); limiar e persistência calibrados sem usar o teste | ✅ |
| Aquisição e análise de dados | Ver ISO 13374 DA/DM/SD | ✅ |
| Determinação da ação de manutenção | Assistente RAG (RF-22/RF-35) | 🟡 |
| Revisão e melhoria contínua | Drift diário (RF-27), estudo de drift adaptativo (RF-34), feedback do técnico (RF-38) | ✅ |

## 3. IEC 62443 — Segurança de sistemas de automação industrial

Mapeamento para os sete *Foundational Requirements* (IEC 62443-3-3):

| FR | Requisito | Implementação | Status |
|---|---|---|---|
| FR1 — IAC | Identificação e autenticação | Contas locais, senha argon2id, JWT com expiração (`src/core/security.py`); a API recusa subir com auth ligada e segredo placeholder | 🟡 Sem MFA nem integração com diretório (LDAP/SSO) |
| FR2 — UC | Controle de uso | RBAC hierárquico operador < técnico < engenheiro < admin (`require_role`) | ✅ |
| FR3 — SI | Integridade do sistema | `pip-audit` sem HIGH/CRITICAL (RNF-62), OWASP ZAP, hash SHA-256 dos manuais indexados (RF-20), imagens Docker | ✅ |
| FR4 — DC | Confidencialidade dos dados | Sanitização de logs (`log_sanitizer.py`, token do bot nunca logado); segredos só via `.env` | 🟡 Sem TLS interno entre containers |
| FR5 — RDF | Fluxo de dados restrito | Rede Docker interna; MCP e Ollama não expostos; Nginx como único ponto de entrada | ✅ |
| FR6 — TRE | Resposta oportuna a eventos | Logs estruturados, Prometheus, **tracing OpenTelemetry → Jaeger**, alertas Telegram/e-mail, **trilha de auditoria imutável** (hash encadeado + trigger) | 🟡 Sem SIEM |
| FR7 — RA | Disponibilidade de recursos | Rate limiting (slowapi), fila Celery para notificações, *health checks* | ✅ |

**Superfície específica de LLM/MCP** (complementa a IEC 62443, que não
trata LLMs): conteúdo de manual tratado como dado, nunca como instrução
(regra 5 do *system prompt*); citações checadas contra os trechos recuperados
(RN-04); o teste de *red-team* de *prompt injection* via manual malicioso é
o RNF-84.

## Lacunas priorizadas

1. **FR1 — MFA / SSO** (RBAC e auditoria já implementados; `AUTH_ENABLED` precisa ser ligado em produção).
2. **PA — prognóstico**: o RUL não é confiável com os dados disponíveis. Precisa de dataset *run-to-failure* (RF-33).
3. **DA — aquisição de campo**: o transporte MQTT existe; falta um gateway real (ou OPC UA) em vez do simulador.
4. **AG — corpus**: falta manual de *fault finding* do equipamento real.
