# Documentação Arquitetural — Sistema Inteligente para Previsão de Falhas em Equipamentos Industriais

**Autor:** Raphael Okuyama  
**Data de geração:** 2026-06-01  
**Repositório:** `projeto-tcc`  

---

## Sumário

1. [Visão Geral da Arquitetura](#1-visão-geral-da-arquitetura)
   - 1.1 [Diagrama Conceitual do Sistema](#11-diagrama-conceitual-do-sistema)
   - 1.2 [Pipeline de Dados — Do Dataset ao Dashboard](#12-pipeline-de-dados--do-dataset-ao-dashboard)
   - 1.3 [Orquestração do Ecossistema via Docker](#13-orquestração-do-ecossistema-via-docker)
2. [Mapeamento e Defesa Tecnológica](#2-mapeamento-e-defesa-tecnológica)
   - 2.1 [Frontend](#21-frontend)
   - 2.2 [Backend](#22-backend)
   - 2.3 [Machine Learning](#23-machine-learning)

---

## 1. Visão Geral da Arquitetura

### 1.1 Diagrama Conceitual do Sistema

O sistema é uma plataforma full-stack de manutenção preditiva organizada em quatro camadas horizontais, comunicando-se de forma assíncrona e em tempo real:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  CAMADA DE APRESENTAÇÃO  (Next.js 16 / React 19 / TypeScript)            │
│  Dashboard Cockpit · Sensor Monitor · Alert Panel · History              │
│  hooks: useSSE() ← SSE  |  useAlertWebSocket() ← WebSocket              │
└────────────────────────────┬─────────────────────────────────────────────┘
                              │ HTTP / SSE / WebSocket (via Nginx)
┌────────────────────────────▼─────────────────────────────────────────────┐
│  CAMADA DE API  (FastAPI + Uvicorn + Starlette)                           │
│  /stream/sensors [SSE]  ·  /ws/alerts [WebSocket]                        │
│  /predict  ·  /predictions  ·  /models  ·  /simulator  ·  /health       │
│                                                                           │
│  InferencePipelineService (asyncio background task)                      │
│    SensorStreamService (pub/sub, 1 Hz) → SensorBuffer (deque 30)         │
│    → MetroPTPreprocessor (feature engineering) → ModelRegistry (ONNX)    │
│    → PredictionService (PostgreSQL) → AlertService (WebSocket push)       │
└────────────────────────────┬──────────────────────────────────────────────┘
                              │ SQLAlchemy async / ONNX Runtime
┌────────────────────────────▼─────────────────────────────────────────────┐
│  CAMADA DE DADOS & MODELOS                                               │
│  PostgreSQL 15  (tabela predictions — histórico de inferências)          │
│  apps/ml/models/*.onnx  (artefatos ONNX servidos em read-only)           │
│  apps/ml/models/*_scaler.joblib  (escaladores StandardScaler)            │
└────────────────────────────┬─────────────────────────────────────────────┘
                              │ Volume Docker (read-only)
┌────────────────────────────▼─────────────────────────────────────────────┐
│  CAMADA DE TREINAMENTO  (apps/ml — Jupyter + PyTorch Lightning)          │
│  MetroPT-3 CSV → Parquet → MetroPTPreprocessor → Treino → ONNX Export   │
│  MLflow (rastreamento de experimentos) · Optuna (HPO do XGBoost)         │
└──────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Pipeline de Dados — Do Dataset ao Dashboard

O pipeline completo percorre as seguintes etapas, detalhadas abaixo em ordem de execução:

#### Etapa 1 — Ingestão e Rotulagem (apps/ml/src/ingest_metropt.py)

O dataset **MetroPT-3** (UCI Repository, ID 791) registra leituras de alta frequência (~1 Hz) de um compressor de ar a bordo de um vagão do Metro do Porto, durante o período de fevereiro a agosto de 2020. O CSV contém aproximadamente **1,5 milhão de linhas** e **15 colunas** (12 sensores analógicos — temperaturas, pressões, corrente de motor, nível de óleo — mais colunas de controle binário).

O script `ingest_metropt.py` realiza:

1. **Download e extração** do ZIP da UCI (ou leitura do arquivo local).
2. **Validação de esquema** — verifica presença das 15 colunas esperadas.
3. **Rotulagem supervisionada** via cruzamento temporal com quatro intervalos de falha documentados no artigo original (vazamentos de ar em abril, maio, junho e julho de 2020). Isso gera a coluna binária `anomaly` (0 = normal, 1 = falha).
4. **Persistência em Parquet** comprimido com Snappy para leitura eficiente em runtime.

O resultado é `data/processed/metropt3.parquet` — arquivo base consumido por todos os scripts de treinamento e pelo `SensorSimulator` em produção.

> **Desequilíbrio de classes:** ~98% das amostras são normais e ~2% representam falhas (ratio ≈ 49:1). Esse desequilíbrio severo motiva estratégias específicas de balanceamento detalhadas na Seção 2.3.

#### Etapa 2 — Feature Engineering (apps/ml/src/preprocessing.py)

A classe `MetroPTPreprocessor` (compatível com a API scikit-learn `BaseEstimator + TransformerMixin`) aplica engenharia de features em dois estágios:

**V1 (34 features):** 12 sensores brutos + 1 delta de pressão (`TP2_delta`) + 7 desvios padrão rolling (janela 5) + 14 médias móveis (janelas 5 e 15).

**V2 (≈ 80 features, extensão para modelos sequenciais):**
- **Cross-sensor**: `TP2_TP3_diff`, `TP2_TP3_ratio`, `work_per_pressure`, `reservoir_drop` — capturam a assinatura física do Air Leak (queda de pressão simultânea ao aumento de corrente do motor).
- **Lags explícitos**: `*_lag_5` e `*_lag_15` — permitem ao classificador observar a trajetória absoluta.
- **Taxa de variação (ROC)**: `*_roc_15` = (agora − 15s atrás) / 15 — derivada suavizada, mais robusta ao ruído que `_delta`.
- **Rolling min/max/range** (janela 15): detecta excursões de pico invisíveis à média.

O mesmo transformador é instanciado identicamente no backend (`apps/backend/src/services/preprocessing.py`), garantindo que as features calculadas em inferência sejam byte-a-byte equivalentes às usadas no treino — um requisito fundamental para que os modelos ONNX não sejam alimentados com distribuições fora do domínio (OOD).

#### Etapa 3 — Treinamento e Exportação ONNX (apps/ml/src/)

Cinco arquiteturas distintas são treinadas, comparadas e exportadas para o formato de intercâmbio **ONNX** (Open Neural Network Exchange):

| Modelo | Script | Tipo | Features | F1 (classe 1) |
|---|---|---|---|---|
| Random Forest | `train_random_forest.py` | Árvores de decisão | 34 (V1) | 0.9703 |
| XGBoost | `train_xgboost.py` | Gradient Boosting | 34 (V1) | 0.9812 |
| MLP | `train_mlp.py` | Rede neural densa | 34 (V1) | ≥ 0.75 |
| TCN | `train_sequential.py --arch tcn` | Conv. dilatada causal | 12 (raw) | 0.8673 |
| BiLSTM | `train_sequential.py --arch bilstm` | LSTM bidirecional | 12 (raw) | 0.8664 |
| PatchTST | `train_sequential.py --arch patchtst` | Transformer de patches | 12 (raw) | ~0.86 |
| Autoencoder | `train_autoencoder.py` | Conv1D não supervisionado | 12 (raw) | — (reconstrução) |

Todos os artefatos gerados são:
- `<modelo>_v1.onnx` + `.onnx.data` — grafo computacional portátil.
- `<modelo>_scaler.joblib` — escalador `StandardScaler` ajustado exclusivamente no conjunto de treinamento.
- `<modelo>_v1_card.json` — cartão de modelo com metadados (features, métricas, threshold de decisão, hiperparâmetros).

#### Etapa 4 — Simulador de Sensores em Tempo Real (apps/backend/src/services/simulator.py)

O `SensorSimulator` é um *data streamer* stateful que **reproduz os dados reais do MetroPT-3** em vez de gerar ruído sintético. Motivo: os modelos treinados no dataset real reconhecem os padrões dinâmicos físicos do compressor (oscilações cíclicas de carga/descarga, correlações entre sensores). Dados Gaussianos aleatórios produzem distribuições fora do domínio de treinamento, causando falsos positivos constantes.

O simulador carrega o Parquet na inicialização, mantendo dois arrays NumPy de forma contígua em memória (`float32`):
- `_normal`: linhas fora dos quatro intervalos de falha conhecidos.
- `_failure`: linhas dentro dos intervalos de falha.

Três modos de operação são suportados:
- **NORMAL**: leitura sequencial circular do array normal.
- **FAILURE**: leitura sequencial circular do array de falha.
- **DEGRADATION**: interpolação linear (`lerp`) entre leituras normais e de falha, com fator de deriva crescendo de 0 → 1 ao longo de 300 passos — simula a degradação gradual do equipamento.

O modo pode ser alterado em tempo de execução via `PUT /api/simulator/mode`, sem reinicialização do serviço.

#### Etapa 5 — Streaming e Inferência Contínua (InferencePipelineService)

Após a inicialização da aplicação FastAPI, o `InferencePipelineService` é iniciado como uma **asyncio.Task** de background que fecha o loop entre o simulador e o modelo:

```
SensorStreamService._broadcast_loop()  (1 Hz, asyncio.Task)
    └── put_nowait(reading) → asyncio.Queue de cada subscriber

InferencePipelineService._run()  (asyncio.Task, loop infinito)
    └── queue.get()
    └── SensorBuffer.append(reading)  [deque(maxlen=30)]
    └── if buffer.is_warm() (≥ 15 amostras):
    │       asyncio.to_thread(→ MetroPTPreprocessor.transform(buffer_df))
    │       asyncio.to_thread(→ ModelService.predict_from_features(last_row))
    └── else:
    │       asyncio.to_thread(→ ModelService.predict(PredictRequest))  [fallback]
    └── save_prediction(db, ...)  [PostgreSQL, sessão própria]
    └── AlertService.process_prediction()
            └── if probability > 0.70:
                    ConnectionManager.broadcast_alert(payload)  [WebSocket]
```

A inferência é despachada via `asyncio.to_thread` para a thread pool do sistema operacional, evitando bloquear o event loop único do FastAPI (que precisa estar livre para atender requisições SSE, WebSocket e REST simultaneamente). A latência ponta-a-ponta (pré-processamento + ONNX + pós-processamento) é medida em cada tick e exposta via o campo `inference_latency_ms` nos alertas WebSocket.

#### Etapa 6 — Transmissão SSE para o Frontend (SensorStreamService)

O mesmo `SensorStreamService` que alimenta o `InferencePipelineService` também serve o endpoint `GET /api/stream/sensors` como **Server-Sent Events (SSE)**. O padrão Pub/Sub utilizado garante:

- O relógio é lido **uma única vez por ciclo**, portanto todos os subscribers recebem exatamente os mesmos dados em sincronia.
- Adição de um novo cliente SSE é O(1) — apenas `set.add()` + `put_nowait()` por ciclo.
- Consumidores lentos são **evictados** (QueueFull) em vez de bloquear o broadcaster.

#### Etapa 7 — Recepção e Visualização no Frontend

O frontend React consome os dois canais de dados em tempo real via hooks customizados:

- **`useSSE`** (`hooks/use-sse.ts`): abre um `EventSource` para `/api/stream/sensors`, parseando cada frame JSON em uma leitura tipada. Implementa *exponential backoff* com jitter ±20% (base 1s, máximo 30s) para reconexão automática.
- **`useAlertWebSocket`** (`hooks/use-alert-websocket.ts`): abre um `WebSocket` para `/ws/alerts`, mantém fila de até 5 alertas (FIFO) com deduplicação por janela de 5 segundos, responde ao heartbeat `ping` com `pong`.

Os dados de sensores alimentam os gráficos Recharts em tempo real; os alertas WebSocket são renderizados no `AlertPanel` com animações de estado crítico.

---

### 1.3 Orquestração do Ecossistema via Docker

O arquivo `docker-compose.yml` define seis serviços isolados em uma rede bridge interna `app-network`:

#### Serviço `nginx` — Gateway e Roteador de Tráfego
Imagem: `nginx:1.25-alpine`. Único ponto de entrada externo (porta 80). O `nginx.conf` implementa roteamento por prefixo de URL com semânticas distintas:

| Rota | Destino | Configuração especial |
|---|---|---|
| `/ws/` | `api:8000` | `Upgrade: websocket`, timeouts 3600s |
| `/api/stream/` | `api:8000/stream/` | `X-Accel-Buffering: no`, chunked off, timeouts 3600s |
| `/api/` | `api:8000/` | REST padrão, timeouts 3600s |
| `/` | `frontend:3000` | WebSocket HMR (`/_next/webpack-hmr`), timeout 180s |

O WebSocket e o SSE requerem configurações especiais no Nginx: o WebSocket exige o header `Upgrade: websocket` e o mapeamento da variável `$connection_upgrade`; o SSE exige a desativação do buffering de proxy (`proxy_buffering off`; `X-Accel-Buffering: no`) para que os eventos sejam entregues ao cliente imediatamente.

#### Serviço `api` — Backend FastAPI
Build: `apps/backend/Dockerfile`. Expõe porta 8000 internamente. O comando de inicialização executa `alembic upgrade head` antes de `uvicorn`, garantindo que as migrações sejam aplicadas antes de aceitar tráfego. O volume `./apps/ml/models` é montado em modo read-only, desacoplando o ciclo de treinamento do ciclo de deploy da API. Limite de memória: 3 GB.

#### Serviço `frontend` — Next.js
Build: `apps/frontend/Dockerfile`. Expõe porta 3000 internamente. O volume `frontend_node_modules` é nomeado para persistir dependências entre restarts e evitar re-instalação. Limite: 4 GB de RAM e 2 vCPUs.

#### Serviço `ml` — Ambiente de Treinamento
Build: `apps/ml/Dockerfile`. Executa Jupyter Notebook na porta 8888. O volume compartilhado `./apps/ml` dá acesso direto ao dataset, aos notebooks e à pasta `models/` onde os artefatos ONNX são escritos.

#### Serviço `db` — PostgreSQL 15
Imagem oficial `postgres:15`. Healthcheck via `pg_isready` garante que a API só inicia após o banco estar pronto (`depends_on: condition: service_healthy`). Dados persistidos em volume nomeado `postgres_data`.

#### Serviço `mlflow` — Rastreamento de Experimentos
Imagem `ghcr.io/mlflow/mlflow:latest`. Usa o próprio PostgreSQL como backend store de metadados e um volume nomeado para artefatos. Porta 5000 exposta para visualização via browser.

**Diagrama de dependências Docker:**

```
nginx ─── api ──── db (healthcheck)
       └─ frontend
              
mlflow ── db (healthcheck)
ml (independente — apenas volume compartilhado com api via /ml/models)
```

---

## 2. Mapeamento e Defesa Tecnológica

### 2.1 Frontend

---

#### Next.js 16 (com App Router)

**Onde foi utilizada:** Toda a camada de apresentação em `apps/frontend/`. Estrutura de rotas em `app/page.tsx` (dashboard principal), `app/history/page.tsx` (histórico) e `app/sensors/[id]/page.tsx` (detalhe de sensor).

**Para que serve:** Framework React com renderização híbrida (Server Components + Client Components), roteamento baseado em sistema de arquivos, suporte nativo a SSE e WebSocket no cliente, e otimizações automáticas de bundle (tree-shaking, code splitting, lazy loading de imagens).

**Como foi implementada:** O layout raiz (`app/layout.tsx`) define o shell da aplicação com sidebar fixa. Componentes de dados em tempo real (gráficos, alertas) são explicitamente marcados com `"use client"` para habilitar hooks do React e acesso ao DOM. Componentes estáticos (header, sidebar, cards de metadados) permanecem como Server Components, reduzindo o JavaScript enviado ao navegador.

**Defesa para a Banca — Por que Next.js e não React puro (Vite/CRA)?**
React puro é uma biblioteca de UI, não um framework. Utilizá-lo em produção requer configuração manual de roteamento (React Router), estratégias de fetch, build toolchain (Webpack/Vite), e não oferece solução integrada para SSR ou SSG. Next.js entrega tudo isso em um único pacote coeso, mantido pela Vercel com suporte de longo prazo. Em projetos acadêmicos de curta duração como um TCC, essa redução de configuração é especialmente relevante — o desenvolvedor foca na lógica de domínio em vez de boilerplate de infraestrutura. Além disso, o App Router do Next.js 13+ adota React Server Components, o modelo de renderização recomendado pelo próprio time do React para produção, reduzindo a quantidade de JavaScript no cliente e melhorando o Time-to-First-Byte.

---

#### TypeScript 5

**Onde foi utilizada:** Em 100% dos arquivos do frontend: hooks, componentes, schemas de API, configurações.

**Para que serve:** Superset tipado de JavaScript que adiciona verificação estática de tipos em tempo de compilação, eliminando uma classe inteira de erros de runtime relacionados a acesso a propriedades inexistentes, incompatibilidades de tipos e chamadas de função com assinatura errada.

**Como foi implementada:** Todos os payloads de API são tipados com `interface` (ex: `WsAlertFrame`, `SensorReading`). O `tsconfig.json` habilita `strict: true`, proibindo implicitamente `any`. O projeto usa `interface` em preferência a `type` para contratos de objeto, seguindo as diretrizes do CLAUDE.md.

**Defesa para a Banca — Por que TypeScript?**
Em um sistema que consome dois canais assíncronos distintos (SSE para leituras de sensores e WebSocket para alertas), a chance de misturar os payloads ou acessar um campo inexistente é alta. TypeScript torna esses erros detectáveis no editor, antes da execução. Academicamente, o uso de TypeScript é o padrão da indústria para qualquer projeto JavaScript de médio porte desde 2020, tendo ultrapassado JavaScript puro em adoção em projetos open-source. A ausência de TypeScript em um TCC de Engenharia seria comparável a não usar tipagem em Java.

---

#### React 19

**Onde foi utilizada:** Biblioteca base de UI; todos os componentes em `components/`.

**Para que serve:** Biblioteca declarativa de componentes para construção de interfaces de usuário reativas. O modelo de reconciliação virtual DOM permite atualizações granulares na tela sem re-renderização completa da página.

**Como foi implementada:** Componentes são funções puras (arrow functions) que recebem props tipadas. `React.memo` é aplicado nos componentes de gráfico (`sensor-chart.tsx`) para evitar re-renders desnecessários durante o streaming de dados. Hooks customizados (`useSSE`, `useAlertWebSocket`, `useSensorData`, `usePredictionHistory`) encapsulam toda lógica de efeitos colaterais.

**Defesa para a Banca — Por que React?**
React possui o maior ecossistema de componentes do mercado frontend (2024: >20M downloads/semana no npm), incluindo Shadcn/ui, Recharts e Radix UI, todos utilizados neste projeto. Alternativas como Vue.js e Svelte teriam ecossistemas de componentes de UI industrial menos maduros. O React 19 também introduz melhorias de desempenho relevantes para streams de dados, com o novo compilador React Compiler reduzindo a necessidade de memoização manual.

---

#### Tailwind CSS 4

**Onde foi utilizada:** Estilização de todos os componentes via classes utilitárias.

**Para que serve:** Framework CSS utility-first que gera exclusivamente as classes utilizadas (PurgeCSS integrado), resultando em bundles CSS mínimos (< 20 KB em produção). Cada propriedade CSS tem uma classe correspondente, eliminando a necessidade de nomenclatura BEM ou CSS Modules.

**Como foi implementada:** O tema customizado é definido em `app/globals.css` via variáveis CSS (`--foreground`, `--background`, `--primary`, etc.), e Tailwind as referencia via `hsl(var(--primary))`. Esta arquitetura é mandatória pelo CLAUDE.md — é proibido usar cores hard-coded (`bg-black`) para garantir consistência temática global.

**Defesa para a Banca — Por que Tailwind e não CSS puro/Styled Components?**
CSS puro em projetos complexos evolui para especificidade crescente e conflitos de seletores difíceis de depurar. Styled Components e CSS-in-JS introduzem overhead de runtime e complicam SSR. O Tailwind resolve esses problemas: é puramente buildtime (zero runtime), integra perfeitamente com Next.js, e sua abordagem utility-first é hoje o padrão de fato em projetos Next.js industriais (adotado por Vercel, GitHub Copilot UI, Linear, Vercel Dashboard).

---

#### Shadcn/ui

**Onde foi utilizada:** Sistema de design principal — componentes `Button`, `Card`, `Badge`, `Skeleton`, `Select`, `Sheet`, `Progress`, `Tooltip` em `components/ui/`.

**Para que serve:** Coleção de componentes acessíveis (compatíveis com WCAG 2.1) construídos sobre Radix UI (comportamento headless) e Tailwind (estilização). Os componentes são **copiados para o repositório** (não importados de um pacote), permitindo customização total sem conflitos de versão.

**Como foi implementada:** Cada componente ocupa um arquivo dedicado (`components/ui/badge.tsx`, `components/ui/button.tsx`, etc.), seguindo o padrão do CLAUDE.md de "um componente por arquivo". O `Skeleton` é utilizado nos estados de loading, o `EmptyState` e `ErrorState` nos casos de falha, garantindo que o sistema nunca exiba telas em branco.

**Defesa para a Banca — Por que Shadcn/ui e não Material UI ou Ant Design?**
Material UI e Ant Design importam o design inteiro como dependência (~300 KB minificados), forçando o projeto a adotar a linguagem visual da Google/Alibaba. O Shadcn/ui entrega apenas os componentes utilizados no código-fonte do projeto, sem overhead, e sem vínculo com uma linguagem visual corporativa externa. Para um dashboard industrial que precisa de identidade visual própria (cockpit, tema escuro, indicadores de criticidade), a flexibilidade do Shadcn/ui é superior.

---

#### Recharts 3

**Onde foi utilizada:** Todos os gráficos do dashboard: série temporal de sensores (`sensor-chart.tsx`), gráfico de radar de equipamentos (`AssetRadarChart.tsx`), gráfico de eficiência (`AssetEfficiencyChart.tsx`), mapa de calor de eventos (`EventHeatmap.tsx`), gráfico de frequência de alertas (`AlertFrequencyChart.tsx`).

**Para que serve:** Biblioteca de visualização de dados declarativa construída sobre D3.js e SVG. Permite criar gráficos compostos (Line, Bar, Area, Radar, Scatter) com responsividade nativa via `ResponsiveContainer`.

**Como foi implementada:** Os gráficos de série temporal atualizam em tempo real consumindo os dados do hook `useSSE`. O componente `SensorChart` usa `React.memo` para evitar re-renders desnecessários quando outros estados do componente pai mudam. O sistema de temas do Recharts é configurado via `components/ui/chart.tsx` para respeitar as variáveis CSS do tema global.

**Defesa para a Banca — Por que Recharts e não Chart.js?**
Chart.js opera sobre Canvas, dificultando a integração com o modelo de componentes do React (imperativo vs declarativo). Recharts é nativo do React, com componentes compostos — cada tipo de série, eixo, tooltip e legenda é um componente React com props tipadas — o que torna o código mais legível e manutenível. Para gráficos de série temporal com atualização em streaming, o modelo declarativo do Recharts também facilita animações suaves de atualização.

---

#### Vitest + Playwright

**Onde foi utilizada:** Vitest em `__tests__/` para testes unitários e de integração de componentes e hooks. Playwright em `e2e/` para testes end-to-end.

**Para que serve:** Vitest é um test runner moderno com HMR nativo e compatível com a API do Jest. Playwright automatiza browsers reais (Chromium, Firefox, WebKit) para validar fluxos completos de usuário.

**Como foi implementada:** Os hooks `useSSE` e `useAlertWebSocket` são testados com mocks de `EventSource` e `WebSocket`. Os testes de componentes usam `@testing-library/react`. Os testes e2e em `dashboard_flow.spec.ts` validam carregamento de KPIs, conexão SSE e exibição do histórico. O `msw` (Mock Service Worker) intercepta chamadas de API nos testes de integração sem necessidade de servidor real.

**Defesa para a Banca:** A pirâmide de testes implementada (unitários com Vitest + e2e com Playwright) demonstra maturidade de engenharia de software. Enquanto testes unitários isolam a lógica de cada hook, os testes Playwright validam o comportamento **observável pelo usuário final** — incluindo a renderização do alerta em estado crítico, que é o fluxo principal do sistema.

---

### 2.2 Backend

---

#### FastAPI 0.115

**Onde foi utilizada:** Toda a camada de API em `apps/backend/src/`. Arquivo principal `src/main.py`, routers em `src/routers/`, services em `src/services/`.

**Para que serve:** Framework web Python assíncrono de alta performance, baseado em Starlette (ASGI) e Pydantic. Gera automaticamente documentação interativa (Swagger UI em `/docs`, ReDoc em `/redoc`) a partir dos schemas Pydantic. Suporta nativamente SSE, WebSocket, dependency injection e middlewares.

**Como foi implementada:** Segue uma **Arquitetura Limpa (Clean Architecture)** em três camadas:
- `routers/`: apenas I/O, autenticação e injeção de dependências via `Depends()`.
- `services/`: toda a lógica de negócio isolada (InferencePipelineService, AlertService, ModelRegistry, etc.).
- `models/`: entidades SQLAlchemy (tabela `predictions`).

O lifecycle da aplicação é gerenciado pelo `lifespan` context manager, que no startup carrega o modelo ONNX e inicia o `InferencePipelineService`, e no shutdown cancela o pipeline e dispõe o pool de conexões do banco.

**Defesa para a Banca — Por que FastAPI e não Flask/Django?**
Flask é síncrono por padrão, exigindo extensões (Flask-SocketIO, gevent) para suportar WebSocket e SSE — o que adiciona complexidade e reduz performance. Django é um framework full-stack orientado a servidores web tradicionais MVC, com overhead de configuração inadequado para uma API pura. FastAPI foi projetado especificamente para APIs assíncronas de alta performance: seu suporte nativo a `async/await`, tipagem via Pydantic, e geração automática de documentação OpenAPI o tornam a escolha tecnicamente superior para este caso de uso. Em benchmarks independentes (TechEmpower Framework Benchmarks), FastAPI supera Flask em até 3–4× em requisições por segundo para endpoints I/O-bound.

---

#### Uvicorn + ASGI

**Onde foi utilizada:** Servidor ASGI que executa a aplicação FastAPI (comando: `uvicorn src.main:app --host 0.0.0.0 --port 8000`).

**Para que serve:** Implementação ASGI de alta performance baseada em `uvloop` (wrapper de `libuv`). Necessário para suportar operações verdadeiramente assíncronas — SSE e WebSocket exigem conexões de longa duração que um servidor WSGI (Gunicorn puro) não suporta eficientemente.

**Defesa para a Banca:** WSGI (Web Server Gateway Interface) é a interface padrão para servidores web Python síncronos, onde cada request ocupa uma thread/processo. ASGI (Asynchronous Server Gateway Interface) permite multiplexar milhares de conexões no mesmo processo via um único loop de eventos. Para este sistema — que precisa manter conexões SSE e WebSocket abertas por horas — ASGI não é uma preferência, é um requisito técnico.

---

#### Pydantic v2

**Onde foi utilizada:** Todos os schemas em `src/schemas/`: `PredictRequest`, `PredictResponse`, `SensorReading`, `AlertPayload`, `SimulatorModeRequest`, etc.

**Para que serve:** Biblioteca de validação de dados via type hints Python. Na v2, foi reescrita em Rust (`pydantic-core`), sendo 5–50× mais rápida que a v1 em benchmarks de validação. Garante que dados externos (JSON de entrada, variáveis de ambiente) sejam validados antes de alcançar a lógica de negócio.

**Como foi implementada:** Cada endpoint FastAPI declara um modelo Pydantic como tipo de entrada e saída. A validação é automática — um POST `/predict` com campo `TP2` ausente retorna 422 Unprocessable Entity com detalhes estruturados antes de executar uma linha do service. As configurações do sistema (`src/core/config.py`) usam `pydantic-settings` para carregar e validar variáveis de ambiente com tipos estritos.

**Defesa para a Banca:** A alternativa seria validação manual via `if "TP2" not in data` — frágil, verbosa e não documentável. Pydantic gera automaticamente o JSON Schema dos modelos, que o FastAPI usa para construir a documentação Swagger. É a solução de validação padrão do ecossistema Python moderno.

---

#### SQLAlchemy 2.0 (Async) + AsyncPG + Alembic

**Onde foi utilizada:** `src/core/database.py` para o engine e session factory. `src/models/prediction.py` para o ORM. `alembic/versions/0001_create_predictions_table.py` para a migração.

**Para que serve:** SQLAlchemy 2.0 com o driver assíncrono `asyncpg` permite executar queries PostgreSQL sem bloquear o event loop do FastAPI. Alembic gerencia migrations de esquema de forma versionada e reversível.

**Como foi implementada:** O engine assíncrono é criado com `create_async_engine` com pool de 10 conexões e `max_overflow=20`. O `AsyncSessionFactory` cria sessões isoladas por operação. O `InferencePipelineService` abre uma sessão própria por leitura processada, garantindo que falhas de persistência não interrompam o fluxo de inferência.

**Defesa para a Banca — Por que PostgreSQL e não SQLite?**
SQLite é single-writer: uma única conexão pode escrever por vez. Com o InferencePipelineService gravando a 1 Hz e múltiplas requisições REST simultâneas lendo o histórico, o SQLite resultaria em contenção e timeouts. PostgreSQL suporta escritas concorrentes via MVCC (Multi-Version Concurrency Control), é o banco relacional open-source de maior adoção em sistemas de produção, e tem suporte nativo a tipos de dados ricos (JSONB, arrays, timestamptz) relevantes para dados de sensores.

---

#### ONNX Runtime 1.18+

**Onde foi utilizada:** Em todos os adaptadores de modelo em `src/services/onnx_*_adapter.py`: `OnnxTreeAdapter` (Random Forest/XGBoost), `OnnxMlpAdapter` (MLP), `OnnxSequenceAdapter` (TCN/BiLSTM/PatchTST), `OnnxAutoencoderAdapter`.

**Para que serve:** Runtime de inferência de alta performance para modelos exportados no formato ONNX. Aplica otimizações automáticas de grafo (fusão de operadores, dobramento de constantes) e suporta múltiplos backends de execução (CPU, CUDA, TensorRT). Para CPU single-row inference, aplica o modo sequencial (`ORT_SEQUENTIAL`) com 1 thread intra-op, minimizando o overhead de scheduling.

**Como foi implementada:** Cada sessão ORT é configurada com:
```python
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
sess_opts.intra_op_num_threads = 1
```

Os modelos são carregados como singletons no lifespan do FastAPI — não há I/O de disco durante a inferência. A inferência CPU-bound é executada via `asyncio.to_thread` para liberar o event loop.

**Defesa para a Banca — Por que ONNX e não pickle/joblib para os modelos PyTorch?**
Modelos PyTorch salvos em `.pt` ou `.ckpt` exigem PyTorch instalado no servidor de inferência — uma dependência de ~2 GB que inclui compiladores CUDA, bindings C++, etc. O ONNX é um formato de intercâmbio portátil: o grafo computacional serializado pode ser executado pelo ONNX Runtime em qualquer linguagem (Python, C++, Java, C#) sem a dependência do framework de treinamento. Isso separa o ambiente de treinamento (`apps/ml` com PyTorch + Lightning) do ambiente de inferência (`apps/backend` com apenas `onnxruntime`), reduzindo a imagem Docker da API de ~5 GB para ~800 MB.

---

#### ModelRegistry + Hot-Swap Atômico

**Onde foi utilizada:** `src/services/model_registry.py`. Consumido por todos os routers via `Depends(get_model_registry)`.

**Para que serve:** Gerenciador in-process que mantém exatamente um `ModelService` ativo por vez, com capacidade de troca a quente (hot-swap) sem reinicialização do servidor e sem janela de indisponibilidade.

**Como foi implementada:** Utiliza uma estratégia de dois estágios:
1. **Fora do lock** (lento): o novo `ModelService` é inicializado completamente em uma thread separada via `asyncio.to_thread` — requisições em voo continuam usando o modelo antigo.
2. **Dentro do lock** (nanossegundos): apenas a atribuição de referência `self._service = new_service` é executada sob `asyncio.Lock`. A GIL do CPython garante que atribuições de referência sejam atômicas.

O endpoint `PUT /api/models/active` aciona o swap em runtime. O `KNOWN_MODELS` frozenset define os 9 modelos válidos: `random_forest`, `xgboost`, `mlp`, `random_forest_v2`, `xgboost_v2`, `tcn`, `bilstm`, `patchtst`, `autoencoder`.

**Defesa para a Banca:** Esta é uma implementação do padrão **Read-Copy-Update (RCU)** adaptada para Python: leitores nunca bloqueiam, escritores pagam o custo do carregamento fora do lock crítico. Em sistemas de produção de ML, a troca de modelo sem downtime é um requisito operacional essencial — por exemplo, para rollback imediato após a detecção de degradação de performance.

---

#### Structlog

**Onde foi utilizada:** `src/core/logging.py`. Importado em todos os services e routers via `structlog.get_logger(__name__)`.

**Para que serve:** Biblioteca de logging estruturado que emite logs em formato JSON (ou console colorido em desenvolvimento). Cada entrada de log é um dicionário tipado com campos consistentes (`event`, `level`, `timestamp`, mais campos de contexto específicos do evento).

**Defesa para a Banca:** Logs de texto não estruturados (`print("Model loaded")`) são difíceis de consultar em sistemas de observabilidade (Elasticsearch, Loki, Datadog). O structlog emite JSON parseável por qualquer stack de agregação de logs, permitindo queries como `{event="inference_completed", latency_ms>100}` — essencial para diagnóstico em produção.

---

#### SlowAPI (Rate Limiting)

**Onde foi utilizada:** `src/core/rate_limit.py`. Middleware `SlowAPIMiddleware` registrado em `create_app()`.

**Para que serve:** Rate limiter baseado no IP do cliente para proteger endpoints custosos (especialmente `POST /predict`) contra uso abusivo.

**Defesa para a Banca:** Um endpoint de inferência ML que processa features e executa um grafo ONNX consome CPU mensurável. Sem rate limiting, um cliente malicioso ou bugado pode monopolizar os recursos do servidor, degradando a experiência de todos os usuários legítimos. O `SlowAPI` implementa o algoritmo de token bucket sobre o `asyncio` event loop — sem overhead de Redis para cargas single-node.

---

### 2.3 Machine Learning

---

#### Dataset MetroPT-3

**Onde foi utilizada:** `apps/ml/data/raw/MetroPT3(AirCompressor).csv` (source) e `apps/ml/data/processed/metropt3.parquet` (formato de trabalho).

**Para que serve:** Dataset real de um compressor de ar a bordo do Metro do Porto (Portugal), coletado pela Universidade do Porto e publicado no UCI Machine Learning Repository (ID 791). Contém ~1,5 milhão de leituras horárias de 12 sensores analógicos (pressões, temperaturas, corrente, nível de óleo, fluxo de ar) com resolução de 1 Hz, cobrindo quatro eventos de falha documentados (vazamentos de ar).

**Defesa para a Banca — Por que MetroPT-3 e não dados sintéticos?**
Datasets sintéticos gerados por distribuições Gaussianas ou Markov chains produzem modelos que não generalizam para condições reais de operação industrial. O MetroPT-3 contém as dinâmicas físicas reais do compressor: oscilações de pressão cíclicas (carga/descarga), correlações entre sensores que surgem da física do sistema (queda de pressão implica aumento de corrente do motor), e transientes de startup/shutdown. Modelos treinados em dados reais e avaliados em dados reais têm validade científica que dados sintéticos não podem fornecer. A disponibilidade pública do dataset também garante reprodutibilidade dos experimentos.

---

#### Scikit-learn (Random Forest + Pré-processamento)

**Onde foi utilizada:** `apps/ml/src/train_random_forest.py` (classificador), `apps/ml/src/preprocessing.py` (MetroPTPreprocessor). `apps/backend/src/services/onnx_tree_adapter.py` (inferência via ONNX).

**Para que serve:** Biblioteca de ML clássico com implementações otimizadas em Cython/NumPy de árvores de decisão, ensemble methods, e utilitários de pré-processamento. A API `Pipeline` do scikit-learn permite encadear transformadores e classificadores de forma reproduzível.

**Como foi implementada:** O `RandomForestClassifier` é treinado com SMOTE para balanceamento de classes (ratio 1:1 após oversampling). O `MetroPTPreprocessor` herda de `BaseEstimator + TransformerMixin` para compatibilidade com a API `fit/transform`. O modelo final é exportado para ONNX via `skl2onnx`.

**Métricas no hold-out set (20%):**
- F1-Score (classe 1 / falha): **0.9703**
- Recall (classe 1): **0.9973**  
- AUC-ROC: **1.0000**

**Defesa para a Banca — Por que Random Forest como baseline?**
Random Forest é um método de ensemble que combina múltiplas árvores de decisão com amostragem aleatória de features (bagging), reduzindo variância sem aumentar viés significativamente. É intrinsecamente robusto a overfitting, não requer normalização de features, é interpretável via feature importance, e apresenta performance competitiva em dados tabulares industriais. Academicamente, é o baseline obrigatório para justificar a adopção de modelos mais complexos — se XGBoost ou redes neurais não superarem o Random Forest de forma consistente, a complexidade adicional não se justifica.

---

#### XGBoost 3.2 + Optuna

**Onde foi utilizada:** `apps/ml/src/train_xgboost.py` (treinamento + HPO), `apps/ml/data/optuna/xgboost_study.db` (SQLite persistente do estudo).

**Para que serve:** XGBoost implementa Gradient Boosted Trees com o algoritmo de histograma aproximado (`tree_method="hist"`), resultando em treino 5–10× mais rápido que o método exato em datasets > 100k linhas. Optuna é um framework de hyperparameter optimization (HPO) baseado em *bayesian optimization* (sampler TPE), mais eficiente que grid search ou random search ao navegar o espaço de hiperparâmetros.

**Como foi implementada:** O estudo Optuna executa 100 trials otimizando 7 hiperparâmetros (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`). Cada trial avalia um F1-Score (classe 1) via 3-fold CV em 200k amostras estratificadas (para manter < 30s por trial). O `scale_pos_weight = 49.64` (ratio majority/minority) substitui SMOTE, pois o XGBoost processa imbalanceamento nativamente durante o cálculo de gradientes, eliminando o overhead de memória do SMOTE (~1 GB de amostras sintéticas).

**Melhores métricas (100 trials, limiar F2-ótimo 0.6068):**
- F1-Score (classe 1): **0.9812**
- Recall (classe 1): **0.9960**
- AUC-ROC: **1.0000**
- Latência single-row p50/p95: **0.263 ms / 0.486 ms**

**Defesa para a Banca — Por que XGBoost e não Random Forest como modelo principal?**
Apesar do Random Forest apresentar F1 ligeiramente superior (0.9703 vs o 0.9302 sem threshold tuning), o XGBoost com limiar F2-ótimo alcança **0.9812** — superando o Random Forest. Mais importante: a latência de inferência do XGBoost (p50 = 0.263 ms) é **8–19×** menor que a do Random Forest (p50 = 2–5 ms) para uma única linha, e o artefato ONNX é **3× menor** (~35 MB vs ~120 MB). Em ambientes de edge computing ou sistemas embarcados — contexto industrial realista — estas diferenças são determinantes. A combinação XGBoost + Optuna demonstra boas práticas de MLOps: rastreabilidade do processo de HPO via SQLite, reprodutibilidade via `random_state=42`, e threshold calibrado via F2-score (favorece recall, minimizando falsos negativos em detecção de falhas).

---

#### PyTorch 2.3 + PyTorch Lightning

**Onde foi utilizada:** `apps/ml/src/models/` (TCN, BiLSTM, PatchTST, Autoencoder), `apps/ml/src/train_sequential.py`, `apps/ml/src/train_autoencoder.py`.

**Para que serve:** PyTorch é o framework de Deep Learning de maior adoção na pesquisa acadêmica (>70% dos papers de ML em 2024 segundo Papers with Code). PyTorch Lightning é uma abstração que remove o boilerplate dos training loops (`Trainer.fit()`), callbacks padronizados (`EarlyStopping`, `ModelCheckpoint`), e logging integrado (MLflow).

**Como foi implementada:** Três arquiteturas sequenciais são implementadas como `LightningModule`:

- **TCN (Temporal Convolutional Network)**: 6 blocos dilatados causais com dilatações 1, 2, 4, 8, 16, 32. Campo receptivo = 253 amostras (cobre toda a janela de 60s). Cada bloco aplica duas Conv1D + BatchNorm + ReLU + Dropout com conexão residual. Global Average Pooling colapsa o eixo temporal. *Referência: Bai et al., 2018, arXiv:1803.01271.*

- **BiLSTM + Atenção Aditiva**: 2 camadas LSTM bidirecionais (hidden=64). A atenção aditiva (Bahdanau et al., 2015) computa pesos de importância por timestep, produzindo um vetor de contexto ponderado. Os pesos de atenção são o *hook de interpretabilidade* — revelam quais segundos da janela influenciaram a classificação.

- **PatchTST (channel-mixing)**: Janela dividida em 9 patches sobrepostos de 12 timesteps. Cada patch é o vetor achatado `(n_channels × patch_len)` projetado para `d_model=64`. Token [CLS] learnable + embedding posicional. TransformerEncoder com 3 camadas e 4 cabeças de atenção. *Referência: Nie et al., ICLR 2023, arXiv:2211.14730.* A variante **channel-mixing** (em oposição ao channel-independent original) foi escolhida porque as assinaturas de falha industriais são inerentemente conjuntas: Air Leak = TP2 cai *enquanto* Motor_current sobe.

Todos os modelos usam:
- Otimizador **AdamW** com weight decay = 1e-4.
- Scheduler **CosineAnnealingLR** para decaimento suave da taxa de aprendizado.
- **EarlyStopping** monitorando `val_f1` com paciência 10 — previne overfitting sem especificar épocas manualmente.
- **pos_weight** inversamente proporcional ao ratio de classes para tratar desequilíbrio.

**Defesa para a Banca — Por que modelos sequenciais além do Random Forest/XGBoost?**
Random Forest e XGBoost tratam cada snapshot de sensores como um vetor independente — a ordem temporal é capturada apenas pelas features rolling/lag engenheiradas manualmente. Modelos sequenciais (TCN, BiLSTM, PatchTST) aprendem automaticamente dependências temporais de ordem arbitrária diretamente do sinal bruto, sem engenharia de features. Isso torna o pipeline mais robusto a mudanças de distribuição e permite detectar padrões temporais sutis invisíveis a features de janela fixa. A inclusão dessas arquiteturas posiciona o TCC em linha com o estado da arte em *fault detection* industrial (2022–2024), onde transformers e TCNs têm substituído métodos clássicos nos benchmarks públicos.

---

#### SMOTE (imbalanced-learn)

**Onde foi utilizada:** `apps/ml/src/balancing.py` (MetroPTBalancer), consumido por `train_random_forest.py` e opcionalmente pelos scripts sequenciais com flag `--apply-smote`.

**Para que serve:** *Synthetic Minority Over-sampling TEchnique* — gera amostras sintéticas da classe minoritária (falha) por interpolação k-NN no espaço de features, em vez de simplesmente duplicar amostras existentes (oversampling ingênuo) ou descartar amostras da maioria (undersampling).

**Como foi implementada:** O `MetroPTBalancer` expõe `fit_resample(X_train, y_train)` — o método é intencionalmente limitado ao conjunto de treinamento via assinatura pública, prevenindo vazamento de dados. A estratégia de amostragem é configurável via variável de ambiente `SMOTE_SAMPLING_STRATEGY`. Um método estático `train_test_split_safe` garante que o split ocorra antes de qualquer resampling — contrato explícito anti-leakage.

**Defesa para a Banca — SMOTE vs scale_pos_weight:** Para o Random Forest, SMOTE aumenta a densidade de amostras de falha na fronteira de decisão, melhorando o recall da classe minoritária sem modificar o algoritmo de aprendizado. Para o XGBoost, `scale_pos_weight` é preferível porque o algoritmo já tem suporte nativo a imbalanceamento via upweighting dos gradientes da classe minoritária — SMOTE adicionaria ~1 GB de amostras sintéticas sem ganho de performance.

---

#### MLflow

**Onde foi utilizada:** `apps/ml/src/train_sequential.py` e `apps/ml/src/train_mlp.py` (logging de experimentos). Serviço `mlflow` no `docker-compose.yml`.

**Para que serve:** Plataforma de MLOps para rastreamento de experimentos de ML: registra hiperparâmetros, métricas, artefatos (modelos ONNX, escaladores) e código-fonte de cada run. Permite comparar visualmente múltiplos treinos e reproduzir qualquer run histórica.

**Como foi implementada:** O `MLFlowLogger` do PyTorch Lightning é usado com fallback gracioso: se o servidor MLflow estiver indisponível, o treinamento prossegue sem rastreamento (comportamento não-fatal). Métricas registradas por run: `test_f1_class1`, `test_roc_auc`, `latency_p50_ms`, `latency_p95_ms`, `onnx_max_abs_diff`, `decision_threshold`. Artefatos registrados: `<arch>_v1.onnx`, `<arch>_scaler.joblib`, `<arch>_v1_card.json`.

**Defesa para a Banca — Por que MLflow?**
Sem rastreamento de experimentos, comparar 7 arquiteturas treinadas com hiperparâmetros diferentes é feito manualmente via planilhas — propenso a erros e não reproduzível. MLflow fornece um servidor de UI (porta 5000) que apresenta todas as runs em uma tabela comparativa, com gráficos de métricas por epoch. É o padrão open-source de fato para MLOps, adotado por empresas como Databricks, Microsoft Azure ML e AWS SageMaker.

---

#### ONNX + skl2onnx + onnxmltools (Exportação)

**Onde foi utilizada:** Em todos os scripts de treinamento como etapa final de exportação.

**Para que serve:** ONNX (Open Neural Network Exchange) é o formato padrão aberto para intercâmbio de modelos de ML entre frameworks. A exportação via `torch.onnx.export` (modelos PyTorch) e `skl2onnx`/`onnxmltools` (scikit-learn/XGBoost) converte o grafo computacional Python para uma representação portátil independente de framework.

**Como foi implementada:** Para os modelos PyTorch sequenciais, é realizada uma **verificação de equivalência pós-exportação**: a saída do modelo PyTorch e a saída do grafo ONNX são comparadas com tolerância absoluta de 1e-5 (TCN/BiLSTM) ou 1e-4 (PatchTST, que usa scaled_dot_product_attention). Se a divergência exceder a tolerância, o script falha com `AssertionError` — garantindo que o artefato ONNX é funcionalmente idêntico ao modelo treinado. O `OnnxSequenceAdapter` no backend aplica as mesmas otimizações de grafo (`ORT_ENABLE_ALL`) e execução sequencial single-thread para minimizar latência p95.

**Defesa para a Banca:** A verificação de equivalência PyTorch↔ONNX é uma prática de qualidade de MLOps raramente implementada em TCCs, que demonstra consciência dos riscos de exportação. Um grafo ONNX divergente produziria probabilidades incorretas em produção sem qualquer erro visível — o modelo "funcionaria" mas daria respostas erradas. A verificação automática torna esse risco detectável no momento do treino.

---

#### Autoencoder Convolucional (Detecção Não Supervisionada)

**Onde foi utilizada:** `apps/ml/src/models/autoencoder.py`, `apps/ml/src/train_autoencoder.py`. Adaptador de inferência: `apps/backend/src/services/onnx_autoencoder_adapter.py`.

**Para que serve:** Rede neural encoder-decoder com convoluções 1D que aprende a reconstruir sequências de sensores normais com erro mínimo. Em produção, o **erro de reconstrução** (MSE ou MAE entre entrada e saída) serve como score de anomalia: sequências de falha que diferem da distribuição normal produzem erros de reconstrução elevados.

**Como foi implementada:** Treinado exclusivamente em amostras normais (aprendizado não supervisionado). A ausência de labels de falha no treinamento torna este modelo aplicável a tipos de falha não vistos durante o treinamento — diferente dos classificadores supervisionados que só detectam falhas semelhantes às do MetroPT-3.

**Defesa para a Banca — Valor do Autoencoder num contexto supervisionado:**
Os classificadores supervisionados (RF, XGBoost, modelos sequenciais) requerem labels de falha para treinamento. Em equipamentos industriais novos ou com histórico de manutenção incompleto, labels confiáveis podem ser escassos. O Autoencoder oferece uma alternativa complementar: é treinado apenas em operação normal, e qualquer desvio da normalidade produz anomalia detectável. A combinação dos dois paradigmas (supervisionado + não supervisionado) aumenta a cobertura de detecção e reduz a dependência de dados rotulados de falha.

---

*Fim do documento — gerado em 2026-06-01.*
