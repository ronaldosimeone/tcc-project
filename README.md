# PredictIQ — Plataforma de Manutenção Preditiva Industrial

> **TCC · Indústria 4.0 · v3.0** — Sistema full-stack para detecção de falhas em tempo real no compressor MetroPT-3, com streaming de dados reais (1 Hz), pipeline de Deep Learning sequencial, Autoencoder não-supervisionado, dashboard interativo, hot-swap atômico de modelos e infraestrutura MLOps via MLflow + ONNX Runtime.

---

## Sumário

1. [Visão Geral](#1-visão-geral)
2. [Stack Tecnológica](#2-stack-tecnológica)
3. [Arquitetura do Monorepo](#3-arquitetura-do-monorepo)
4. [Fluxo de Dados em Tempo Real](#4-fluxo-de-dados-em-tempo-real)
5. [Modelos de Machine Learning](#5-modelos-de-machine-learning)
6. [Pré-requisitos](#6-pré-requisitos)
7. [Variáveis de Ambiente](#7-variáveis-de-ambiente)
8. [Como Rodar (Docker Compose)](#8-como-rodar-docker-compose)
9. [Banco de Dados & Migrations](#9-banco-de-dados--migrations)
10. [Endpoints da API](#10-endpoints-da-api)
11. [Frontend — Dashboard de Frota](#11-frontend--dashboard-de-frota)
12. [Pipeline de Machine Learning](#12-pipeline-de-machine-learning)
13. [MLOps com MLflow](#13-mlops-com-mlflow)
14. [Testes](#14-testes)
15. [Demo do Simulador](#15-demo-do-simulador)
16. [Troubleshooting](#16-troubleshooting)
17. [Autores & Licença](#17-autores--licença)

---

## 1. Visão Geral

O **PredictIQ** é um sistema completo de **manutenção preditiva** para o compressor de ar do metrô de Porto (dataset oficial **MetroPT-3** da UCI). Ele integra três aplicações independentes em um monorepo, orquestradas via Docker Compose:

| Aplicação | Papel |
|---|---|
| `apps/backend` | API FastAPI com Clean Architecture: I/O em `routers/`, regras em `services/`, ORM em `models/`. Roda o pipeline de inferência contínuo, expõe SSE, WebSocket e endpoints REST. |
| `apps/frontend` | Dashboard Next.js 16 com Server/Client Components, telemetria SSE @1 Hz, radar chart com normalização absoluta, alertas WebSocket e modo LIVE para o ativo `APU-Trem-042`. |
| `apps/ml` | Pipeline de treinamento, exportação ONNX (opset 17), tracking via MLflow e script de promoção do melhor modelo para produção. |

**Diferenciais técnicos**:

- **Hot-Swap atômico de modelos** (`PUT /models/active`) sem reiniciar o servidor — usa protocolo de duas fases (load fora do lock, swap dentro do lock).
- **9 modelos servidos via ONNX Runtime**: Random Forest, XGBoost, MLP, RF v2, XGBoost v2, TCN, BiLSTM, PatchTST e Conv1D **Autoencoder não-supervisionado**.
- **Simulador real**: replay sequencial do parquet MetroPT-3 (~1.5M linhas) com modos `NORMAL`, `DEGRADATION` (interpolação vetorizada) e `FAILURE` (4 janelas reais de air-leak).
- **Pipeline de inferência contínuo**: `SensorStreamService → SensorBuffer (deque) → MetroPTPreprocessor → ModelService → DB + WS`.
- **Reverse proxy Nginx** configurado para SSE/WebSocket com `proxy_read_timeout 3600s`.

---

## 2. Stack Tecnológica

| Camada | Tecnologia | Versão | Responsabilidade |
|---|---|---|---|
| **Frontend** | Next.js | 16.2.3 | App Router, Server Components |
| | React | 19.2.4 | UI |
| | TypeScript | 5.x (strict) | Tipagem |
| | Tailwind CSS | 4.x | Estilização |
| | shadcn/ui + Radix UI | latest | Componentes |
| | Recharts | 3.8 | Radar, line charts, gauge |
| | pnpm | latest (via Corepack) | Gerenciador de pacotes |
| | Vitest + Playwright | 4.1 / 1.59 | Unit + E2E |
| | MSW | 2.13 | Mock Service Worker (testes) |
| **Backend** | FastAPI | 0.115 | Framework HTTP/SSE/WS |
| | Pydantic v2 + pydantic-settings | 2.12 / 2.9 | DTOs e config |
| | SQLAlchemy (async) | 2.0.40 | ORM |
| | asyncpg | 0.30 | Driver Postgres async |
| | Alembic | 1.18 | Migrations |
| | structlog | 25.3 | Logging JSON |
| | slowapi | 0.1.9 | Rate limiting |
| | Celery | 5.4 | Fila assíncrona de notificações (RNF-50/RNF-51) |
| | Redis | 7 (Docker) | Broker do Celery (só transporte de mensagens — não é o rate limiter) |
| | ONNX Runtime | ≥1.18 | Servir todos os modelos DL |
| | PyArrow + pandas | 14.x / 3.x | Leitura do parquet do simulador |
| **Banco** | PostgreSQL | 15 (Docker) | Histórico de predições (`predictions`) |
| **ML / Treino** | scikit-learn | 1.8 | RF + pipelines |
| | XGBoost | ≥2.1 | Gradient Boosting |
| | Optuna | ≥3.6 | Tuning de hiperparâmetros |
| | imbalanced-learn | 0.14 | SMOTE para classe minoritária |
| | PyTorch + Lightning | ≥2.3 | MLP, TCN, BiLSTM, PatchTST, Autoencoder |
| | torchmetrics | ≥1.4 | Métricas durante treino |
| | skl2onnx / onnxmltools | ≥1.17 / ≥1.12 | Export ONNX para árvores |
| **MLOps** | MLflow | ≥2.14 (Docker) | Tracking + promotion |
| **Infra** | Docker + Compose | 24.x / v2 | Orquestração |
| | Nginx | 1.25-alpine | Reverse proxy / SSE-WS |
| **IA Local (próxima fase)** | Ollama (Llama 3.2 3B) + MCP + ChromaDB | — | Assistente RAG para sugestões de reparo |

---

## 3. Arquitetura do Monorepo

```
projeto-tcc/
├── apps/
│   ├── backend/                          # FastAPI — Clean Architecture
│   │   ├── alembic/                      # Migrations versionadas
│   │   │   └── versions/0001_create_predictions_table.py
│   │   ├── src/
│   │   │   ├── main.py                   # create_app() + lifespan
│   │   │   ├── core/
│   │   │   │   ├── config.py             # Settings (pydantic-settings)
│   │   │   │   ├── database.py           # Async engine + session
│   │   │   │   ├── auth.py               # require_admin_token (X-Admin-Token)
│   │   │   │   ├── exceptions.py         # AppError + handlers JSON
│   │   │   │   ├── logging.py            # structlog config
│   │   │   │   ├── rate_limit.py         # slowapi limiter
│   │   │   │   └── ws_manager.py         # ConnectionManager singleton
│   │   │   ├── models/prediction.py      # ORM SQLAlchemy
│   │   │   ├── routers/
│   │   │   │   ├── health.py             # GET /health
│   │   │   │   ├── predict.py            # POST /predict/  (rate-limited)
│   │   │   │   ├── predictions.py        # GET /v1/predictions  (paginado)
│   │   │   │   ├── models.py             # GET /models, PUT /models/active
│   │   │   │   ├── stream.py             # GET /stream/sensors (SSE)
│   │   │   │   ├── simulator.py          # GET/PUT /simulator/mode
│   │   │   │   └── alerts_ws.py          # WS /ws/alerts
│   │   │   ├── schemas/                  # DTOs Pydantic v2
│   │   │   └── services/
│   │   │       ├── model_registry.py     # Hot-swap atômico (asyncio.Lock)
│   │   │       ├── model_service.py      # Adapter unificado
│   │   │       ├── mlp_adapter.py        # MLP ONNX
│   │   │       ├── onnx_tree_adapter.py  # RF v2 / XGBoost v2 ONNX
│   │   │       ├── onnx_sequence_adapter.py    # TCN / BiLSTM / PatchTST
│   │   │       ├── onnx_autoencoder_adapter.py # Conv1D AE (MSE → sigmoid)
│   │   │       ├── simulator.py          # SensorSimulator (parquet streamer)
│   │   │       ├── sensor_stream_service.py    # Broadcast SSE @1Hz
│   │   │       ├── inference_pipeline.py # Loop: stream → ML → DB → WS
│   │   │       ├── feature_buffer.py     # Deque thread-safe (janela=30)
│   │   │       ├── preprocessing.py      # Mirror do MetroPTPreprocessor
│   │   │       ├── prediction_service.py # save_prediction + list_predictions
│   │   │       ├── alert_service.py      # Push WS quando prob > 0.70
│   │   │       └── health_service.py
│   │   ├── tests/                        # pytest + SQLite em memória
│   │   ├── alembic.ini
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── frontend/                         # Next.js 16 (App Router)
│   │   ├── app/
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx                  # Dashboard principal
│   │   │   ├── sensors/[id]/             # Detalhe de sensor
│   │   │   ├── history/page.tsx
│   │   │   └── globals.css               # Tema (variáveis CSS)
│   │   ├── components/
│   │   │   ├── dashboard/
│   │   │   │   ├── FleetDashboard.tsx    # Orquestrador
│   │   │   │   ├── FleetKPIs.tsx
│   │   │   │   ├── AssetTable.tsx        # Tabela de frota
│   │   │   │   ├── AssetRadarChart.tsx   # Radar (normalização absoluta)
│   │   │   │   └── AssetEfficiencyChart.tsx
│   │   │   ├── header.tsx / sidebar.tsx
│   │   │   ├── alert-panel.tsx / alert-toast-queue.tsx
│   │   │   ├── sensor-chart.tsx / sensor-monitor.tsx
│   │   │   ├── error-boundary.tsx
│   │   │   └── ui/                       # shadcn/ui
│   │   ├── hooks/
│   │   │   ├── use-sse.ts
│   │   │   ├── use-sensor-data.ts        # Seed + SSE + poll + WS
│   │   │   ├── use-alert-websocket.ts
│   │   │   └── use-prediction-history.ts
│   │   ├── lib/api-client.ts             # fetch wrappers tipados
│   │   ├── mocks/                        # MSW handlers
│   │   ├── __tests__/                    # Vitest
│   │   ├── e2e/                          # Playwright
│   │   ├── Dockerfile / package.json / next.config.ts
│   │   └── playwright.config.ts / vitest.config.ts
│   │
│   └── ml/                               # Treinamento + MLOps
│       ├── data/raw/                     # CSV original (ignorado pelo git)
│       ├── data/processed/               # Parquet (ignorado pelo git)
│       ├── models/                       # Artefatos .onnx / .joblib / *_card.json
│       ├── src/
│       │   ├── ingest_metropt.py         # CSV → Parquet rotulado
│       │   ├── preprocessing.py          # MetroPTPreprocessor (espelhado no backend)
│       │   ├── balancing.py              # SMOTE + split estratificado
│       │   ├── datamodule_sequence.py    # Sliding windows (treino sequencial)
│       │   ├── datamodule_unsupervised.py# Apenas janelas saudáveis (AE)
│       │   ├── models/
│       │   │   ├── tcn.py / bilstm.py / patchtst.py / autoencoder.py
│       │   ├── train_random_forest.py    # SMOTE + GridSearchCV + F2-threshold
│       │   ├── train_xgboost.py          # Optuna
│       │   ├── train_mlp.py              # PyTorch Lightning → ONNX
│       │   ├── train_sequential.py       # TCN / BiLSTM / PatchTST
│       │   ├── train_autoencoder.py      # Conv1D não-supervisionado
│       │   ├── evaluate_sequential.py    # CV temporal para DL
│       │   └── promote_model.py          # MLflow → models/
│       ├── notebooks/01_eda_metropt.ipynb
│       ├── tests/                        # pytest
│       ├── Dockerfile / requirements.txt
│       └── EDA.ipynb
│
├── infra/
│   └── nginx/nginx.conf                  # SSE + WS + timeouts 3600s
│
├── docs/
│   └── model_comparison.*                # Benchmarks
│
├── docker-compose.yml                    # Orquestração unificada
├── CLAUDE.md                             # Diretrizes do projeto
├── .gitignore
└── README.md
```

---

## 4. Fluxo de Dados em Tempo Real

> Diagramas e documentação de SSE/WebSocket/Nginx abaixo atendem RNF-41.

### 4.1. Pipeline de dados — da origem ao broadcast

```
                   ┌─────────────────────────────────────┐
                   │   apps/ml/data/processed/           │
                   │      metropt3.parquet (~1.5M rows)  │
                   └────────────────┬────────────────────┘
                                    │ leitura única em memória
                                    ▼
                   ┌─────────────────────────────────────┐
                   │  SensorSimulator  (services/simulator.py) │
                   │  modes: NORMAL · DEGRADATION · FAILURE    │
                   └────────────────┬────────────────────┘
                                    │ 1 Hz
                                    ▼
                   ┌─────────────────────────────────────┐
                   │  SensorStreamService (broadcast)    │
                   │  asyncio.Queue por subscriber       │
                   └─────┬───────────────────────┬───────┘
                         │                       │
              (subscribe)│                       │(subscribe)
                         ▼                       ▼
        ┌────────────────────────┐     ┌────────────────────────────┐
        │ Router /stream/sensors │     │ InferencePipelineService   │
        │ (SSE — event-stream)   │     │  buffer (window=30)        │
        │                        │     │  → MetroPTPreprocessor     │
        │  → Frontend Next.js    │     │  → ModelService.predict()  │
        │  (useSensorData @1Hz)  │     │  → save_prediction (DB)    │
        └────────────────────────┘     │  → AlertService (prob>0.70)│
                                       └──────────────┬─────────────┘
                                                      ▼
                                          ┌────────────────────────┐
                                          │   WebSocket /ws/alerts │
                                          │   → Frontend toast     │
                                          └────────────────────────┘
```

**Warmup**: nos primeiros 15 ticks o buffer ainda não tem amostras suficientes para as features rolling (std/ma/lag/roc/min/max). Nesse intervalo o pipeline cai para inferência **stateless** (`predict(request)`) — sem bloqueio de startup.

### 4.2. Diagrama arquitetural (Mermaid)

Topologia real de rede — quem conecta em quem, por qual proxy, e onde SSE/WS/DB participam. Extraído do código (`infra/nginx/nginx.conf`, `routers/stream.py`, `routers/alerts_ws.py`, `sensor_stream_service.py`, `ws_manager.py`, `inference_pipeline.py`), não de um design idealizado.

```mermaid
flowchart TB
    B["Browser<br/>Next.js Client Components<br/>useSSE · useAlertWebSocket"]
    N["Nginx :80<br/>infra/nginx/nginx.conf"]

    subgraph API["FastAPI — serviço api:8000 (root_path=/api)"]
        SR["GET /stream/sensors<br/>routers/stream.py (SSE)"]
        WR["WS /ws/alerts<br/>routers/alerts_ws.py"]
        SS["SensorStreamService<br/>broadcast loop @1Hz"]
        CM["ConnectionManager<br/>core/ws_manager.py"]
        IP["InferencePipelineService"]
        AS["AlertService"]
    end

    SIM["SensorSimulator<br/>replay metropt3.parquet"]
    DB[("PostgreSQL<br/>tabela predictions")]

    SIM -->|"1 Hz"| SS
    SS -->|subscribe| SR
    SS -->|subscribe| IP
    IP -->|"predict + save_prediction"| DB
    IP --> AS
    AS -->|"probability > 0.70"| CM
    CM --> WR

    B ==>|"① GET /api/stream/sensors<br/>Accept: text/event-stream"| N
    N ==>|"proxy_pass /stream/ (nginx remove /api)"| SR
    SR ==>|"event: sensor_reading (1x/s)"| N
    N ==>|"text/event-stream, sem buffering"| B

    B -.->|"② WS /ws/alerts<br/>Upgrade: websocket"| N
    N -.->|"proxy_pass, headers Upgrade/Connection"| WR
    WR -.->|"{type:alert} / {type:ping} 30s"| N
    N -.->|"frame WS"| B
```

**Leitura do diagrama:**

1. **Conexão do cliente** — o browser abre DUAS conexões persistentes e independentes: uma SSE (`GET /api/stream/sensors`, seta cheia ①) e uma WebSocket (`WS /ws/alerts`, seta pontilhada ②). Nenhuma depende da outra estar aberta.
2. **Onde a comunicação passa** — ambas atravessam o Nginx (`nginx.conf`), que decide o proxy por `location` (path-based), nunca indo direto do browser à API.
3. **Quem usa WebSocket** — só `/ws/alerts` (alertas). Full-duplex: servidor manda `alert`/`ping`, cliente manda `ack`/`pong`.
4. **Quem usa SSE** — só `/stream/sensors` (telemetria contínua). Unidirecional servidor→cliente.
5. **Direção dos eventos** — SSE é sempre servidor→browser; WS é bidirecional, mas o `alert` sempre nasce no servidor (nunca o cliente inicia um alerta).
6. **Onde ocorre o broadcast** — dois broadcasters distintos, sem relação um com o outro: `SensorStreamService._broadcast_loop` (1 fila `asyncio.Queue` por SSE subscriber) e `ConnectionManager.broadcast`/`broadcast_alert` (itera o `set` de WebSockets ativos). Cada um serve exatamente um dos dois canais.
7. **Onde o banco participa** — **só no caminho de inferência** (`InferencePipelineService.save_prediction`), nunca diretamente na entrega de SSE ou WS. Nem `GET /stream/sensors` nem `WS /ws/alerts` abrem sessão de banco por cliente conectado.
8. **Onde o Nginx atua** — único ponto de entrada externo (porta 80); decide por prefixo de path qual serviço upstream recebe cada request (`api` ou `frontend`), reescreve headers, e é responsável por manter a conexão aberta (buffering desligado, timeouts longos) — detalhado no §4.5.

### 4.3. SSE — `GET /api/stream/sensors`

#### Como funciona

| Aspecto | Valor real (código) |
|---|---|
| Endpoint | `GET /stream/sensors` (backend) → `GET /api/stream/sensors` (via Nginx) |
| Router | [`stream.py`](apps/backend/src/routers/stream.py) |
| Método de conexão | `EventSource` nativo do browser, via [`useSSE`](apps/frontend/hooks/use-sse.ts) |
| Formato do evento | `event: sensor_reading\ndata: {...12 sensores + timestamp + inference_latency_ms}\nid: <epoch_ms>\nretry: 3000\n\n` |
| Frequência | 1 evento/segundo — `BROADCAST_INTERVAL = 1.0` em [`sensor_stream_service.py`](apps/backend/src/services/sensor_stream_service.py) |
| Heartbeat/broadcast | Broadcast único e compartilhado (não é heartbeat de protocolo): `SensorStreamService._broadcast_loop` gera 1 leitura/s e faz `put_nowait` na fila de **cada** subscriber. Fila cheia (`_QUEUE_MAX_SIZE=10`) → cliente lento é evicted, não trava os demais. |
| Reconexão | Dupla camada: (a) `EventSource` nativo tenta reconectar sozinho (`retry: 3000` no evento); (b) [`useSSE.ts`](apps/frontend/hooks/use-sse.ts) trata `onerror` com backoff exponencial próprio (1s → 30s, jitter ±20%), criando uma **nova** instância de `EventSource` — as duas reconexões não competem porque `useSSE` fecha o `EventSource` antigo antes de recriar. |
| Timeout de leitura no servidor | `asyncio.wait_for(queue.get(), timeout=2.0)` — se não houver leitura em 2s, apenas verifica `is_disconnected()` e tenta de novo (não é um timeout de conexão). |
| Encerramento | Cliente fecha a aba/`EventSource.close()` → Starlette detecta e chama `aclose()` no generator → bloco `finally` roda `service.unsubscribe(queue)` — sem vazamento de fila. |

**Nota de arquitetura**: o broadcast é compartilhado com **um clock só** — um cliente que acabou de conectar pode esperar até ~1s pelo próximo tick antes do primeiro dado (não há "replay" do último valor ao conectar). Isso é intencional (cadência de 1Hz do produto), documentado também em [`locust_streaming.py`](locust_streaming.py) por afetar como medir latência de SSE (RNF-37, §14.5).

#### Diagrama de sequência — SSE

```mermaid
sequenceDiagram
    participant FE as Frontend (useSSE)
    participant NG as Nginx
    participant API as FastAPI (/stream/sensors)
    participant SS as SensorStreamService

    FE->>NG: GET /api/stream/sensors (Accept: text/event-stream)
    NG->>API: proxy_pass /stream/sensors (prefixo /api removido)
    API->>SS: subscribe() -> asyncio.Queue
    API-->>NG: 200 OK, Content-Type: text/event-stream
    NG-->>FE: 200 OK (headers passthrough, sem buffer)
    loop a cada 1s
        SS->>SS: SensorSimulator.generate_reading()
        SS->>API: put_nowait(reading) em cada fila
        API-->>NG: event: sensor_reading\ndata: {...}
        NG-->>FE: encaminhado (X-Accel-Buffering: no)
    end
    FE-->>API: aba fechada / EventSource.close()
    API->>SS: unsubscribe() [bloco finally]
    Note over FE,API: em erro de rede: EventSource nativo tenta<br/>reconectar (retry:3000) E useSSE.ts aplica<br/>backoff próprio (1s-30s, jitter) recriando o EventSource
```

#### Diagnóstico

```bash
# 1. API está saudável? (verifica também o Postgres)
curl -s http://localhost/api/health
# {"status":"ok","version":"1.0.0","database":{"connected":true,"latency_ms":1.3,"error":null}}

# 2. Containers de pé?
docker compose ps

# 3. Endpoint SSE acessível e emitindo? (Ctrl+C após alguns eventos)
curl -N http://localhost/api/stream/sensors

# 4. A conexão passa pelo Nginx corretamente? Compare direto na API...
docker compose exec nginx sh -c "apk add --no-cache curl >/dev/null 2>&1; curl -N http://api:8000/stream/sensors" | head -c 300
# ...com o caminho completo via Nginx:
curl -N http://localhost/api/stream/sensors | head -c 300
# Os dois devem ter o mesmo formato de evento (event:/data:/id:/retry:).

# 5. Eventos realmente chegando (contagem em N segundos)?
timeout 5 curl -N http://localhost/api/stream/sensors | grep -c "^event: sensor_reading"

# 6. Logs do broadcast/subscribers
docker compose logs api --tail=50 | grep -i sse
```

#### Problemas comuns

| Sintoma | Causas possíveis (reais, deste projeto) | Diagnóstico | Solução |
|---|---|---|---|
| SSE não conecta (erro de rede no browser) | `api` não está de pé; Nginx não está rodando; porta 80 ocupada | `docker compose ps`; `docker compose logs nginx` | `docker compose up -d nginx api` |
| Conexão fecha imediatamente | Simulador sem parquet gerado → `FileNotFoundError` no startup do pipeline (não afeta a rota SSE em si, mas o broadcast pode não ter dados) | `docker compose logs api \| grep -i simulator` | Gerar o parquet — ver §16 "`FileNotFoundError: Simulator parquet not found`" |
| Eventos não chegam (conexão abre, mas fica muda) | Buffer do subscriber saturado e cliente evicted silenciosamente (`_QUEUE_MAX_SIZE=10`); ou 0 subscribers ativos parando o `_broadcast_loop` (ele só reinicia no próximo `subscribe()`) | `docker compose logs api \| grep sse_slow_consumers_evicted` | Reconectar (novo `EventSource`); se persistir, checar CPU do container (`docker stats api`) |
| Funciona direto na API mas não via Nginx | `location /api/stream/` sem `X-Accel-Buffering: no` ou `chunked_transfer_encoding off` (nesta config eles existem — se removidos, o Nginx passa a bufferizar) | Comparar diagnóstico #4 acima | Restaurar os headers/directivas em `infra/nginx/nginx.conf` (ver §4.5) |
| Timeout / conexão cai após ~60s | Timeout padrão de proxy (60s) em vez do configurado | `curl -N` e cronometrar quando cai | Já mitigado nesta config: `proxy_read_timeout 3600s` no bloco `/api/stream/` — se voltar a acontecer, confirme que o `nginx.conf` ativo no container é o do repo (`docker compose exec nginx cat /etc/nginx/conf.d/*.conf` ou o `nginx.conf` principal) |
| Múltiplas conexões (várias abas) | Comportamento esperado — cada SSE subscriber tem sua própria fila; não há limite de conexões configurado no código | `docker compose logs api \| grep sse_subscribed` (contagem em `total=`) | N/A — validado até 100+ conexões simultâneas em `locust_streaming.py` (RNF-37, §14.5) |
| Browser não recebe eventos, mas `curl -N` funciona | Extensão de browser/DevTools bloqueando `EventSource`; ou `NEXT_PUBLIC_API_URL` mal configurada no frontend (ver §7) | Console do browser (erros de rede/CORS); `curl -N` como baseline de comparação | Corrigir `apps/frontend/.env.local` (`NEXT_PUBLIC_API_URL=/api`) e recarregar |
| Problemas de proxy buffering | `proxy_buffering` (default Nginx = on) não é desativado explicitamente no bloco `/api/stream/` — ele confia só em `chunked_transfer_encoding off` + `X-Accel-Buffering: no` | Ver diagnóstico #4; se o delay entre eventos parecer maior que 1s de forma consistente | **Observação, não corrigido nesta task** (fora do escopo alterar Nginx): adicionar `proxy_buffering off;` explicitamente no bloco seria mais robusto — ver "Known Issue" no final do documento |
| Problemas de headers | Cliente sem `Accept: text/event-stream` — o backend não valida isso, mas alguns proxies/CDNs intermediários poderiam | Inspecionar request headers no DevTools → Network | Garantir que `EventSource` (que já envia o header certo automaticamente) seja o método de conexão, não `fetch` manual sem esse header |
| CORS | Não aplicável neste fluxo — Nginx serve frontend e API no mesmo `http://localhost` (same-origin); `ALLOWED_ORIGINS` no backend é usado por `CORSMiddleware` só para chamadas REST fora desse domínio | — | — |
| Container/API indisponível | `api` crashou (ex.: modelo ativo com artefato ausente no startup) | `docker compose ps` (status `Exited`); `docker compose logs api` | Ver §16 "`POST /predict/` retorna 503" e "`ACTIVE_MODEL` ignorado" |

### 4.4. WebSocket — `WS /ws/alerts`

#### Como funciona

| Aspecto | Valor real (código) |
|---|---|
| Endpoint | `WS /ws/alerts` (backend) → `WS /ws/alerts` via Nginx (**sem** prefixo `/api` — bloco `location /ws/` dedicado, diferente do REST/SSE) |
| Router | [`alerts_ws.py`](apps/backend/src/routers/alerts_ws.py) |
| Gerenciador | [`ConnectionManager`](apps/backend/src/core/ws_manager.py) — singleton `manager`, registro em `set[WebSocket]` |
| Autenticação | **Nenhuma** — endpoint público, sem `Depends`/token. Não confundir com `X-Admin-Token` (usado só em `/models`, REST) |
| Conexão | Frontend: [`useAlertWebSocket`](apps/frontend/hooks/use-alert-websocket.ts) monta `ws://`/`wss://` a partir de `window.location` |
| Mensagens servidor→cliente | `{"type": "alert", "message_id", "probability", "predicted_class", "timestamp", "inference_latency_ms"}` (RF-14, quando `probability > 0.70`) e `{"type": "ping"}` (heartbeat, a cada 30s) |
| Mensagens cliente→servidor | `{"type": "ack", "message_id"}` (confirmação de entrega) e `{"type": "pong"}` (resposta ao heartbeat) |
| Desconexão | Qualquer fechamento (limpo ou abrupto) cai no bloco `finally` de `alerts_ws.py` → `ws_manager.disconnect()` — sem vazamento de referência |
| Reconexão | [`useAlertWebSocket.ts`](apps/frontend/hooks/use-alert-websocket.ts) — mesmo padrão do SSE: backoff exponencial 1s→30s, jitter ±20% |

**Nota de arquitetura**: o heartbeat (`_heartbeat_loop`, 30s) só roda enquanto existe ≥1 cliente conectado (`_ensure_heartbeat`/`_cancel_heartbeat`); ele detecta conexões mortas de forma **implícita** — não há `pong` timeout ativo no servidor, uma conexão só é removida quando um `send` falha.

#### Diagrama de sequência — WebSocket

```mermaid
sequenceDiagram
    participant FE as Frontend (useAlertWebSocket)
    participant NG as Nginx
    participant API as FastAPI (/ws/alerts)
    participant CM as ConnectionManager

    FE->>NG: WS /ws/alerts (Upgrade: websocket)
    NG->>API: proxy_pass (headers Upgrade/Connection preservados)
    API->>CM: connect() -> accept() + registro no set
    CM->>CM: _ensure_heartbeat() inicia Task

    par heartbeat a cada 30s
        CM->>API: {"type":"ping"}
        API->>NG: frame ping
        NG->>FE: frame ping
        FE->>NG: {"type":"pong"}
        NG->>API: frame pong
    and alerta quando probability > 0.70
        Note over CM: AlertService.process_prediction()
        CM->>API: {"type":"alert", message_id, probability, ...}
        API->>NG: frame alert
        NG->>FE: frame alert
        FE->>NG: {"type":"ack", message_id}
        NG->>API: frame ack
        API->>CM: send_personal({"type":"ack", ...})
    end

    FE-->>API: conexão cai (aba fechada / rede)
    API->>CM: disconnect() [bloco finally]
    Note over FE: useAlertWebSocket.ts reconecta com<br/>backoff exponencial (1s-30s, jitter ±20%)
```

#### Diagnóstico

```bash
# 1. API saudável?
curl -s http://localhost/api/health

# 2. Handshake WS funciona via Nginx? (código 101 Switching Protocols)
curl -i -N \
  -H "Connection: Upgrade" -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
  http://localhost/ws/alerts

# 3. Direto na API (dentro da rede Docker), para isolar Nginx:
docker compose exec nginx sh -c "apk add --no-cache curl >/dev/null 2>&1; curl -i -N \
  -H 'Connection: Upgrade' -H 'Upgrade: websocket' \
  -H 'Sec-WebSocket-Version: 13' -H 'Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==' \
  http://api:8000/ws/alerts"

# 4. Quantos clientes conectados agora (logs)
docker compose logs api --tail=100 | grep ws_connected

# 5. Alertas sendo disparados?
docker compose logs api --tail=200 | grep alert_triggered
```

#### Problemas comuns

| Sintoma | Causas possíveis (reais) | Diagnóstico | Solução |
|---|---|---|---|
| Handshake falhando (não chega a 101) | Nginx sem os headers `Upgrade`/`Connection` no bloco `/ws/`; ou request sem os headers WS obrigatórios | Diagnóstico #2 vs #3 acima — se #3 (direto na API) funciona mas #2 (via Nginx) não, o problema é o proxy | Confirmar `proxy_http_version 1.1`, `proxy_set_header Upgrade $http_upgrade`, `proxy_set_header Connection $connection_upgrade` no bloco `/ws/` de `nginx.conf` (já presentes nesta config — se removidos, é a causa) |
| Conexão encerrando sem motivo aparente | Nenhum `pong` no timeout esperado NÃO fecha a conexão neste código (não há enforcement ativo) — se está caindo, é mais provável rede/proxy | `docker compose logs api \| grep ws_disconnected` (ver o `client=` e correlacionar com hora) | Verificar se é o browser fechando a aba, não o servidor |
| Erro no proxy | `location /ws/` ausente/mal configurado, ou `proxy_pass http://api;` alterado para incluir path (mudaria a URI encaminhada) | Comparar a URI recebida pela API: `docker compose logs api \| grep ws_connected` mostra o `client`, não a URI — usar diagnóstico #3 para confirmar a URI final | Restaurar `nginx.conf` original do repo |
| Erro de autenticação | **Não existe autenticação neste endpoint** — se um cliente está recebendo 401/403, o erro não vem de `/ws/alerts` | Confirmar a URL exata sendo chamada (não confundir com `/api/models`, que exige `X-Admin-Token`) | Corrigir a URL no cliente |
| Mensagens não chegando | `probability <= 0.70` (não é bug — `broadcast_alert` silenciosamente ignora, é a regra RF-14); ou 0 clientes conectados no momento do broadcast | `docker compose logs api \| grep alert_skipped` vs `alert_triggered` | Esperado — só dispara acima do limiar. Para forçar, use `PUT /api/simulator/mode {"mode":"FAILURE"}` (§15) |
| Problemas específicos do Nginx | `proxy_read_timeout`/`proxy_send_timeout` menores que o necessário fechariam conexões idle | Medir tempo até a queda | Já mitigado: `3600s` para ambos no bloco `/ws/` |

### 4.5. Papel do Nginx na arquitetura real-time

Config real: [`infra/nginx/nginx.conf`](infra/nginx/nginx.conf). Único ponto de entrada externo (porta 80/443) — `frontend` e `api` não são expostos diretamente (só `expose:`, sem `ports:` no `docker-compose.yml`).

| Bloco (`location`) | Destino (`proxy_pass`) | Papel | Headers/diretivas chave |
|---|---|---|---|
| `/ws/` | `http://api` (URI original preservada — sem prefixo removido) | Proxy WebSocket | `proxy_http_version 1.1`; `Upgrade: $http_upgrade`; `Connection: $connection_upgrade` (mapeado: `upgrade` → mantém upgrade, vazio → `close`); timeouts **3600s** |
| `/api/stream/` | `http://api/stream/` (prefixo `/api/stream/` removido, vira `/stream/...`) | Proxy SSE | `Connection: ''`; `Cache-Control: no-cache`; `X-Accel-Buffering: no`; `chunked_transfer_encoding off`; timeouts **3600s** |
| `/docs`, `/redoc`, `/openapi.json` | `http://api/docs` etc. | Proxy da documentação OpenAPI | `proxy_http_version 1.1` |
| `/api/` | `http://api/` (prefixo `/api/` removido) | Proxy REST genérico | timeouts **10s** connect / **3600s** read/send (herda o timeout longo mesmo sendo REST — não diferenciado por rota dentro deste bloco) |
| `/` (catch-all) | `http://frontend` | Proxy do Next.js (inclui HMR/webpack em dev) | `Upgrade`/`Connection` (para o WS do Turbopack/webpack HMR); timeout **180s** (Turbopack compila rotas on-demand) |

**Por que `/ws/` não usa o prefixo `/api/`**: o backend monta a rota WebSocket literal em `/ws/alerts` (sem prefixo — `root_path="/api"` do FastAPI só afeta geração de URLs no OpenAPI, não o roteamento real). Por isso o Nginx precisa de um bloco `location /ws/` **separado** do `/api/`, encaminhando a URI original sem remover nenhum prefixo. Já o router SSE está registrado em `/stream/sensors` e é exposto ao público como `/api/stream/sensors` — aqui o Nginx **remove** o prefixo `/api/stream/` ao repassar (por isso `proxy_pass` termina em `/stream/`, com barra).

**Buffering desligado para streaming**: `X-Accel-Buffering: no` + `chunked_transfer_encoding off` no bloco SSE evitam que o Nginx acumule a resposta antes de entregar ao cliente — sem isso, os eventos apareceriam em lote em vez de 1x/segundo. `proxy_buffering` (a diretiva "oficial" para isso) não é setada explicitamente neste bloco — ver observação no §4.3 (não corrigido nesta task, fora do escopo alterar Nginx).

**Upgrade de conexão**: só os blocos `/ws/` e `/` (frontend, por causa do HMR) enviam os headers `Upgrade`/`Connection: $connection_upgrade`. O bloco `/api/stream/` **não** precisa disso — SSE é HTTP puro (não faz upgrade de protocolo), só precisa de buffering desligado e timeout longo.

### 4.6. MCP Server — arquitetura (RF-19 / RNF-43)

**O que é**: um servidor [MCP](https://modelcontextprotocol.io) (Model Context Protocol) **standalone**, em `apps/mcp-server/`, processo e container Docker próprios — não roda dentro do processo do FastAPI, não compartilha memória/estado com ele. Hoje expõe exatamente **uma** ferramenta, `search_maintenance_manual`, que faz **busca semântica real** (RF-21) sobre os manuais indexados pela pipeline da §4.7 (RF-20).

**Por que é separado**: o propósito de longo prazo (CLAUDE.md §4) é o Ollama (LLM local) consultar manuais técnicos via MCP para sugerir reparos — isso é um domínio de responsabilidade diferente do FastAPI (que faz predição/streaming), roda em processo próprio para poder escalar/reiniciar/trocar de implementação (real RAG) independentemente do backend, e para não acoplar as dependências de busca/embeddings ao container da API.

```mermaid
flowchart LR
    C["Browser / Client"] --> N["Nginx"]
    N --> API["FastAPI (api:8000)"]
    N -.->|"ainda não roteado<br/>(sem endpoint MCP público)"| MCP

    API -->|"MCP_SERVER_URL<br/>(consumida desde RF-22, ver §4.8)"| MCP["MCP Server<br/>apps/mcp-server (container próprio)"]
    MCP --> TOOL["search_maintenance_manual<br/>(busca semântica real — RF-21)"]
    TOOL --> SVC["SemanticSearchService"]
    SVC --> CHROMA[("ChromaDB<br/>maintenance_manuals — RF-20")]

    style MCP stroke-dasharray: 3 3
```

> A seta tracejada `API -.-> MCP` existe só para mostrar a URL **configurada** (`MCP_SERVER_URL`) — o backend **não faz** essa chamada ainda nesta task (ver Known Issues/estado atual abaixo). Não é roteado pelo Nginx: é alcançado só pela rede interna Docker (`http://mcp-server:8100`), como `db`/`mlflow`.

| Aspecto | Valor real |
|---|---|
| Localização | `apps/mcp-server/server.py` |
| SDK | [`mcp`](https://pypi.org/project/mcp/) (SDK oficial do Model Context Protocol) — `mcp==2.2.0` |
| Classe do servidor | `mcp.server.mcpserver.MCPServer` (renomeada de `FastMCP` na v1.x do SDK — `mcp.server.fastmcp` não existe mais nesta versão) |
| Transporte | `streamable-http` — o transporte HTTP direto-por-URL suportado por este SDK (os outros dois, `stdio` e `sse`, não servem para um serviço standalone acessível pela rede) |
| Endpoint | `POST http://mcp-server:8100/mcp` (JSON-RPC 2.0, protocolo MCP) |
| Host/porta | `0.0.0.0` / `MCP_SERVER_PORT` (default `8100`) |
| Container | `mcp-server` (`docker-compose.yml`) — `build: ./apps/mcp-server`, rede `app-network`, `expose` (só interno, sem `ports:`) |
| `MCP_SERVER_URL` | `http://mcp-server:8100` — ver `.env.example`. Declarada em `Settings.mcp_server_url` (`apps/backend/src/core/config.py`). Consumida desde a RF-22 por `MCPSearchClient` (`apps/backend/src/services/mcp_client.py`) — ver §4.8 |

#### Ferramenta

| Tool | Input | Estado | Função |
|---|---|---|---|
| `search_maintenance_manual` | `query: string` (obrigatório) | **Busca semântica real (RF-21)** | Recupera, no máximo, os 5 trechos mais relevantes dos manuais indexados |

**Fluxo real** (`server.py` → `semantic_search.py` → ChromaDB, ver §4.7 para a indexação):

```mermaid
flowchart TD
    Q["query: string"] --> ENC["SentenceTransformer.encode<br/>(mesmo EMBEDDING_MODEL da indexação)"]
    ENC --> CAND["ChromaDB.query<br/>(candidatos por proximidade, ate 20)"]
    CAND --> SCORE["cosine similarity<br/>(calculada aqui, nao a distancia do Chroma)"]
    SCORE --> FILTER{"score > 0.6 ?"}
    FILTER -->|"nao"| DROP["descartado"]
    FILTER -->|"sim"| TOP5["top 5 por score"]
    TOP5 --> RESP["query + results[]<br/>(text, score, metadata)"]
```

Resposta real (capturada via chamada MCP de verdade — ver "Validação ponta-a-ponta" abaixo):

```json
{
  "query": "vazamento na bomba centrifuga",
  "results": [
    {
      "text": "Manual de manutencao da bomba centrifuga modelo BC-200. Verificar o alinhamento do eixo e o estado dos rolamentos a cada 1000 horas de operacao continua.",
      "score": 0.7108,
      "metadata": {
        "source": "manual-bomba-centrifuga.pdf",
        "file_name": "manual-bomba-centrifuga.pdf",
        "file_hash": "35af10763bc9...",
        "page": 1,
        "chunk_index": 0,
        "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
      }
    }
  ]
}
```

| Regra | Valor |
|---|---|
| Máximo de resultados | 5 |
| Threshold de score | `score > 0.6` (estrito — `score == 0.6` é descartado) |
| Métrica de score | Cosine similarity, calculada em `semantic_search.py` a partir dos embeddings brutos (ver "Por que cosine similarity" abaixo) — **não** a distância que o ChromaDB devolve diretamente |
| Sem resultado relevante | Resposta válida `{"query": ..., "results": []}` — nunca lança exceção |
| Modelo de embeddings | O mesmo `EMBEDDING_MODEL` da indexação (§4.7), carregado **uma vez por processo** (singleton lazy em `server._get_service`) |
| Collection consultada | `maintenance_manuals` (a mesma da indexação — nenhum ChromaDB novo) |

#### Por que cosine similarity, não a distância bruta do ChromaDB

A collection é criada sem `hnsw:space` explícito (§4.7) — o ChromaDB usa o default `l2` (distância L2 ao quadrado sobre embeddings não normalizados). Medido contra o corpus real desta task, essa distância fica em ~15–50 e uma conversão ingênua (`1/(1+distancia)`, fórmula padrão para espaços L2) nunca passava de `~0.07` — o threshold `> 0.6` da RF-21 rejeitaria **toda** consulta, por mais relevante que fosse. Corrigido calculando a cosine similarity diretamente a partir dos embeddings brutos (query + candidatos, este último obtido via `collection.query(..., include=["embeddings"])`) — invariante à norma dos vetores, portanto insensível a essa característica do modelo. O índice ANN do ChromaDB continua usado só para obter os candidatos de forma eficiente; o score que decide o filtro é sempre a cosine similarity. Ver docstring de [`semantic_search.py`](apps/mcp-server/semantic_search.py) para a investigação completa.

#### Como iniciar

```bash
docker compose up -d --build mcp-server
docker compose logs mcp-server --tail=20
```

#### Como verificar se está funcionando

```bash
# 1. Container de pé e processo vivo?
docker compose ps mcp-server

# 2. Handshake MCP real, pela rede interna Docker (a partir de qualquer
#    outro container na mesma rede — aqui usando o nginx como exemplo,
#    já que ele já tem curl instalável via apk):
docker compose exec nginx sh -c "apk add --no-cache curl >/dev/null 2>&1; \
  curl -s -X POST http://mcp-server:8100/mcp \
    -H 'Content-Type: application/json' \
    -H 'Accept: application/json, text/event-stream' \
    -d '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{},\"clientInfo\":{\"name\":\"check\",\"version\":\"0\"}}}'"
# Resposta esperada: evento SSE com "serverInfo":{"name":"predictiq-mcp",...}
```

O header de resposta `mcp-session-id` da chamada `initialize` deve ser reenviado (`-H "mcp-session-id: <valor>"`) nas chamadas seguintes (`tools/list`, `tools/call`) — é assim que a sessão MCP é mantida sobre HTTP stateless.

#### Validação ponta-a-ponta (RF-21) — mesma sessão acima, `tools/call` real

```bash
# reenviar o mesmo mcp-session-id do initialize acima
docker compose exec nginx sh -c "curl -s -X POST http://mcp-server:8100/mcp \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -H 'mcp-session-id: <valor capturado no initialize>' \
  -d '{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"search_maintenance_manual\",\"arguments\":{\"query\":\"vazamento na bomba centrifuga\"}}}'"
```

Rodado de verdade contra o ChromaDB real desta task (3 manuais indexados via §4.7): a query relevante acima devolveu 3 trechos do `manual-bomba-centrifuga.pdf` com score `0.71`/`0.70`/`0.63` (ver JSON de exemplo acima); uma query deliberadamente sem relação ("receita de bolo de chocolate") devolveu `{"results": []}`, sem erro.

#### Testes

```bash
docker compose exec mcp-server python -m pytest tests/ -v
# ou localmente:
cd apps/mcp-server && python -m venv .venv && .venv\Scripts\Activate.ps1
pip install -r requirements.txt && pytest tests/ -v
```

`tests/test_semantic_search.py` cobre `SemanticSearchService`/`cosine_similarity` isoladamente (threshold, ordenação, limite de 5, metadados); `tests/test_mcp_server.py` cobre a integração `search_maintenance_manual -> SemanticSearchService` (com um `SemanticSearchService` real sobre ChromaDB temporário — não um mock que só devolve um dict fixo).

#### Benchmark de busca semântica (RNF-45)

```bash
docker compose exec mcp-server python benchmark_semantic_search.py
```

Mede a operação completa (`search_maintenance_manual` → `SemanticSearchService` → embedding da query → ChromaDB → resposta), contra o ChromaDB real (não mock), com warm-up excluído da medição. Resultado real desta task (3 manuais / 9 chunks indexados, 24 queries variadas, 3 de warm-up):

```text
Semantic Search Benchmark

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
Documents: 9 chunks
Queries: 24
Warm-up: 3 iterations (not measured)
Avg results/query: 0.75

p50: 16.56 ms
p95: 18.08 ms
p99: 18.08 ms

RNF-45 (p95 < 500.0 ms): PASS
```

Evidência completa (latência de cada uma das 24 queries): [`semantic_search_benchmark.json`](apps/mcp-server/semantic_search_benchmark.json).

> `avg_results_per_query = 0.75` é esperado, não um bug: o corpus de teste tem só 3 manuais sintéticos e o threshold `> 0.6` é estrito (ver "Por que cosine similarity" acima) — várias das 24 queries do benchmark são só tematicamente relacionadas, não paráfrases próximas de um chunk específico, então ficam abaixo do threshold. O benchmark mede **latência**, não taxa de acerto da busca.

> **Nota de nomenclatura**: CLAUDE.md/AGENTS.md descrevem esta ferramenta como `search_manuals`; a task RF-19 que criou o stub original especificou explicitamente o nome `search_maintenance_manual`, mantido em RF-21. Sinalizado aqui para quem for alinhar com a documentação conceitual mais antiga.

### 4.7. Ingestão de manuais — pipeline RAG offline (RF-20 / RNF-44)

**O que é**: `apps/mcp-server/index_manuals.py` — um script batch, independente do transporte MCP, que lê PDFs, extrai texto, gera chunks com overlap, gera embeddings localmente e persiste tudo em um ChromaDB local. Prepara a base de conhecimento que a ferramenta `search_maintenance_manual` consome de verdade desde a RF-21 (§4.6).

```mermaid
flowchart TD
    PDFS["PDF manuals<br/>(data/manuals/*.pdf)"] --> READER["pypdf<br/>(extrai texto por página)"]
    READER --> CHUNK["text chunks<br/>(determinístico, com overlap)"]
    CHUNK --> EMBED["SentenceTransformer<br/>(EMBEDDING_MODEL)"]
    EMBED --> VEC["embeddings"]
    VEC --> CHROMA[("ChromaDB persistente<br/>collection: maintenance_manuals")]
```

```mermaid
flowchart TD
    START["PDF"] --> HASH["SHA-256 do conteúdo"]
    HASH --> CHECK{"já indexado<br/>com este hash?"}
    CHECK -->|"sim"| SKIP["skip<br/>(nenhum chunk criado)"]
    CHECK -->|"não, hash mudou"| PURGE["remove chunks antigos<br/>deste file_name"]
    CHECK -->|"não, nunca visto"| EXTRACT
    PURGE --> EXTRACT["extract → chunk → embed"]
    EXTRACT --> UPSERT["upsert no ChromaDB<br/>+ novo file_hash nos metadados"]
```

| Aspecto | Valor real |
|---|---|
| Script | `apps/mcp-server/index_manuals.py` — executável direto: `python index_manuals.py` |
| Dependências novas | `pypdf==5.1.0`, `sentence-transformers==3.3.1`, `chromadb==0.5.23` (`apps/mcp-server/requirements.txt`) |
| Modelo de embeddings | `EMBEDDING_MODEL` (env) — default `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilíngue, cobre PT-BR dos manuais/queries; 100% local, sem API externa). Carregado **uma vez por execução**, nunca por chunk |
| Diretório dos PDFs | `MANUALS_DIR` (env) — default `data/manuals` (relativo a `apps/mcp-server/`, resolvido pelo próprio script, não pelo cwd) |
| ChromaDB | `CHROMA_DB_PATH` (env) — default `data/chroma` (`chromadb.PersistentClient`, nunca efêmero) |
| Collection | `maintenance_manuals` |
| Chunking | `CHUNK_SIZE` / `CHUNK_OVERLAP` (env) — default `1000`/`150` caracteres, por página (não cruza página, mantém `page` correto nos metadados). Determinístico: mesmo PDF + mesma config = mesmos chunks |
| IDs dos chunks | `{file_hash}_p{page}_c{chunk_index}` — determinístico, sem UUID aleatório |
| Metadados por chunk | `source`, `file_name`, `file_hash`, `page`, `chunk_index`, `embedding_model` |
| Volume Docker | Nenhuma linha nova em `docker-compose.yml` — `data/manuals` e `data/chroma` vivem dentro de `apps/mcp-server/`, já coberto pelo bind mount `./apps/mcp-server:/app` do serviço `mcp-server` (mesmo padrão do `apps/ml/data` do serviço `ml`) |

#### Onde colocar os PDFs

```text
apps/mcp-server/data/manuals/
├── manual-bomba.pdf
├── manual-compressor.pdf
└── manual-motor.pdf
```

#### Como indexar

```bash
docker compose exec mcp-server python index_manuals.py
```

(Localmente, sem Docker: `cd apps/mcp-server && python index_manuals.py` — mesmo comportamento, resolve `data/manuals`/`data/chroma` relativo ao próprio script.)

#### Incremental (RF-20)

Não existe arquivo de estado separado — a fonte de verdade é o próprio ChromaDB (`file_hash` nos metadados de cada chunk). SHA-256 é calculado sobre o **conteúdo** do arquivo (nunca nome/data/tamanho): se o hash não mudou desde a última execução, o PDF é pulado (`skip`) — rodar duas vezes seguidas sem alterar nada não cria chunks duplicados (idempotente).

#### Reindexação

Alterar o conteúdo do PDF muda o hash. A pipeline detecta a divergência, remove **todos** os chunks antigos daquele `file_name` e só então insere os novos — nunca ficam chunks de hash antigo e hash novo misturados. Se a extração/chunking/embedding falhar no meio do processo, nada é removido/alterado no ChromaDB (a versão anterior, se existia, permanece válida).

#### Testes

```bash
docker compose exec mcp-server pytest tests/test_index_manuals.py -v
```

Todos os testes mockam o `SentenceTransformer` (nenhum download de modelo real na suíte) e usam diretório temporário para o ChromaDB (nunca o `data/chroma` real).

### 4.8. Sugestão automática de manutenção — RAG via Ollama (RF-22 / RNF-46)

**O que é**: `POST /v1/maintenance/suggest` (backend FastAPI) — quando a probabilidade de falha do modelo preditivo excede um limiar, orquestra MCP (RF-21) + Llama 3.2 3B local (Ollama) para gerar um plano de manutenção em Markdown fundamentado exclusivamente nos manuais recuperados. Implementado em `MaintenanceSuggestionService` (`apps/backend/src/services/maintenance_suggestion_service.py`), nunca na rota.

```mermaid
flowchart TD
    REQ["POST /v1/maintenance/suggest<br/>failure_probability + equipamento/sintoma"] --> THR{"probability > 0.7 ?"}
    THR -->|"não"| SKIP["triggered=false<br/>MCP e Ollama NÃO são chamados"]
    THR -->|"sim"| MCP["MCPSearchClient<br/>-> search_maintenance_manual (RF-21)"]
    MCP --> CTX["contexto: trechos + file_name/page/score"]
    CTX --> PROMPT["System Prompt (fixo) + prompt do usuário<br/>(contexto delimitado, RF-22 §13)"]
    PROMPT --> OLLAMA["Ollama /api/chat<br/>Llama 3.2 3B, local"]
    OLLAMA --> VALID["validação: não vazio, não JSON, não HTML, tem cabeçalho"]
    VALID --> RESP["MaintenanceSuggestionResponse<br/>(markdown + references)"]
```

| Aspecto | Valor real |
|---|---|
| Endpoint | `POST /v1/maintenance/suggest` (via Nginx: `POST /api/v1/maintenance/suggest`) |
| Threshold (RF-22) | `failure_probability > 0.7` — **estrito**, `MAINTENANCE_SUGGESTION_THRESHOLD` em `maintenance_suggestion_service.py`. `0.70` exato **não** dispara |
| Tool MCP usada | `search_maintenance_manual` (nome real confirmado no código — CLAUDE.md/AGENTS.md mencionam "search_manuals" como nome conceitual) |
| Cliente MCP | `MCPSearchClient` (`apps/backend/src/services/mcp_client.py`) — SDK oficial `mcp` (mesmo pacote do mcp-server), transporte `streamable-http`, `initialize` → `call_tool` |
| Modelo Ollama | `OLLAMA_MODEL` — default `llama3.2:3b` (Llama 3.2 3B, CLAUDE.md §4) |
| Cliente Ollama | `OllamaClient` (`apps/backend/src/services/ollama_client.py`) — HTTP puro via `httpx` (já dependência do backend) contra `POST {OLLAMA_BASE_URL}/api/chat`, `stream: false` |
| Processamento local (RNF-46) | Nenhuma chamada a OpenAI/Anthropic/Gemini/API externa — `OLLAMA_BASE_URL` aponta para o Ollama do host (`host.docker.internal`, resolvido nativamente pelo Docker Desktop, **sem** `extra_hosts`) |
| ChromaDB | Reaproveitado — **nenhum segundo ChromaDB criado**. O backend nunca fala com o Chroma diretamente, só via MCP (ver "Decisão de arquitetura" abaixo) |

#### Exemplo — requisição/resposta reais

```bash
curl -X POST http://localhost/api/v1/maintenance/suggest \
  -H "Content-Type: application/json" \
  -d '{
    "failure_probability": 0.92,
    "equipment_name": "Bomba centrifuga BC-200",
    "symptom_description": "vazamento na bomba centrifuga"
  }'
```

```json
{
  "triggered": true,
  "failure_probability": 0.92,
  "markdown": "# Plano de Manutenção\n\n## Diagnóstico provável\nVazamento na bomba centrifuga é um sintoma que pode ser relacionado ao desalinhamento do eixo ou à deterioração dos rolamentos.\n\n## Procedimento recomendado\n1. Verificar o alinhamento do eixo e o estado dos rolamentos.\n2. Realizar uma verificação mais detalhada da bomba centrifuga...\n\n## Ferramentas / peças\nNão especificado nos trechos recuperados.\n\n## Cuidados de segurança\nVerificar o manual específico do modelo BC-200...\n\n## Referências\n- `manual-bomba-centrifuga.pdf`, página 1",
  "references": [
    {"file_name": "manual-bomba-centrifuga.pdf", "page": 1, "chunk_index": 0, "source": "manual-bomba-centrifuga.pdf", "score": 0.7802}
  ],
  "model": "llama3.2:3b",
  "message": null
}
```

Com `failure_probability <= 0.7` (ex.: `0.70` exato), a resposta é imediata, sem chamar MCP/Ollama:

```json
{"triggered": false, "failure_probability": 0.7, "markdown": null, "references": [], "model": null, "message": "Probabilidade de falha (0.70) não excede o limiar de 0.7 — sugestão automática não acionada."}
```

#### Estratégia RAG e System Prompt

O System Prompt (constante `SYSTEM_PROMPT` em `maintenance_suggestion_service.py`) instrui o Llama a: responder em português; usar **exclusivamente** o contexto fornecido; nunca inventar peças/torques/temperaturas/pressões; declarar "Limitações" quando o contexto for insuficiente; preservar avisos de segurança; nunca afirmar execução física; citar arquivo+página. O contexto de cada chunk é delimitado (`[MANUAL N]` com `Arquivo`/`Página`/`Score`/`Conteúdo`) e enviado como mensagem **separada** do System Prompt (`role: user`) — nunca concatenado.

**Segurança contra prompt injection**: o System Prompt instrui explicitamente o modelo a tratar o conteúdo de cada `[MANUAL N]` como DADO, nunca como instrução — mesmo que um PDF malicioso contenha algo como "ignore as regras anteriores". Testado em `test_manual_content_never_merged_into_system_prompt`.

**Sem contexto relevante**: o serviço sempre chama o Ollama quando o threshold é ultrapassado (mesmo com 0 resultados do MCP) — o prompt deixa isso explícito ("Nenhum trecho de manual relevante foi recuperado") e é o **próprio modelo**, via System Prompt, quem decide declarar a seção "Limitações" (não o código Python, que não tem como julgar "informação suficiente"). Ver Known Issues abaixo — Llama 3.2 3B nem sempre segue esse branch perfeitamente.

#### Decisão de arquitetura — sem segundo ChromaDB

A especificação desta task menciona `chromadb`/`CHROMADB_HOST`/porta `8001:8000` como possíveis requisitos de infraestrutura. **Não foram criados** — o ChromaDB já existe, persistido e funcional, dentro de `apps/mcp-server` (RF-20/RF-21). Criar uma segunda instância exigiria: (a) decidir qual delas é a fonte de verdade, (b) migrar/reindexar os dados, (c) manter as duas sincronizadas — duas fontes de verdade sem benefício real, já que o backend só precisa de **busca**, não de acesso direto ao vetor store. O backend fala com o ChromaDB **exclusivamente através do MCP** (`search_maintenance_manual`), preservando a separação de responsabilidades da RF-19/RNF-43 (MCP como domínio próprio, backend como consumidor). `chromadb`/`pypdf` **não foram adicionados** a `apps/backend/requirements.txt` — continuam exclusivos do mcp-server.

#### Testes

```bash
docker compose exec api pytest tests/test_maintenance_suggestion.py tests/test_maintenance_endpoint.py -v
```

26 testes — threshold (0.69/0.70/0.7001/0.90 parametrizados), MCP/Ollama indisponíveis, resposta inválida do Ollama, contexto/metadados preservados, prompt contém só o contexto recuperado, prompt injection, contrato HTTP (200/503/422). Todos mockam MCP e Ollama — ver "Validação real" abaixo para a prova sem mocks.

#### Validação real (não só mocks)

Executado de ponta a ponta contra o ambiente real desta task (mcp-server real + Ollama real com `llama3.2:3b` rodando localmente):

| Cenário | probability | Resultado real |
|---|---|---|
| Abaixo do threshold | `0.40` | `triggered=false`, MCP/Ollama não chamados (confirmado via `docker compose exec api`) |
| Exatamente no threshold | `0.70` | `triggered=false` — confirma `>` estrito, não `>=` |
| Acima, contexto relevante | `0.92` (bomba centrífuga) | `triggered=true`, 1 referência (`manual-bomba-centrifuga.pdf`, página 1, score 0.78), Markdown completo e coerente |
| Acima, contexto relevante (2º manual) | `0.87` (motor elétrico) | `triggered=true`, referência correta ao `manual-motor-eletrico.pdf` |
| Acima, sem contexto relevante | `0.95` (equipamento inexistente no corpus) | `triggered=true`, `references=[]`, plano com "Não especificado nos trechos recuperados" em vez de inventar procedimento |

Validado via `curl`/`http.client` diretamente contra `http://localhost/api/v1/maintenance/suggest` (Nginx → api → mcp-server/Ollama reais).

#### Benchmark (RNF-46)

```bash
docker compose exec api python benchmark_maintenance_suggestion.py
```

RNF-46 não define um SLA de latência explícito — o benchmark documenta a latência real medida (MCP, Ollama e total separados), sem inventar um PASS/FAIL. Resultado real desta task (10 execuções + 1 warm-up, `llama3.2:3b`, CPU):

| Etapa | p50 | p95 | p99 | média | min | max |
|---|--:|--:|--:|--:|--:|--:|
| MCP | 31.0 ms | 51.3 ms | 57.4 ms | 34.9 ms | 29.9 ms | 59.0 ms |
| Ollama | 1506.1 ms | 1954.9 ms | 2066.2 ms | 1512.2 ms | 1044.8 ms | 2094.0 ms |
| **Total** | **1542.2 ms** | **1987.6 ms** | **2099.3 ms** | **1547.1 ms** | **1075.0 ms** | **2127.3 ms** |

Evidência completa: [`apps/backend/maintenance_suggestion_benchmark.json`](apps/backend/maintenance_suggestion_benchmark.json). O Ollama (Llama 3.2 3B em CPU) domina a latência total, como esperado para geração de texto local sem GPU — o MCP contribui só ~2% do tempo total.

### 4.9. Streaming da sugestão de manutenção — SSE (RF-23 / RNF-47)

**O que é**: `POST /v1/maintenance/suggest/stream` — mesma regra de negócio da §4.8 (threshold `> 0.7`, MCP, RAG, System Prompt), mas transmite os **tokens reais do Ollama** via Server-Sent Events conforme são gerados, em vez de esperar a resposta completa. `MaintenanceAssistant` (painel lateral, `apps/frontend/components/maintenance-assistant.tsx`) consome esse stream e atualiza o Markdown incrementalmente.

> **Canal independente do SSE de sensores.** `/api/stream/sensors` (RF-12, §4.3) e `/api/v1/maintenance/suggest/stream` são dois endpoints SSE completamente separados — hooks diferentes (`useSSE` vs. o parser próprio em `lib/maintenance-stream.ts`), protocolos de evento diferentes, ciclos de vida diferentes (o de sensores é permanente/infinito; o de manutenção é um request-response de vida curta). Nenhum contrato do SSE de sensores foi alterado por esta task.

```mermaid
flowchart TD
    UI["MaintenanceAssistant<br/>(clique em 'Gerar sugestão')"] --> FETCH["fetch() POST + ReadableStream<br/>lib/maintenance-stream.ts"]
    FETCH --> EP["POST /v1/maintenance/suggest/stream"]
    EP --> SVC["MaintenanceSuggestionService.suggest_stream()"]
    SVC -->|"searching"| MCP["MCP: search_maintenance_manual"]
    MCP -->|"token* (por chunk NDJSON real do Ollama)"| OLLAMA["Ollama /api/chat, stream: true"]
    OLLAMA -->|"done: markdown completo + references"| UI
```

#### Protocolo de eventos SSE

| Evento | `data` | Quando |
|---|---|---|
| `searching` | `{}` | Logo após o threshold passar, MCP acabou de ser chamado (real — não é enviado antes de a chamada acontecer) |
| `token` | `{"token": "..."}` | Um chunk de texto novo, direto do NDJSON de streaming do Ollama — nunca um split artificial de uma resposta já completa |
| `done` | `{"markdown": "...", "references": [...]}` | Geração concluída — Markdown completo (concatenação exata dos tokens) + referências (`file_name`, `page`, `chunk_index`, `source`, `score`) |
| `skipped` | `{"message": "..."}` | `failure_probability <= 0.7` — nem MCP nem Ollama chamados |
| `error` | `{"message": "..."}` | MCP/Ollama indisponível, resposta inválida, ou falha inesperada — mensagem sempre segura (sem URL interna, sem traceback) |

Cada evento é `event: <tipo>\ndata: <json>\n\n` — mesmo padrão hand-rolled de `routers/stream.py` (RF-12), sem `sse_starlette`.

#### Streaming real do Ollama (RNF-47)

`OllamaClient.generate_stream` (`apps/backend/src/services/ollama_client.py`) chama `POST {OLLAMA_BASE_URL}/api/chat` com `"stream": true` — o Ollama devolve NDJSON (uma linha JSON por chunk, `done: true` só na última). Cada `message.content` é repassado ao chamador assim que chega — **não** há geração completa seguida de `.split()`; o gargalo de latência (ver benchmark abaixo) é inteiramente o tempo real de inferência do Llama 3.2 3B, chunk a chunk.

#### Cliente SSE do frontend — `fetch()`, não `EventSource`

O payload carrega `failure_probability`/`equipment_name`/`symptom_description` — `EventSource` nativo não suporta `POST` com corpo JSON, então `lib/maintenance-stream.ts` consome o stream via `fetch()` + `ReadableStream` com um parser SSE escrito à mão (sem lib nova): acumula bytes num buffer, corta em `\n\n`, sobrevive a um evento dividido entre dois chunks de rede e a vários eventos no mesmo chunk (testado em `__tests__/maintenance-stream.test.ts`).

#### Cancelamento

`AbortController` (`useMaintenanceStream`, `apps/frontend/hooks/use-maintenance-stream.ts`) — o botão "Cancelar" aborta o `fetch`, libera o `reader` e limpa o estado; desmontar o painel (fechar o Sheet) também aborta automaticamente (nenhuma atualização de estado após unmount).

#### Segurança do Markdown

`react-markdown` **sem** `rehype-raw` — HTML bruto do texto gerado (ex.: `<script>`) vira texto literal, nunca um elemento executado. Links Markdown (`[texto](url)`) passam por `isSafeHref`, que só permite `http:`/`https:` — `javascript:`, `data:` e qualquer outro esquema viram texto simples, nunca um `<a>` clicável. Testado em `__tests__/maintenance-assistant.test.tsx` (cenário K) com `<script>` e link `javascript:` reais no Markdown gerado.

#### Referências aos manuais — nunca um link inventado

As referências vêm exclusivamente dos metadados estruturados do evento `done` (nunca de uma URL que o LLM tenha escrito). Como não existe hoje nenhum endpoint que sirva os PDFs de `apps/mcp-server/data/manuals/`, o painel mostra cada referência como **texto** (`manual-x.pdf — página N (score 0.XX)`), nunca como link — ver `ReferencesList` em `maintenance-assistant.tsx`.

#### Estados do painel

`idle` · `connecting` · `searching` · `generating` · `done` · `skipped` · `error` · `offline` — nunca fica preso em "Gerando…" indefinidamente: qualquer falha (MCP indisponível, Ollama indisponível, rede) termina num estado de erro com mensagem clara e permite tentar de novo (o formulário reabilita e o botão "Gerar sugestão" reinicia do zero).

#### Testes

```bash
docker compose exec api pytest tests/test_maintenance_suggestion.py tests/test_maintenance_endpoint.py -v   # 42 testes (service + endpoint, incl. streaming)
docker compose exec frontend pnpm exec vitest run __tests__/maintenance-stream.test.ts __tests__/maintenance-assistant.test.tsx   # 21 testes
```

#### Validação real (E2E, sem mocks)

Executado de ponta a ponta contra o ambiente real (mcp-server real + Ollama real): painel abre, formulário preenchido, tokens aparecem progressivamente na tela (confirmado visualmente, Markdown crescendo incrementalmente antes da conclusão), estado muda `Conectando… → Buscando manual… → Gerando… → Concluído`, referências exibidas como texto (nunca link). Também validado: `failure_probability` no limiar exato `0.70` não dispara; Ollama parado → estado `Erro` com mensagem clara; mcp-server parado → estado `Erro` com mensagem diferente (identifica a causa); reiniciar os serviços e tentar de novo funciona (nenhum travamento).

Métrica real de uma execução completa (`docker compose exec api`, contra Ollama/mcp-server reais e já aquecidos):

| Métrica | Valor real |
|---|---|
| Tempo até o 1º token | 0.28 s |
| Duração total | 1.71 s |
| Chunks/tokens recebidos | 197 |
| Tamanho final do Markdown | 730 caracteres |

Evidência: [`apps/backend/maintenance_stream_e2e_metrics.json`](apps/backend/maintenance_stream_e2e_metrics.json) (uma amostra real, não uma série estatística — RNF-47 não define SLA).

#### Limitações conhecidas

- **Llama 3.2 3B às vezes cita o placeholder interno `[MANUAL N]` literalmente** no texto do "Diagnóstico provável" (deveria usá-lo só como referência mental, citando o manual pelo nome na seção "Referências"). Não compromete a segurança/veracidade do conteúdo (o texto citado continua vindo do contexto real), é um artefato de um modelo pequeno seguindo instruções de formatação de forma imperfeita.
- **Sem endpoint de arquivo estático para os PDFs** — por isso as referências são texto, não link (ver acima). Se um endpoint seguro de download for adicionado no futuro, `ReferencesList` é o único lugar que precisa mudar.
- **Cancelamento manual em ambiente de teste local é difícil de observar via automação de browser** quando o Ollama já está "aquecido" — uma segunda geração pode completar em ~1-2s, mais rápido que o round-trip de uma ferramenta de automação externa. O comportamento de cancelamento em si é validado de forma determinística e confiável no teste automatizado (`__tests__/maintenance-assistant.test.tsx`, cenário L), que controla precisamente o timing via um generator mockado.

### 4.10. Notificações críticas via Telegram (RF-24 / RNF-48)

**O que é**: quando o modelo preditivo reporta `probability > 0.85` (falha **crítica** — limiar próprio, independente do `ALERT_PROBABILITY_THRESHOLD` de 0.70 da RF-14/§4.3 e do `MAINTENANCE_SUGGESTION_THRESHOLD` de 0.70 da RF-22/§4.8), o backend envia uma mensagem de texto simples para um chat do Telegram via `TelegramNotificationAdapter`, respeitando um rate limit de **1 alerta por equipamento a cada 15 minutos**. Reaproveita o mesmo ponto de integração da RF-14 (`AlertService.process_prediction`) — nenhuma segunda inferência, nenhum novo hook no pipeline de ML.

```mermaid
flowchart TD
    PRED["InferencePipelineService / predict.py<br/>(mesma predição da RF-14)"] --> ALERT["AlertService.process_prediction()"]
    ALERT -->|"probability > 0.70 (RF-14)"| WS["ConnectionManager.broadcast_alert()"]
    ALERT -->|"sempre (RF-24)"| CRIT["CriticalFailureNotificationService.notify_if_critical()"]
    CRIT -->|"probability <= 0.85"| SKIP["nada acontece"]
    CRIT -->|"probability > 0.85"| LOCK["TelegramAlertRateLimiter.try_acquire(equipment_id)<br/>(UPSERT atômico no Postgres)"]
    LOCK -->|"já bloqueado (< 15 min)"| SKIP2["nada acontece — sem 2ª mensagem"]
    LOCK -->|"lock adquirido"| SEND["TelegramNotificationAdapter.send_critical_failure()"]
    SEND -->|"falha (timeout/HTTP/etc.)"| RELEASE["rate_limiter.release() — permite retry"]
    SEND -->|"sucesso"| DONE["lock mantido até expirar (TTL 900s)"]
```

#### Threshold (RF-24) — independente dos demais

| Constante | Valor | Onde |
|---|---|---|
| `ALERT_PROBABILITY_THRESHOLD` (RF-14) | `0.70` | `apps/backend/src/core/ws_manager.py` |
| `MAINTENANCE_SUGGESTION_THRESHOLD` (RF-22) | `0.70` | `apps/backend/src/services/maintenance_suggestion_service.py` |
| `CRITICAL_FAILURE_THRESHOLD` (RF-24) | **`0.85`** | `apps/backend/src/services/critical_failure_notification_service.py` |

As três constantes têm o mesmo valor numérico só por coincidência em dois casos — cada task pediu explicitamente uma constante própria, não reutilização. `probability == 0.85` exato **não** dispara (regra estrita `>`), validado em `test_notifications.py` com os quatro valores do critério de aceite (`0.0`/`0.5`/`0.84`/`0.85` → nunca chama o adapter; `0.8501`/`0.90`/`0.9999` → sempre chama).

#### Rate limiting — por que Postgres, não Redis

Auditoria do projeto antes de implementar: **não existe Redis (nem qualquer outro cache) em nenhum lugar do `docker-compose.yml` ou do backend**, e nenhuma tabela/serviço tem hoje o conceito de `equipment_id` (o pipeline simula um único ativo real, `APU-Trem-042`). Duas opções avaliadas:

1. **Adicionar Redis só para isto** — rejeitada: um serviço/container novo (mais uma peça de infra para subir, versionar e documentar) para resolver um único contador com TTL, quando o projeto já tem um banco transacional rodando exatamente para este tipo de estado.
2. **Postgres existente** (escolhida) — uma tabela nova de uma linha por equipamento (`telegram_alert_locks`, migração Alembic `0002`), com o mesmo padrão de acesso (SQLAlchemy async) já usado por toda a persistência do projeto.

**Mecanismo** (`TelegramAlertRateLimiter.try_acquire`, `apps/backend/src/services/telegram_alert_rate_limiter.py`): um único `INSERT ... ON CONFLICT (equipment_id) DO UPDATE ... WHERE telegram_alert_locks.expires_at <= :now RETURNING equipment_id` — equivalente funcional ao `SET NX EX 900` do Redis:

- Linha não existe → `INSERT` cria com `expires_at = now + 900s`, `RETURNING` devolve a linha → lock adquirido.
- Linha existe e **já expirou** (`expires_at <= now`) → a cláusula `WHERE` do `DO UPDATE` casa, a linha é atualizada com um novo `expires_at`, `RETURNING` devolve a linha → lock adquirido (retry após expirar funciona).
- Linha existe e **ainda não expirou** → a cláusula `WHERE` não casa, nenhuma linha é afetada, `RETURNING` não devolve nada → lock negado.

Tudo em uma única instrução SQL — sem "ler o valor, decidir em Python, escrever de volta" (que teria uma janela de corrida entre o SELECT e o UPDATE). `release()` (chamado quando o envio ao Telegram falha, para permitir retry imediato em vez de esperar o TTL) é um `DELETE` simples pela chave primária.

Implementação dialect-aware: `sqlalchemy.dialects.postgresql.insert` em produção, `sqlalchemy.dialects.sqlite.insert` nos testes (mesmo `ON CONFLICT` funciona nos dois) — selecionado em runtime via `db.get_bind().dialect.name`, o que permite testar o **mesmo código** contra SQLite em memória (`StaticPool`, para todas as conexões da suíte de teste compartilharem o mesmo `:memory:`) e contra o Postgres real de produção sem nenhum branch condicional na lógica de negócio.

**Limitações conhecidas desta abordagem**: (a) o lock não é liberado automaticamente por um processo de limpeza (linhas expiradas só somem quando um novo `try_acquire` do mesmo equipamento roda o `UPSERT` — em prática nunca acumula mais que 1 linha por equipamento, então não é um vazamento real); (b) diferente de um TTL nativo do Redis, a expiração aqui é **verificada na leitura** (`WHERE expires_at <= now`), não expurgada em background — irrelevante para o volume atual (1 linha por ativo simulado), mas não escalaria da mesma forma para milhares de equipamentos sem um índice dedicado em `expires_at` (não criado nesta task — não necessário no volume atual).

#### Injeção de relógio para testes determinísticos

`TelegramAlertRateLimiter(clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc))` — o teste de limite de TTL (`test_notifications.py`) substitui `clock` por uma função que lê de um `dict` mutável, avançando o "tempo" para `t=0`, `+5min`, `+14min59s`, `+15min` sem nenhum `sleep()` real (suíte inteira roda em milissegundos).

#### Concorrência — prova de exclusão mútua real

Teste dedicado dispara **duas chamadas concorrentes reais** de `try_acquire("mesmo-equipamento")` via `asyncio.gather`, contra o mesmo banco SQLite compartilhado (`StaticPool`), e verifica `sorted(resultados) == [False, True]` — exatamente uma das duas adquire o lock, nunca as duas, nunca nenhuma. A atomicidade vem do `INSERT ... ON CONFLICT` ser uma única instrução SQL avaliada pelo motor do banco, não de qualquer lock em memória do Python (que não protegeria contra múltiplos workers/processos do Uvicorn).

#### Dashboard URL — construção segura (anti open-redirect)

`CriticalFailureNotificationService._build_dashboard_url` monta a URL **exclusivamente** a partir de `settings.dashboard_url` (config do operador, variável de ambiente `DASHBOARD_URL`) concatenada com um path fixo `/sensors/{equipment_id}`, onde `equipment_id` vem sempre do **estado interno do pipeline** (nunca da predição do modelo ou da saída do LLM). Nenhum campo do payload de predição, do MCP ou do Ollama jamais é lido para compor esta URL — não há como injetar um domínio externo através de nenhum desses caminhos.

#### Proteção do token

`TELEGRAM_BOT_TOKEN` nunca aparece em log: a URL completa (que carrega o token no path, convenção da Bot API — `.../bot<TOKEN>/sendMessage`) nunca é logada; em erro HTTP, só `status_code` é logado (nunca o corpo da resposta, que em alguns erros 4xx pode ecoar dados da requisição de volta). Testado explicitamente (`test_notifications.py`, teste parametrizado que varre todos os `caplog` de todos os cenários de falha e garante que a substring do token nunca aparece).

#### Criando o Bot do Telegram (BotFather)

1. Abra uma conversa com [@BotFather](https://t.me/BotFather) no Telegram.
2. Envie `/newbot`, escolha um nome de exibição e um `username` terminado em `bot` (ex.: `predictiq_alerts_bot`).
3. O BotFather devolve o token no formato `123456789:AAdefault-example-do-not-use-this-token` — copie para `TELEGRAM_BOT_TOKEN` no `.env` (nunca commitar este valor).
4. Para descobrir o `TELEGRAM_CHAT_ID`: envie qualquer mensagem para o bot recém-criado, depois abra `https://api.telegram.org/bot<TOKEN>/getUpdates` no navegador e leia `result[0].message.chat.id` (número, pode ser negativo para grupos).
5. Preencha `TELEGRAM_CHAT_ID` no `.env` e reinicie o backend (`docker compose up -d --force-recreate api`).

Sem `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` configurados, `notify_if_critical` simplesmente loga `telegram_config_missing` e não faz nenhuma tentativa de rede — o restante do pipeline (predição, alerta WS, sugestão de manutenção) continua funcionando normalmente.

#### Variáveis de ambiente (RF-24)

| Variável | Obrigatória | Default | Descrição |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Não* | _(vazio)_ | Token do BotFather. *Sem ele, notificação fica desativada (não é erro) |
| `TELEGRAM_CHAT_ID` | Não* | _(vazio)_ | Chat/grupo de destino. *Mesma regra acima |
| `TELEGRAM_API_BASE_URL` | Não | `https://api.telegram.org` | Só para testes/mock — produção não precisa mudar |
| `TELEGRAM_CLIENT_TIMEOUT_SECONDS` | Não | `10.0` | Timeout do HTTP client do adapter |
| `DASHBOARD_URL` | Não | `http://localhost` | Base da URL do alerta — trocar para o domínio real em produção |
| `CRITICAL_FAILURE_RATE_LIMIT_SECONDS` | Não | `900` | TTL do rate limit (15 min) |
| `DEFAULT_EQUIPMENT_ID` | Não | `APU-Trem-042` | Mesmo ID já usado na rota do frontend `/sensors/APU-Trem-042` |
| `DEFAULT_EQUIPMENT_NAME` | Não | `Compressor de Ar Industrial (MetroPT-3)` | Nome exibido na mensagem do Telegram |

#### Testes

```bash
docker compose exec api pytest tests/test_notifications.py -v   # 35 testes
```

Cobre: `TelegramNotificationAdapter` (sucesso, config ausente, timeout, HTTP não-200, erro de conexão, JSON inválido, `ok: false`, token nunca aparece em log); `TelegramAlertRateLimiter` (1º alerta passa, 2º bloqueado, equipamentos diferentes não se bloqueiam mutuamente, `release()` permite retry imediato, limite de TTL nos 4 pontos do critério de aceite, concorrência real via `asyncio.gather`); `CriticalFailureNotificationService` (limiar nos 7 valores do critério de aceite, rate-limit impede chamada ao Telegram, envio bem-sucedido não libera o lock, falha do Telegram libera o lock e não propaga exceção, exceção inesperada não propaga, URL do dashboard construída corretamente).

#### Validação real de infraestrutura (sem Telegram real)

- Migração `0002_create_telegram_alert_locks_table` aplicada de verdade (`docker compose exec api alembic upgrade head`) e a tabela `telegram_alert_locks` inspecionada via `psql \d telegram_alert_locks` no Postgres real do projeto.
- Concorrência atômica provada contra o Postgres real (além do teste automatizado contra SQLite): `asyncio.gather` de dois `try_acquire` simultâneos para o mesmo equipamento, confirmando exatamente um `True`.
- Cadeia real `get_alert_service()` → `AlertService.process_prediction()` → `CriticalFailureNotificationService` exercitada com payloads sintéticos de alta probabilidade (0.90/0.92/0.80) diretamente contra o singleton real de produção (sem token real configurado — validou o caminho "config ausente, não derruba o pipeline").
- Caminho de sucesso/falha do Telegram validado com classes reais (`AlertService`, `CriticalFailureNotificationService`, `TelegramNotificationAdapter`) e só a chamada HTTP (`httpx.AsyncClient.post`) mockada: 1ª chamada envia e mantém o lock; 2ª chamada para o mesmo equipamento é bloqueada pelo rate limit e o Telegram **não** é chamado de novo (`await_count == 1` nas duas tentativas).
- **Telegram real (API pública) não foi testado** — não há `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` reais disponíveis neste ambiente de desenvolvimento. Todos os testes de HTTP usam mock (`respx`/`unittest.mock`) ou um `TELEGRAM_API_BASE_URL` apontando para um servidor fake — nenhum resultado de envio real foi fabricado ou assumido como sucesso.
- Simulador restaurado para `NORMAL` e `telegram_alert_locks` confirmada vazia (0 linhas) ao final da validação manual.

### 4.11. Configuração de alertas — página `/settings/alerts` (RF-25 / RNF-49)

**O que é**: uma página (`apps/frontend/app/settings/alerts`) onde o operador ajusta, via slider, o limiar que dispara a notificação crítica (RF-24) e liga/desliga cada **canal de notificação** — Telegram e/ou e-mail (via [Resend](https://resend.com)) — sem precisar editar `.env`/reiniciar o backend. Um botão **Testar Notificação** valida os canais habilitados sem depender de uma predição real.

> **Configuração GLOBAL, não por usuário.** O projeto não possui nenhum sistema de autenticação/usuários (auditado explicitamente antes desta task — só existe o token de admin compartilhado do RF-11, binário, sem identidade individual). Por isso `alert_settings` é uma tabela **singleton** (uma única linha para o sistema inteiro), protegida pelo mesmo `X-Admin-Token` já usado por `/models` — nenhuma autenticação nova foi criada.

> **E-mail é um canal novo.** Auditoria explícita antes desta task confirmou que **nenhuma integração de e-mail existia no projeto** (nem Resend, nem SMTP, nem qualquer outro provedor) — o `EmailNotificationAdapter` introduzido aqui é o primeiro canal de e-mail do PredictIQ, não uma reintegração de algo já existente.

```mermaid
flowchart TD
    UI["/settings/alerts<br/>AlertSettingsForm"] -->|"GET"| EP1["GET /v1/settings/alerts"]
    UI -->|"PUT ao salvar"| EP2["PUT /v1/settings/alerts"]
    EP1 --> SVC["AlertSettingsService"]
    EP2 --> SVC
    SVC --> DB[("alert_settings<br/>singleton, id=1<br/>threshold + canais + e-mail")]
    SVC -.->|"get_settings()"| CRIT["CriticalFailureNotificationService (RF-24)"]
    CRIT -->|"telegram_enabled"| TG["TelegramNotificationAdapter"]
    CRIT -->|"email_enabled + alert_email"| EM["EmailNotificationAdapter (Resend)"]
    UI -->|"Testar Notificação"| EP3["POST /v1/settings/alerts/test"]
    EP3 --> NTS["NotificationTestService"]
    NTS -->|"rate limit próprio, 10s"| LOCK[("telegram_alert_locks<br/>chave reservada __settings_notification_test__")]
    NTS -->|"só canais habilitados"| TG
    NTS -->|"só canais habilitados"| EM
```

#### Threshold — o que o slider representa

O slider (`0.5`–`0.95`) configura o **mesmo limiar de falha crítica do RF-24** (`CRITICAL_FAILURE_THRESHOLD`) — não um novo tipo de alerta. Essa decisão vem diretamente do enunciado da task (que amarra o RF-25 ao `CRITICAL_FAILURE_THRESHOLD = 0.85` do RF-24 e manda preservar `0.85` como default) e do fato de o botão "Testar Notificação" só fazer sentido para os canais Telegram/e-mail. `CriticalFailureNotificationService._resolve_settings()` agora lê `AlertSettingsService.get_settings()` a cada predição: usa o valor salvo (threshold + canais) se existir, ou os defaults do RF-24 (`0.85`, Telegram ligado, e-mail desligado) se nunca configurado — o comportamento do RF-14 (alerta WebSocket, `ALERT_PROBABILITY_THRESHOLD = 0.70`) **não é afetado** por esta task.

#### Banco — singleton real, não "combinado por convenção"

`alert_settings` (migração `0003`) tem `id` travado em `1` por `CHECK (id = 1)` — combinado com a chave primária, é fisicamente impossível ter uma segunda linha, não é uma regra só respeitada pelo código da aplicação. Reforçados também por CHECK constraint no próprio banco (defesa em profundidade — a primeira barreira é sempre a validação do Pydantic, que já devolve 422 antes de qualquer SQL rodar):

| Constraint | Regra |
|---|---|
| `ck_alert_settings_singleton` | `id = 1` |
| `ck_alert_settings_threshold_range` | `0.5 <= alert_threshold <= 0.95` |
| `ck_alert_settings_email_required_when_enabled` | `NOT email_enabled OR alert_email IS NOT NULL` |

`AlertSettingsService.upsert_settings` usa o mesmo padrão de UPSERT atômico do RF-24 (`INSERT ... ON CONFLICT (id) DO UPDATE ... RETURNING`, dialect-aware Postgres/SQLite) — salvar de novo sempre atualiza a mesma linha, nunca cria uma nova. `get_settings()` nunca cria a linha (sem efeito colateral numa leitura) — antes do primeiro `PUT`, devolve os defaults sem tocar no banco. `get_threshold()`/`upsert_threshold()` (RF-24) continuam funcionando — implementados sobre `get_settings()`/`upsert_settings()`, preservando os campos de canal já salvos quando só o limiar é alterado por esse caminho legado.

#### Canais de notificação — Telegram e e-mail (Resend)

| Campo | Default | Descrição |
|---|---|---|
| `telegram_enabled` | `true` | Reaproveita o `TelegramNotificationAdapter` do RF-24 — nenhum cliente novo |
| `email_enabled` | `false` | Novo canal — `EmailNotificationAdapter`, API REST do Resend via `httpx` puro (sem SDK, mesmo estilo do Telegram) |
| `alert_email` | `null` | Obrigatório quando `email_enabled=true` (Pydantic `EmailStr` + CHECK constraint) |

**O frontend nunca configura credenciais** — só os toggles ON/OFF e o endereço de e-mail. `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`/`RESEND_API_KEY` continuam exclusivamente em variáveis de ambiente do backend (RF-24 §2, mesma regra estendida ao Resend).

**Falha parcial de canais** (RF-25 §17) — um canal falhar nunca impede a tentativa do outro. `CriticalFailureNotificationService.notify_if_critical` tenta Telegram e e-mail independentemente e só libera o rate limit (permitindo retry) se **nenhum** dos canais habilitados enviou com sucesso; se ao menos um funcionou, a janela de 15 min é mantida (evita reenviar o mesmo alerta pelo canal que já funcionou). O mesmo raciocínio vale para `NotificationTestService`: a resposta de sucesso mostra o resultado por canal (ex.: `"Telegram: enviado · E-mail: falhou"`) — só retorna erro (502) se **todos** os canais habilitados falharem, e erro (400) se **nenhum** canal estiver habilitado.

#### Endpoints

| Método | Rota | Auth | Descrição |
|---|---|---|---|
| `GET` | `/v1/settings/alerts` | `X-Admin-Token` | Configuração atual, ou os defaults se nunca salva |
| `PUT` | `/v1/settings/alerts` | `X-Admin-Token` | Valida threshold (`0.5–0.95`) e e-mail (formato + obrigatório se habilitado) — 422 caso contrário; persiste; devolve a configuração salva |
| `POST` | `/v1/settings/alerts/test` | `X-Admin-Token` | Testa só os canais habilitados — 400 se nenhum habilitado, 502 se todos falharem |

Exemplo de `GET`/`PUT`:

```json
{
  "alert_threshold": 0.75,
  "telegram_enabled": true,
  "email_enabled": true,
  "alert_email": "alertas@empresa.com.br"
}
```

#### Botão "Testar Notificação" — reaproveita os adapters, rate limit próprio

Testa exatamente os canais habilitados na configuração **já salva** (não o rascunho não salvo na tela) via `TelegramNotificationAdapter.send_test_notification()`/`EmailNotificationAdapter.send_test_email()` — mesmos clientes HTTP do fluxo real, mensagens claramente marcadas como teste (nunca o texto de falha crítica real). **Nunca** aciona `CriticalFailureNotificationService`/`AlertService` — não cria alerta no histórico, não consome uma predição real, não afeta nenhum equipamento.

Rate limiter **separado** do de falha crítica: mesma classe `TelegramAlertRateLimiter`, mesma tabela `telegram_alert_locks`, mas uma instância com TTL de 10s (só anti-duplo-clique) e uma chave reservada (`__settings_notification_test__`) que nunca colide com um `equipment_id` real — um clique manual não pode consumir a janela de 15 min que protegeria um alerta crítico genuíno do mesmo equipamento logo em seguida. Uma falha do teste libera o lock imediatamente (permite corrigir a configuração e tentar de novo sem esperar); `TELEGRAM_BOT_TOKEN`/`RESEND_API_KEY` nunca aparecem na resposta HTTP nem em log.

#### Variáveis de ambiente — Resend (novas nesta task)

| Variável | Obrigatória | Default | Descrição |
|---|---|---|---|
| `RESEND_API_KEY` | Não* | _(vazio)_ | API key do Resend. *Sem ela, o canal de e-mail fica desativado (não é erro) |
| `RESEND_FROM_EMAIL` | Não | `PredictIQ <alerts@predictiq.dev>` | Precisa ser um remetente de domínio verificado no Resend em produção |
| `RESEND_API_BASE_URL` | Não | `https://api.resend.com` | Só para testes/mock |
| `RESEND_CLIENT_TIMEOUT_SECONDS` | Não | `10.0` | Timeout do HTTP client do adapter |

Variáveis do Telegram (`TELEGRAM_*`) inalteradas — ver §4.10.

#### Testes

```bash
docker compose exec api pytest tests/test_alert_settings.py -v   # 46 testes (service, HTTP, validação de e-mail, canais, integração real com RF-24, rate limit do teste)
docker compose exec frontend pnpm exec vitest run __tests__/alert-settings-form.test.tsx   # 22 testes
```

#### Validação real (E2E, sem mocks de rede)

Executado contra o ambiente real (Postgres real, backend real, frontend real, via navegador): abrir `/settings/alerts` → mostra `85%`/Telegram ligado/e-mail desligado (defaults, nenhuma config salva); habilitar o switch de e-mail → campo de endereço aparece, botão "Salvar" fica desabilitado com "Informe um e-mail válido..." até um endereço válido ser digitado; digitar `operador@predictiq-dev.com` → "Salvar configurações" habilita → clicar → "Configuração salva." exibida → `SELECT * FROM alert_settings` no Postgres real confirma `telegram_enabled=t, email_enabled=t, alert_email='operador@predictiq-dev.com'`. `Testar Notificação` (sem `TELEGRAM_BOT_TOKEN`/`RESEND_API_KEY` reais neste ambiente) → `502`, log do backend mostra `telegram_config_missing` + `resend_config_missing` separadamente e a resposta combinada `"Telegram: falhou · E-mail: falhou"`, botão vira "Tentar novamente". Validado também via chamada direta à API: desabilitar os dois canais → `POST /test` retorna `400 Nenhum canal de notificação está habilitado.`. `telegram_alert_locks` confirmada vazia (0 linhas) ao final. Configuração devolvida aos defaults do RF-24 (`0.85`, Telegram ligado, e-mail desligado) ao fim da validação.

**Telegram e e-mail reais não foram testados** — nenhuma credencial real (`TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`/`RESEND_API_KEY`) disponível neste ambiente de desenvolvimento (mesma limitação já documentada no RF-24 para o Telegram).

#### Limitações conhecidas

- Sem sistema de usuários no projeto, a configuração é **global** — se o projeto ganhar autenticação multiusuário no futuro, `alert_settings` precisará de uma migração para sair do modelo singleton (`id` fixo) para um modelo por usuário/tenant.
- `RESEND_FROM_EMAIL` usa um domínio placeholder (`predictiq.dev`) — precisa ser trocado para um domínio verificado no Resend antes de qualquer envio real funcionar em produção.

### 4.12. Fila assíncrona de notificações — Celery + Redis (RNF-50 / RNF-51)

**O que é**: o ENVIO real das notificações críticas (RF-24/RF-25 — as chamadas de rede ao Telegram/Resend, que podem levar segundos ou dar timeout) deixou de acontecer dentro do caminho síncrono da inferência. `CriticalFailureNotificationService.notify_if_critical()` continua decidindo se dispara (threshold) e adquirindo o rate limit (Postgres, atômico, inalterado) — mas em vez de chamar os adapters diretamente, **enfileira** um payload serializável via Celery e retorna imediatamente. O envio de fato acontece depois, num processo separado (`celery-worker`), consumindo da fila via Redis.

> **Auditoria antes desta task**: nenhum Celery/Redis existia no projeto. O rate limiter do RF-24 usa Postgres deliberadamente (ver §4.10) — este Redis é **só o broker do Celery**, nunca um substituto do Postgres nem de nenhum outro dado do domínio.

```mermaid
flowchart TD
    PRED["Predição crítica<br/>(probability > threshold)"] --> AS["AlertService.process_prediction()"]
    AS --> CFS["CriticalFailureNotificationService.notify_if_critical()"]
    CFS -->|"threshold + rate limit<br/>(Postgres, síncrono, inalterado)"| DECIDE{"adquiriu o lock?"}
    DECIDE -->|"não"| SKIP["nada acontece"]
    DECIDE -->|"sim"| ENQUEUE["enqueue_notification(payload)<br/>.delay() — RETORNA JÁ"]
    ENQUEUE -.->|"processo separado"| REDIS[("Redis<br/>broker")]
    REDIS --> WORKER["celery-worker<br/>(container próprio)"]
    WORKER --> DISPATCH["dispatch_channels()<br/>MESMA lógica do RF-24/RF-25"]
    DISPATCH -->|"telegram_enabled"| TG["TelegramNotificationAdapter"]
    DISPATCH -->|"email_enabled + alert_email"| EM["EmailNotificationAdapter"]
```

#### Por que Celery — o que exatamente deixou de bloquear

Antes: `notify_if_critical` chamava `adapter.send_critical_failure(...)`/`email_adapter.send_critical_failure_email(...)` diretamente, com `await` — se o Telegram ou o Resend demorassem (rede lenta, timeout de 10s configurado em `TELEGRAM_CLIENT_TIMEOUT_SECONDS`/`RESEND_CLIENT_TIMEOUT_SECONDS`), a predição inteira (`AlertService.process_prediction`, chamada pelo `InferencePipelineService` a cada tick) ficava presa esperando. Depois: o mesmo método monta um `dict` só com tipos serializáveis (`equipment_id`, `probability`, `timestamp`, `dashboard_url`, flags de canal, `alert_email`) e chama `enqueue_notification(payload)` — uma função síncrona que só publica a mensagem no Redis (`Celery.delay()`), sem esperar o consumo. Medição real (ver "Validação real" abaixo): **94 ms** para uma predição crítica completa (decisão + rate limit + enqueue) contra o pipeline de produção real, quando o envio de fato (que teria tentado o Telegram e recebido "configuração ausente") aconteceu **depois**, no worker, em processo separado.

#### Onde cada responsabilidade ficou (nada de RF-24/RF-25 mudou de lugar)

| Responsabilidade | Onde | Mudou? |
|---|---|---|
| Ler threshold/canais configurados | `AlertSettingsService.get_settings()` | Não — mesmo lugar, RF-25 |
| Decidir disparar (`probability > threshold`) | `CriticalFailureNotificationService.notify_if_critical()` | Não — mesmo método, síncrono |
| Adquirir o rate limit (Postgres, atômico) | `TelegramAlertRateLimiter.try_acquire()` | Não — continua **antes** do enqueue, no processo web, exatamente como antes (garante "no máximo um envio por janela" sem depender de idempotência na fila) |
| Enviar ao Telegram/e-mail (chamada de rede) | `dispatch_channels()` — extraído do que antes era o final de `notify_if_critical` | **Sim** — agora roda no `celery-worker`, não mais no processo web |
| Liberar o rate limit se todos os canais falharem | `dispatch_channels()` (dentro do `release()`) | Não — mesma regra, só o processo mudou |

`dispatch_channels()` é a MESMA lógica de fan-out/falha-parcial (RF-25 §17) reaproveitada por dois caminhos: o fallback síncrono (`enqueue_notification=None`, usado por **todos os testes existentes de RF-24/RF-25 sem nenhuma alteração**) e a task do Celery (produção real) — nenhuma regra de negócio duplicada entre os dois.

#### Celery — configuração

`src/core/celery_app.py`: broker Redis (`CELERY_BROKER_URL`), `task_serializer="json"`/`accept_content=["json"]` (**nunca pickle** — evita execução arbitrária de código a partir de uma mensagem da fila), `task_ignore_result=True` e **sem result backend** (fire-and-forget deliberado — nenhum código deste projeto precisa ler de volta o resultado de uma task; adicionar um result backend sem essa necessidade seria complexidade sem benefício real). `broker_connection_retry_on_startup=True` explícito (Celery 6.0 muda o default) — tolera o `celery-worker`/`api` subirem antes do healthcheck do Redis terminar.

#### `celery-worker` — o worker

Container próprio no `docker-compose.yml`, **mesma imagem do backend** (`build: ./apps/backend`, mesmo `Dockerfile`, mesmo bind mount de código) — reaproveita 100% do código já existente, não é uma segunda aplicação nem sobe um servidor HTTP. Comando: `celery -A src.core.celery_app worker --loglevel=info`. `depends_on` Postgres e Redis saudáveis (mesmo `condition: service_healthy` já usado por `api`).

`src/tasks/notification_tasks.py` reconstrói os adapters (`TelegramNotificationAdapter`, `EmailNotificationAdapter`, `TelegramAlertRateLimiter`) **dentro do processo do worker**, a partir das MESMAS variáveis de ambiente do processo web — os adapters do processo web nunca atravessam o broker (não são serializáveis).

> **Detalhe de engenharia não óbvio**: cada execução da task roda `asyncio.run(...)` (Celery é síncrono por padrão; o código de domínio é `async`) — cada chamada cria um **novo event loop**. O `engine` async do SQLAlchemy é um singleton de módulo (`core/database.py`, compartilhado com o processo web) — sem cuidado, a SEGUNDA task executada por um worker de vida longa quebraria ao tentar reutilizar uma conexão asyncpg aberta no loop da PRIMEIRA (`RuntimeError: Future attached to a different loop`). Corrigido chamando `engine.dispose()` ao final de cada task (fecha as conexões ainda dentro do loop atual, seguro; a próxima task, em loop novo, abre conexões sob demanda). Validado na prática: 3 tasks reais em sequência no mesmo worker, sem erro (ver "Validação real" abaixo).

#### Redis no Docker

Imagem oficial `redis:7-alpine`, `healthcheck` via `redis-cli ping` (mesmo padrão de timing do healthcheck do Postgres). **Sem `ports:`** — só `expose: 6379` (mesmo padrão de `api`/`frontend`/`mcp-server`), nunca alcançável fora da rede Docker interna. **Sem volume persistente** — decisão deliberada: o broker é uma fila de trabalho efêmera (`task_ignore_result=True`), a decisão de negócio (threshold + rate limit) já foi tomada e persistida no Postgres *antes* do enqueue, então perder uma mensagem ainda não consumida num restart do Redis não perde nenhum dado de domínio — só, na pior hipótese, uma notificação que teria sido enviada. Sem `restart:` — mesma convenção do `db` (nenhum serviço deste projeto define uma restart policy hoje).

#### Testes

```bash
docker compose exec api pytest tests/test_celery_notifications.py -v   # 13 testes
```

Cobre: task registrada no app Celery (§9.1); payload 100% serializável em JSON (§9.2); `notify_if_critical` enfileira em vez de chamar o adapter diretamente, e só enfileira se o rate limit foi adquirido (§9.3); **prova de não bloqueio** — `process_prediction()` com um adapter artificialmente lento (`asyncio.sleep`) retorna em <200ms com enqueue, contra ≥300ms no caminho antigo sem Celery, mesmo teste, mesma latência artificial (§9.4); execução direta da task chamando o adapter certo (§9.5); falha do adapter dentro da task não propaga e libera o rate limit (§9.6); Telegram + e-mail juntos continuam sendo tentados pela task (§9.7); nunca pickle; nenhuma credencial no payload ou nos logs.

#### Validação real (E2E, Docker real — sem mocks)

Executado contra `docker compose up -d redis celery-worker api` real: `celery-worker` conectou (`Connected to redis://redis:6379/0`) e registrou a task (`[tasks] . notifications.send_critical_failure`) no boot. Enfileiradas 3 tasks reais via `send_critical_failure_notification_task.delay(payload)` dentro do container `api` (broker real, sem mock) — o worker **recebeu** (`Task ... received`), **executou** (`notification_task_received` → `telegram_config_missing` → `critical_failure_telegram_failed` → `telegram_rate_limit_released` → `notification_task_completed`) e **concluiu com sucesso** (`Task ... succeeded in 0.03–0.06s`) as 3, incluindo duas seguidas no MESMO worker de vida longa (prova de que o `engine.dispose()` evita o bug de event loop). `telegram_alert_locks` confirmada vazia ao final (rate limit corretamente liberado pelas 3 falhas simuladas — sem `TELEGRAM_BOT_TOKEN` real).

**Prova de não bloqueio contra o pipeline de produção real** (não só o teste automatizado): chamado `get_alert_service().process_prediction({"probability": 0.93, ...})` — o singleton REAL, com Postgres e Redis reais — `process_prediction` retornou em **94.4 ms**; o envio (que falhou por falta de token, como esperado) aconteceu depois, no log do `celery-worker`, em processo totalmente separado.

**Telegram/Resend reais não foram testados** — mesma limitação já documentada no RF-24/RF-25 (nenhuma credencial real neste ambiente). O objetivo desta task — provar `producer -> Redis -> celery-worker -> task` e o desacoplamento do caminho de inferência — foi validado de ponta a ponta com infraestrutura 100% real.

#### Limitações conhecidas

- Sem retry automático (`max_retries=0`, deliberado — ver docstring da task): uma falha de rede não tenta de novo sozinha; a próxima predição crítica real do mesmo equipamento (após o rate limit liberado) é o mecanismo de retry natural. Reavaliar se o volume de falhas transitórias justificar um retry com backoff no futuro.
- `celery-worker` depende de Postgres/Redis saudáveis no `docker-compose.yml`, mas **não** espera a migração Alembic do `api` terminar (`depends_on` não sincroniza com o comando interno de outro serviço) — na prática irrelevante, pois nenhuma task real é enfileirada antes do `api` (e seu `alembic upgrade head`) terminar de subir.
- Sem result backend — o processo web nunca sabe se uma notificação específica foi entregue com sucesso (só os logs do worker mostram isso). Aceitável dado o design fire-and-forget desta task; um result backend (ex.: o mesmo Redis) poderia ser adicionado depois se essa visibilidade se tornar necessária.

### 4.13. Suíte de integração do pipeline completo — RF-26 / RNF-52

**O que é**: `apps/backend/tests/test_full_pipeline.py` — uma suíte de integração que atravessa as fronteiras reais entre todos os subsistemas construídos nas tasks anteriores, provando que continuam funcionando **em conjunto**, não só isoladamente: Detecção → `AlertService` → RF-14 (alerta WebSocket) + RF-24/25 (decisão de falha crítica) → RNF-50 (enfileiramento Celery, nunca envio síncrono) → task real; e, do mesmo evento, `MaintenanceSuggestionService` → MCP/RF-20-21 → LLM/RF-22 → plano validado.

> **Achado de auditoria (não uma limitação inventada para o teste)**: `MaintenanceSuggestionService` nunca é chamado a partir de `AlertService`/`CriticalFailureNotificationService` em produção — confirmado por busca em todo `src/`, o único ponto de construção é `routers/maintenance.py`, acionado sob demanda pelo frontend (RF-22/23). Os dois "ramos" do diagrama abaixo são hoje fluxos **independentes** que só compartilham a mesma predição de origem, não uma cadeia de chamadas única. A suíte exercita os dois ramos a partir do mesmo evento determinístico de detecção, sem inventar uma integração direta que não existe no código real.

```mermaid
flowchart TD
    SENSOR["Sensor Data<br/>(determinístico)"] --> DETECT["Detecção<br/>(contrato exato de InferencePipelineService)"]
    DETECT --> AS["AlertService.process_prediction()"]
    AS -->|"RF-14: > 0.70"| WS["broadcast_alert() — WebSocket"]
    AS --> CFS["CriticalFailureNotificationService<br/>RF-24/25: > 0.85"]
    CFS -->|"RNF-50"| ENQ["enqueue_notification() — NUNCA síncrono"]
    ENQ -.->|"testado via chamada direta e controlada"| TASK["celery-worker task"]
    TASK --> ADAPT["Telegram/EmailNotificationAdapter (mockados)"]

    DETECT -.->|"ramo independente, mesma detecção"| MS["MaintenanceSuggestionService.suggest()<br/>RF-22/23: > 0.70"]
    MS --> MCP["MCPSearchClient (mockado na fronteira real)"]
    MCP --> LLM["OllamaClient (mockado)"]
    LLM --> PLAN["Markdown validado + referências"]
```

#### Por que o MCP é mockado, não uma ChromaDB real neste arquivo

`SemanticSearchService`/ChromaDB vivem em `apps/mcp-server` — pacote Python e container Docker inteiramente separados de `apps/backend`, sem `chromadb`/`sentence-transformers` nas dependências do backend (confirmado — nenhum dos dois está instalado no container `api`). O backend só fala com o MCP pela rede (`MCPSearchClient`, protocolo streamable-http), nunca em processo. Mockar em `MCPSearchClient.search_maintenance_manual()` é o **menor ponto de integração real** — a fronteira que `MaintenanceSuggestionService` de fato usa — evitando tanto reimplementar a lógica de cosine similarity (duplicação de produção) quanto inflar o backend com uma dependência pesada só para um teste (mesma classe de problema já documentada para o `mcp-server` no §14 "Limitações conhecidas" do torch/CUDA).

A cobertura real e determinística da busca semântica com ChromaDB efêmero — a que os itens 3-5 do enunciado desta task pediam — já existe em `apps/mcp-server/tests/test_semantic_search.py` (RF-21) e foi **estendida** nesta mesma task com `test_distinguishes_pump_motor_and_irrelevant_queries_deterministically`: um modelo de embeddings fake mas sensível ao conteúdo (ao contrário do `_FixedVectorModel` original, que ignora o texto) prova, com um ChromaDB real em diretório temporário, que uma consulta sobre bomba recupera só o manual de bomba, uma sobre motor só o de motor, e uma irrelevante não recupera nada.

#### RF-26 — limite de 5 segundos

Medido com `time.perf_counter()` no teste principal (`test_full_pipeline_critical_prediction`) — **3.1 ms** medidos numa execução isolada, muito abaixo do limite. O gate mede a orquestração real (threshold → rate limit → enqueue → chamada MCP → montagem de prompt → validação de Markdown) — o LLM é mockado e retorna instantaneamente, então o tempo de geração REAL do Ollama (segundos, medido separadamente em `benchmark_maintenance_suggestion.py`, RF-22) está deliberadamente fora desta medição.

#### RNF-52 — determinismo

Sem `random`/`uuid4` sem controle, sem `datetime.now()` nas entradas do teste (timestamps/queries/scores fixos), sem download de modelo, sem Ollama/MCP/Telegram/Resend/Redis/Postgres reais. `AlertService.process_prediction` (código de produção, inalterado) gera internamente um `message_id` (`uuid4()`) e um timestamp de servidor (`datetime.now()`) — nenhuma asserção do teste depende do valor exato desses dois campos: RNF-52 exige que as **entradas** do teste sejam determinísticas, não que código de produção pré-existente pare de usar relógio/uuid real internamente. Executado duas vezes em sequência durante a validação desta task — resultado idêntico (7 passed) nas duas.

#### Fallbacks testados

| Cenário | Comportamento real confirmado |
|---|---|
| LLM indisponível (`OllamaUnavailableError`) | `suggest()` propaga a exceção estruturada (sem try/except nesse caminho) — nenhum plano fabricado. O MCP já havia sido consultado com sucesso antes (falha isolada ao LLM) |
| MCP indisponível (`MCPUnavailableError`) | `suggest()` propaga a exceção — o Ollama nunca chega a ser chamado sem contexto |
| Nenhum manual indexado (Chroma vazio → `results: []`) | `MaintenanceSuggestionService` **não tem** curto-circuito para contexto vazio — o Ollama É chamado mesmo assim, com um placeholder no prompt; a System Prompt (Regra 4) instrui o modelo a declarar isso em "Limitações" em vez de inventar. `references` fica vazio |
| Predição abaixo de RF-14/RF-22 (0.65) | Nem WS, nem MCP, nem LLM, nem enqueue |
| Predição entre RF-22 (0.70) e RF-24 (0.85) — ex. 0.80 | **Sugestão de manutenção DISPARA** (MCP/LLM chamados), mas a notificação crítica **não é enfileirada** — prova de que os dois limiares são realmente independentes (ver nota de auditoria abaixo) |

> **Nota sobre o valor `0.80`**: o enunciado original desta task usa `0.80` como exemplo de "abaixo do threshold" partindo da premissa de um único limiar para todo o pipeline. A auditoria confirmou que RF-22 (`MAINTENANCE_SUGGESTION_THRESHOLD = 0.70`) e RF-24 (`CRITICAL_FAILURE_THRESHOLD = 0.85`) são constantes independentes (documentado desde o RF-24). O teste reflete o comportamento real do projeto em vez de forçar a premissa do enunciado.

#### Celery — enqueue validado com a task real

`test_enqueued_payload_is_consumed_by_the_real_celery_task` usa o payload exato que o passo de notificação produziria e chama `send_critical_failure_notification_task(...)` diretamente (sem broker/worker reais, RNF-52) para fechar o loop `pipeline -> enqueue -> task -> adapter`. Roda como função síncrona (não `async def`) de propósito — a task real faz `asyncio.run(...)` internamente, e chamar isso de dentro do event loop do pytest-asyncio replicaria um erro que um worker Celery real (processo síncrono) nunca teria.

#### Testes e CI

```bash
docker compose exec api pytest tests/test_full_pipeline.py -v   # 7 testes
```

Adicionado como step isolado no job `test-python` existente do `.github/workflows/ci.yml` (não um workflow novo) — **não** reabilita o step geral de `pytest` daquele job, que segue comentado desde a criação do CI (fora do escopo desta task: reabilitá-lo exigiria provisionar Postgres no runner para as dezenas de testes pré-existentes que dependem dele, um trabalho maior e não relacionado). `test_full_pipeline.py` é 100% determinístico e não depende de nenhum serviço externo, então é seguro rodá-lo isoladamente no CI hoje.

---

## 5. Modelos de Machine Learning

### Modelos disponíveis (9 — todos via `PUT /models/active` sem reiniciar)

| `model_name` | Família | Artefato | Pré-processamento | Observação |
|---|---|---|---|---|
| `random_forest` | Tree (supervisionado) | `.joblib` | feature engineering (rolling) | Baseline; F2-threshold tuning na PR curve |
| `xgboost` | GBM (supervisionado) | `.joblib` | mesmo | Otimizado via Optuna |
| `mlp` | DL feed-forward | `.onnx` + StandardScaler `.joblib` | scaler per-feature | PyTorch Lightning → ONNX (opset 17) |
| `random_forest_v2` | Tree | `.onnx` | feature engineering | Export skl2onnx (paridade com tree models) |
| `xgboost_v2` | GBM | `.onnx` | feature engineering | Export onnxmltools |
| `tcn` | Sequencial DL | `.onnx` + scaler | janela (B, T=60, C=12) | Temporal Convolutional Network |
| `bilstm` | Sequencial DL | `.onnx` + scaler | mesmo | Bidirectional LSTM |
| `patchtst` | Sequencial DL | `.onnx` + scaler | mesmo | Transformer com patching |
| `autoencoder` | **Não-supervisionado** | `.onnx` + scaler | janela (B, T=60, C=12) | Conv1D — score por erro de reconstrução |

### Autoencoder Conv1D — Detecção Não-Supervisionada

```
Entrada: (B, T=60, C=12)         ← janela de 60 ticks × 12 sensores

Encoder:
  Conv1D(12→32, k=4, s=2) + BN + GELU   → (B, 30, 32)
  Conv1D(32→64, k=4, s=2) + BN + GELU   → (B, 15, 64)
  Conv1D(64→128, k=4, s=2) + BN + GELU  → (B,  8, 128)

Decoder:
  ConvTranspose1D(128→64, k=4, s=2)     → (B, 15, 64)
  ConvTranspose1D( 64→32, k=4, s=2)     → (B, 30, 32)
  ConvTranspose1D( 32→12, k=4, s=2)     → (B, 60, 12)

Saída: (B, 60, 12)                ← reconstrução
```

Score de anomalia via sigmoid centrada no threshold de treino:

```
score = sigmoid( (mse − mse_threshold) / (mse_threshold / 3) )

mse = mse_threshold     → score = 0.50  (limiar)
mse = 2 × mse_threshold → score ≈ 0.95  (anomalia clara)
mse ≪ mse_threshold     → score ≈ 0.02  (operação saudável)
```

O `mse_threshold` é o **percentil 99** do MSE calculado nas janelas saudáveis do conjunto de validação, persistido em `autoencoder_v1_card.json`.

### Simulador — Streaming Real do MetroPT-3

| Modo | Comportamento |
|---|---|
| `NORMAL` | Replay sequencial das linhas com `label=0` (operação saudável) |
| `FAILURE` | Replay sequencial das **4 janelas de falha reais** identificadas no paper do MetroPT-3 |
| `DEGRADATION` | Interpolação linear vetorizada (`lerp`) entre linha normal e linha de falha, com drift `0 → 1` em 300 ticks |

Janelas de falha utilizadas:

```
2020-04-18 00:00 → 2020-04-18 23:59
2020-05-29 23:30 → 2020-05-30 06:00
2020-06-05 10:00 → 2020-06-07 14:30
2020-07-15 14:30 → 2020-07-15 19:00
```

```python
drift   = min(step / 300, 1.0)
blended = normal_row + drift * (failure_row - normal_row)
```

O índice é circular (`idx = (idx + 1) % len(array)`) — operação contínua independentemente do tamanho do dataset.

---

## 6. Pré-requisitos

| Ferramenta | Versão mínima | Como instalar |
|---|---|---|
| **Docker Desktop** | 4.x (Compose v2) | https://www.docker.com/products/docker-desktop/ |
| **Git** | 2.30+ | https://git-scm.com/ |
| **Python** (apenas para ML local / Alembic local) | 3.11+ | https://www.python.org/ |
| **Node.js + pnpm** (opcional, apenas para dev sem Docker) | Node 20 + pnpm (via Corepack) | `corepack enable` |

Verificação rápida:

```powershell
docker --version           # Docker version 24.x.x ou superior
docker compose version     # Docker Compose v2.x.x
```

> O projeto foi validado em **Windows 11 com Docker Desktop + WSL2**, mas funciona em Linux/macOS.

---

## 7. Variáveis de Ambiente

> Tabelas de variáveis abaixo atendem RNF-42.

Há **dois arquivos** de ambiente distintos — confundi-los é a causa mais comum de "funciona no meu Docker mas não localmente":

| Arquivo | Onde | Consumido por | Versionado? |
|---|---|---|---|
| `.env` (raiz do projeto) | `projeto-tcc/.env` | Todos os contêineres (`api`, `frontend`, `ml`, `db`, `mlflow`) via `env_file: .env` no `docker-compose.yml` | Não (`.gitignore`) — copie de [`.env.example`](.env.example) |
| `.env.local` (frontend) | `apps/frontend/.env.local` | **Só** o Next.js, e **só** em build/runtime do frontend (lido por `process.env.NEXT_PUBLIC_API_URL` em [`lib/api-client.ts`](apps/frontend/lib/api-client.ts)) | Não (`apps/frontend/.gitignore`: `.env*`) — copie de [`apps/frontend/.env.local.example`](apps/frontend/.env.local.example) |

```bash
cp .env.example .env
cp apps/frontend/.env.local.example apps/frontend/.env.local
```

> **Sem o segundo arquivo, o frontend lança `[api-client] NEXT_PUBLIC_API_URL não está definida.`** em runtime — o Next.js **não lê** o `.env` da raiz, e o `docker-compose.yml` também não injeta essa variável para o serviço `frontend` (nenhum `environment:` a define). Como `apps/frontend` é montado por bind mount no container, o arquivo precisa existir no **host** antes de `docker compose up`.

### 7.1. Variáveis do `.env` (raiz)

| Variável | Obrigatória | Default | Descrição |
|---|---|---|---|
| `POSTGRES_USER` | Sim | — | Usuário do Postgres (usado pelo serviço `db` e por `DATABASE_URL`) |
| `POSTGRES_PASSWORD` | Sim | — | Senha do Postgres |
| `POSTGRES_DB` | Sim | — | Nome do banco (`tcc_db`) |
| `DATABASE_URL` | Sim | `postgresql+asyncpg://postgres:postgres@localhost:5432/tcc_db` (fallback do `Settings`, não usável em Docker) | String de conexão async — hostname `db` dentro da rede Docker, `localhost` se rodando o backend fora do Docker |
| `ACTIVE_MODEL` | Não | `random_forest` | RF-10 — modelo ativo no startup. Valores válidos: `random_forest`, `xgboost`, `mlp`, `random_forest_v2`, `xgboost_v2`, `tcn`, `bilstm`, `patchtst`, `autoencoder` (ver [`model_registry.py`](apps/backend/src/services/model_registry.py):`KNOWN_MODELS`). Hot-swap em runtime via `PUT /models/active` sem precisar mudar esta variável |
| `ADMIN_API_TOKEN` | Não | `change-me-in-production` | RF-11 — token do header `X-Admin-Token` em `/models/*`. **Com o valor default, a autenticação é desativada** (modo dev — ver [`auth.py`](apps/backend/src/core/auth.py)) |
| `SMOTE_SAMPLING_STRATEGY` | Não | `auto` | Só usado pelos scripts de treino em `apps/ml/src/train_*.py` (via `balancing.py`) — **não afeta o runtime da API** |
| `UVICORN_WORKERS` | Não | `1` | RNF-37 — nº de processos Uvicorn do serviço `api`, interpolado no `command:` do `docker-compose.yml` (`--workers ${UVICORN_WORKERS:-1}`). Mudar exige `docker compose up -d --force-recreate api`. **Testado e mantido em 1** — ver §14.7 |
| `DB_POOL_SIZE` | Não | `10` | Tamanho do pool SQLAlchemy/asyncpg, por worker — [`config.py`](apps/backend/src/core/config.py) → [`database.py`](apps/backend/src/core/database.py) |
| `DB_MAX_OVERFLOW` | Não | `20` | Conexões extra sob pico, além do `DB_POOL_SIZE` |
| `PROJECT_NAME` | Não | `Predictive Maintenance API` | Metadado exibido no Swagger (`/docs`) |
| `VERSION` | Não | `0.1.0` | Versão exibida em `/health` e no Swagger |
| `ALLOWED_ORIGINS` | Não | `["http://localhost:3000","http://127.0.0.1:3000"]` | Lista JSON de origens para `CORSMiddleware` — só relevante para chamadas REST **fora** do same-origin do Nginx (ver §4.3, CORS não aplicável a SSE/WS neste setup) |
| `OLLAMA_BASE_URL` | Não | `http://host.docker.internal:11434` | RF-22/RNF-46 — consumido por `OllamaClient` (§4.8). `host.docker.internal` resolvido nativamente pelo Docker Desktop, sem `extra_hosts` |
| `OLLAMA_MODEL` | Não | `llama3.2:3b` | RF-22/RNF-46 — modelo usado por `MaintenanceSuggestionService` |
| `OLLAMA_CLIENT_TIMEOUT_SECONDS` | Não | `120.0` | RF-22 — timeout do cliente HTTP do Ollama (Llama 3.2 3B em CPU pode levar dezenas de segundos) |
| `MCP_CLIENT_TIMEOUT_SECONDS` | Não | `60.0` | RF-22 — timeout do cliente MCP (cobre o cold-start do `SemanticSearchService` no mcp-server, ~30s na 1ª chamada) |
| `MCP_SERVER_PORT` | Não | `8100` | RF-19/RNF-43 — porta do serviço `mcp-server` (`streamable-http`) |
| `MCP_SERVER_URL` | Não | `http://mcp-server:8100` | RF-19/RNF-43 — declarada, ainda não consumida por nenhum código (ver §4.6) |
| `EMBEDDING_MODEL` | Não | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | RF-20/RNF-44 — modelo usado por `apps/mcp-server/index_manuals.py` (ver §4.7). Trocar exige reindexar (o `embedding_model` fica registrado nos metadados de cada chunk, mas embeddings antigos e novos não são comparáveis entre modelos diferentes) |
| `MANUALS_DIR` | Não | `data/manuals` | RF-20 — diretório dos PDFs de entrada, relativo a `apps/mcp-server/` |
| `CHROMA_DB_PATH` | Não | `data/chroma` | RF-20 — diretório persistente do ChromaDB, relativo a `apps/mcp-server/` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Não | `1000` / `150` | RF-20 — tamanho do chunk e overlap (caracteres) usados pelo chunking determinístico |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Não* | _(vazio)_ | RF-24/RNF-48 — credenciais do Bot (ver §4.10). *Sem elas, notificação crítica fica desativada (não é erro) |
| `TELEGRAM_API_BASE_URL` | Não | `https://api.telegram.org` | RF-24 — só para testes/mock |
| `TELEGRAM_CLIENT_TIMEOUT_SECONDS` | Não | `10.0` | RF-24 — timeout do HTTP client do `TelegramNotificationAdapter` |
| `DASHBOARD_URL` | Não | `http://localhost` | RF-24 — base da URL do alerta (nunca vem da predição/LLM — ver §4.10) |
| `CRITICAL_FAILURE_RATE_LIMIT_SECONDS` | Não | `900` | RF-24 — TTL do rate limit de alertas críticos (15 min) |
| `DEFAULT_EQUIPMENT_ID` / `DEFAULT_EQUIPMENT_NAME` | Não | `APU-Trem-042` / `Compressor de Ar Industrial (MetroPT-3)` | RF-24 — identifica o único ativo simulado hoje pelo pipeline |
| `RESEND_API_KEY` | Não* | _(vazio)_ | RF-25/RNF-49 — API key do Resend (ver §4.11). *Sem ela, o canal de e-mail fica desativado (não é erro) |
| `RESEND_FROM_EMAIL` | Não | `PredictIQ <alerts@predictiq.dev>` | RF-25 — precisa ser um remetente de domínio verificado no Resend em produção |
| `RESEND_API_BASE_URL` | Não | `https://api.resend.com` | RF-25 — só para testes/mock |
| `RESEND_CLIENT_TIMEOUT_SECONDS` | Não | `10.0` | RF-25 — timeout do HTTP client do `EmailNotificationAdapter` |
| `CELERY_BROKER_URL` | Não | `redis://redis:6379/0` | RNF-50/RNF-51 — broker do Celery (ver §4.12). Nome do serviço `redis` do `docker-compose.yml` — nunca hardcoded no código |
| `MODEL_PATH`, `XGBOOST_MODEL_PATH`, `MLP_ONNX_PATH`, `MLP_SCALER_PATH`, `RF_V2_ONNX_PATH`, `XGBOOST_V2_ONNX_PATH`, `TCN_ONNX_PATH`, `TCN_SCALER_PATH`, `BILSTM_ONNX_PATH`, `BILSTM_SCALER_PATH`, `PATCHTST_ONNX_PATH`, `PATCHTST_SCALER_PATH`, `AUTOENCODER_ONNX_PATH`, `AUTOENCODER_SCALER_PATH` | Não | resolvidos automaticamente para `apps/ml/models/<arquivo>` | Overrides individuais de caminho de artefato — raramente necessários; usados só se você mover os artefatos para fora de `apps/ml/models/` |

Definidas **pelo `docker-compose.yml`** (não pelo `.env` — não precisam ir no arquivo):

| Variável | Serviço | Valor | Descrição |
|---|---|---|---|
| `PYTHONPATH` | `api` | `/app` | Permite `from src...` funcionar dentro do container |
| `SIMULATOR_PARQUET_PATH` | `api` | `/ml/data/processed/metropt3.parquet` | Caminho do parquet **dentro do container** (volume `./apps/ml/data:/ml/data:ro`) |
| `NODE_OPTIONS` | `frontend` | `--max-old-space-size=3584` | Limite de heap do Node para o dev server |
| `WATCHPACK_POLLING` | `frontend` | `true` | Polling de arquivos (webpack) — **não afeta o Turbopack** (usado por padrão nesta versão do Next.js), que tem seu próprio watcher e não recarrega de forma confiável via bind mount no Windows (ver §16) |

### 7.2. Variáveis do `apps/frontend/.env.local`

| Variável | Obrigatória | Default | Descrição |
|---|---|---|---|
| `NEXT_PUBLIC_API_URL` | **Sim** | nenhum — lança erro em runtime se ausente | RNF-13 — base URL das chamadas REST/paginação (`lib/api-client.ts`). `/api` para Docker Compose (same-origin via Nginx); `http://localhost:8000` para FastAPI standalone sem Nginx. **Não afeta SSE/WS** — `useSSE`/`useAlertWebSocket` montam a URL a partir de `window.location`, não desta variável |
| `NEXT_PUBLIC_MSW_ENABLED` | Não — **não defina manualmente** | (ausente) | Liga o Mock Service Worker (`components/msw-provider.tsx`). Injetada automaticamente como `"true"` só pelo `webServer.env` de [`playwright.config.ts`](apps/frontend/playwright.config.ts) durante `pnpm e2e` — não faz parte do fluxo normal de desenvolvimento/Docker |

### 7.3. Variáveis opcionais do load test (raiz, só para `locust_streaming.py`)

Já documentadas em detalhe no §14.5 — resumo:

| Variável | Obrigatória | Default | Descrição |
|---|---|---|---|
| `STREAMING_AUTH_TOKEN` | Não | nenhum | Enviaria `Authorization: Bearer <token>` — hoje sem efeito, pois `/stream/sensors` não exige auth |
| `SSE_HOLD_SECONDS` | Não | `45` | Quanto tempo cada cliente simulado mantém a conexão SSE aberta |
| `SSE_PATH` | Não | `/api/stream/sensors` | Path do endpoint testado |

> **Atenção**: o backend lê o parquet do simulador em `/ml/data/processed/metropt3.parquet` **dentro do container** (`SIMULATOR_PARQUET_PATH`, definida pelo `docker-compose.yml`). Você precisa **gerar o parquet no host antes da primeira execução** (ver §12) — o volume `./apps/ml/data:/ml/data:ro` só expõe o que já existe em `apps/ml/data/processed/` no seu disco.

---

## 8. Como Rodar (Docker Compose)

O projeto é orquestrado **exclusivamente via Docker Compose**. Todos os 6 serviços sobem com um único comando.

### 8.1. Subindo tudo

```powershell
# Na raiz do projeto — os DOIS arquivos de ambiente precisam existir (§7)
copy .env.example .env
copy apps\frontend\.env.local.example apps\frontend\.env.local

docker compose up --build
```

Para rodar em background:

```powershell
docker compose up --build -d
```

### 8.2. Serviços expostos

| Serviço | URL externa | Notas |
|---|---|---|
| **Nginx (entrypoint)** | `http://localhost` | Proxy unificado para frontend + API + SSE/WS |
| **Frontend Next.js** | `http://localhost` | Servido via Nginx (catch-all `/`) |
| **API FastAPI** | `http://localhost/api/*` | REST + SSE + WS via Nginx |
| **Swagger UI** | `http://localhost/docs` | Documentação interativa OpenAPI |
| **ReDoc** | `http://localhost/redoc` | Documentação alternativa |
| **MLflow UI** | `http://localhost:5000` | Tracking server (porta dedicada) |
| **Jupyter (ML)** | `http://localhost:8888` (rede interna) | Notebook server — adicione um `ports:` mapping no compose se quiser acesso externo |
| **PostgreSQL** | `localhost:5432` | Apenas para conexões locais (já exposto) |
| **Redis** | rede interna (`expose: 6379`) | Broker do Celery (RNF-50/RNF-51) — **sem** `ports:`, nunca alcançável fora da rede Docker |
| **celery-worker** | (nenhuma — não é um servidor HTTP) | Processo `celery worker` puro; ver §4.12 |

> O serviço `api` está em `expose: 8000` (rede interna), **não em `ports:`**. Acesse-o sempre através do Nginx em `http://localhost/api/...`. Por exemplo, `POST http://localhost/api/predict/`.

### 8.3. Parar os serviços

```powershell
# Para e remove os contêineres (volumes persistem)
docker compose down

# Para, remove contêineres E apaga todos os dados (Postgres + MLflow)
docker compose down -v
```

### 8.4. Recriar um serviço específico

```powershell
docker compose up --build api -d
docker compose up --build frontend -d
```

### 8.5. Limites de recursos (já configurados)

| Serviço | Memory limit | CPU limit |
|---|---|---|
| `api` | 3 GB | — |
| `frontend` | 4 GB | 2.0 |
| `ml` | sem limite | — |

Se o seu Docker Desktop estiver com menos de 8 GB de RAM disponíveis, abra **Settings → Resources** e aumente.

---

## 9. Banco de Dados & Migrations

O serviço `api` no `docker-compose.yml` já roda `alembic upgrade head` **antes** de iniciar o Uvicorn:

```yaml
command: sh -c "alembic upgrade head && uvicorn src.main:app --host 0.0.0.0 --port 8000"
```

Portanto, ao subir o sistema pela primeira vez com `docker compose up`, a tabela `predictions` (RF-09) já é criada automaticamente.

### 9.1. Executar migrations manualmente

```powershell
# Container vivo
docker compose exec api alembic upgrade head

# Estado atual
docker compose exec api alembic current

# Histórico
docker compose exec api alembic history --verbose

# Reverter uma migration
docker compose exec api alembic downgrade -1

# Criar nova migration a partir dos models
docker compose exec api alembic revision --autogenerate -m "descrição"
```

### 9.2. Rodando Alembic localmente (fora do Docker)

Necessário Python 3.11+. Suba apenas o banco:

```powershell
docker compose up db -d
```

Crie `apps/backend/.env` apontando para `localhost`:

```dotenv
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/tcc_db
ACTIVE_MODEL=random_forest
ADMIN_API_TOKEN=change-me
```

Instale e execute:

```powershell
cd apps\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
alembic upgrade head
```

### 9.3. Esquema da tabela `predictions`

| Coluna | Tipo | Descrição |
|---|---|---|
| `id` | INTEGER PK | autoincrement |
| `timestamp` | TIMESTAMPTZ | indexed para `ORDER BY DESC` |
| `TP2, TP3, H1, DV_pressure, Reservoirs, Motor_current, Oil_temperature` | FLOAT | sensores analógicos |
| `COMP, DV_eletric, Towers, MPG, Oil_level` | FLOAT | switches digitais (0.0/1.0) |
| `predicted_class` | INTEGER | 0 ou 1 |
| `failure_probability` | FLOAT | [0.0, 1.0] |

---

## 10. Endpoints da API

> Todos os endpoints são expostos via Nginx em `http://localhost/api/...`. A API interna roda em `root_path="/api"` (FastAPI), por isso o Swagger em `http://localhost/docs` mostra os caminhos sem o prefixo `/api`.

### 10.1. REST

| Método | Caminho (via Nginx) | Auth | Descrição |
|---|---|---|---|
| `GET` | `/api/health` | — | Liveness + ping ao Postgres |
| `POST` | `/api/predict/` | — | Inferência stateless + persiste no DB. **Rate limit: 100 req/min/IP**. Retorna 503 se nenhum modelo carregado. |
| `GET` | `/api/v1/predictions?page=1&size=20` | — | Histórico paginado (size ≤ 100) |
| `GET` | `/api/simulator/mode` | — | Retorna `NORMAL` / `DEGRADATION` / `FAILURE` |
| `PUT` | `/api/simulator/mode` | — | Troca o modo do simulador em ≤1 s |
| `GET` | `/api/models` | `X-Admin-Token` | Lista modelos + `artefact_ready` boolean |
| `PUT` | `/api/models/active` | `X-Admin-Token` | Hot-swap atômico (RNF-25); retorna `202 Accepted` |
| `POST` | `/api/v1/maintenance/suggest` | — | RF-22 — sugestão de manutenção via RAG (MCP + Ollama/Llama 3.2 3B) quando `failure_probability > 0.7`. Ver §4.8 |
| `GET` | `/docs` · `/redoc` · `/openapi.json` | — | Documentação |

### 10.2. Streaming

| Tipo | Caminho | Descrição |
|---|---|---|
| **SSE** | `GET /api/stream/sensors` | `text/event-stream` — emite `sensor_reading` 1× por segundo com timestamp + 12 sensores |
| **SSE** | `POST /api/v1/maintenance/suggest/stream` | RF-23/RNF-47 — tokens do Llama 3.2 3B em tempo real (`searching`/`token`/`done`/`skipped`/`error`). Canal independente do SSE de sensores. Ver §4.9 |
| **WebSocket** | `WS /ws/alerts` | JSON: server → `alert` / `ping`; client → `ack` / `pong`. Push imediato quando `probability > 0.70`. Heartbeat a cada 30 s. |

### 10.3. Exemplo — Inferência

```powershell
curl -X POST http://localhost/api/predict/ `
  -H "Content-Type: application/json" `
  -d '{
    "TP2": 5.02, "TP3": 9.21, "H1": 8.97, "DV_pressure": 2.10,
    "Reservoirs": 8.85, "Motor_current": 4.5, "Oil_temperature": 72.3,
    "COMP": 1.0, "DV_eletric": 0.0, "Towers": 1.0, "MPG": 1.0, "Oil_level": 1.0
  }'
```

Resposta:

```json
{
  "predicted_class": 0,
  "failure_probability": 0.12,
  "timestamp": "2026-05-16T12:34:56.789Z"
}
```

### 10.4. Exemplo — Hot-swap de modelo

```powershell
curl -X PUT http://localhost/api/models/active `
  -H "X-Admin-Token: change-me-in-production" `
  -H "Content-Type: application/json" `
  -d '{"model_name": "autoencoder"}'
```

Resposta `202 Accepted`:

```json
{
  "previous_model": "random_forest",
  "active_model": "autoencoder",
  "message": "Model swap from 'random_forest' to 'autoencoder' accepted. Loading in background — use GET /models to track the active model."
}
```

---

## 11. Frontend — Dashboard de Frota

Acesse `http://localhost` após `docker compose up`.

### 11.1. Tela principal

- **Header**: status de conexão SSE (`Telemetria ao vivo` / `Reconectando…`).
- **FleetKPIs**: cards com risco efetivo (`NORMAL` / `ALERTA` / `CRÍTICO`) calculado como `max(probability, alertProb)`.
- **AssetTable**: tabela de frota com o ativo `APU-Trem-042` em modo LIVE (atualizado pelo SSE).
- **AssetRadarChart**: radar com **normalização absoluta** (`min(100, (raw/teto) × 100)`) — sensores em grandezas físicas diferentes são comparados na mesma escala 0–100%.
- **AssetEfficiencyChart**: linha temporal de eficiência.

### 11.2. Normalização do radar

| Sensor | Teto físico | Mediana saudável | % de capacidade |
|---|---|---|---|
| TP2 | 12 bar | 10.1 | 84.2 % |
| TP3 | 12 bar | 10.1 | 84.2 % |
| H1 | 12 bar | 8.5 | 70.8 % |
| Motor_current | 10 A | 3.8 | 38.0 % |
| Oil_temperature | 80 °C | 64 | 80.0 % |
| Reservoirs | 12 bar | 7.0 | 58.3 % |

O polígono verde ("Ótimo") são as medianas saudáveis; o polígono azul ("Atual") é recalculado via `useMemo([sensorData])` a cada tick.

### 11.3. Hooks

| Hook | Função |
|---|---|
| `useSSE(url)` | Conexão SSE com reconexão exponencial; emite `SSEStatus` (`connecting` / `connected` / `error`) |
| `useSensorData()` | Seed via `GET /v1/predictions?size=30`, SSE @1Hz, prediction poll @5s, integração com WS de alertas |
| `useAlertWebSocket()` | Conecta em `/ws/alerts`, envia `ack`, responde `pong`, mantém estado de alertas |
| `usePredictionHistory()` | Paginação do histórico |

### 11.4. Rodando o frontend localmente (sem Docker)

```powershell
cd apps\frontend
corepack enable
pnpm install
pnpm dev
```

Acesse em `http://localhost:3000`. Defina `NEXT_PUBLIC_API_URL=http://localhost:8000` no `.env.local` se rodar a API separadamente.

---

## 12. Pipeline de Machine Learning

O diretório `apps/ml/` é independente, com seu próprio venv e Dockerfile (Jupyter Notebook na porta 8888).

### 12.1. Ambiente local

```powershell
cd apps\ml
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 12.2. Dataset

Baixe o **MetroPT-3 Air Compressor Dataset** da UCI:

https://archive.ics.uci.edu/dataset/791/metropt-3+dataset

Coloque o arquivo CSV em:

```
apps/ml/data/raw/MetroPT3(AirCompressor).csv
```

### 12.3. Pipeline completo

Todos os comandos rodam a partir de `apps/ml/` com o venv ativo:

```powershell
# 1. Ingestão e geração do Parquet (rotulando anomalias via janelas de falha)
python -m src.ingest_metropt

# 2a. Random Forest baseline (SMOTE + GridSearchCV + F2-threshold)
python -m src.train_random_forest

# 2b. XGBoost com tuning Optuna
python src/train_xgboost.py --n-trials 100
# Smoke run: --n-trials 5  (~2 min)

# 2c. MLP via PyTorch Lightning → ONNX
python src/train_mlp.py --max-epochs 50
# Smoke run: --max-epochs 10  (~5 min)

# 2d. Modelos sequenciais (TCN / BiLSTM / PatchTST)
python src/train_sequential.py --arch tcn       --max-epochs 30
python src/train_sequential.py --arch bilstm    --max-epochs 30
python src/train_sequential.py --arch patchtst  --max-epochs 30 --batch-size 128

# Smoke run em 200 mil linhas
python src/train_sequential.py --arch tcn --max-epochs 3 --subsample-rows 200000

# 2e. Autoencoder Conv1D (não-supervisionado — apenas janelas saudáveis)
python src/train_autoencoder.py --max-epochs 50
```

### 12.4. Artefatos gerados

Em `apps/ml/models/`:

```
random_forest_final.joblib       ← ACTIVE_MODEL=random_forest
random_forest_v2.onnx            ← ACTIVE_MODEL=random_forest_v2

xgboost_v1.joblib                ← ACTIVE_MODEL=xgboost
xgboost_v2.onnx                  ← ACTIVE_MODEL=xgboost_v2
xgboost_v1_card.json

mlp_v1.onnx + mlp_v1.onnx.data   ← ACTIVE_MODEL=mlp
mlp_scaler.joblib
mlp_v1_card.json

tcn_v1.onnx + tcn_scaler.joblib + tcn_v1_card.json        ← ACTIVE_MODEL=tcn
bilstm_v1.onnx + bilstm_scaler.joblib + bilstm_v1_card.json
patchtst_v1.onnx + patchtst_scaler.joblib + patchtst_v1_card.json

autoencoder_v1.onnx + autoencoder_scaler.joblib
autoencoder_v1_card.json         ← contém o mse_threshold (p99)

model_card.json                  ← Random Forest baseline
eval_*_cv.json                   ← Cross-validation temporal por arquitetura
```

### 12.5. Feature engineering

`MetroPTPreprocessor` (transformer sklearn) é compartilhado entre treino (`apps/ml/src/preprocessing.py`) e inferência (`apps/backend/src/services/preprocessing.py`). Cria **rolling features** sobre os 7 sensores analógicos: `std`, `ma` (moving average), `lag`, `roc` (rate of change), `min`, `max`. Como o backend não importa o pacote `ml/`, as duas implementações são mantidas **byte-for-byte equivalentes** — alterar uma exige espelhar na outra e re-treinar.

---

## 13. MLOps com MLflow

O serviço `mlflow` no `docker-compose.yml`:

- **Backend store**: PostgreSQL (`postgresql://user:password@db:5432/postgres`)
- **Artifact root**: volume Docker nomeado `mlflow_artifacts`
- **UI**: `http://localhost:5000`

### 13.1. Subindo apenas MLflow + DB

```powershell
docker compose up mlflow db -d
```

### 13.2. Treinos já logam automaticamente

Todos os scripts em `apps/ml/src/train_*.py` instanciam `MLFlowLogger` ou usam `mlflow.start_run()`. Configure o tracking URI:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
python src/train_mlp.py --max-epochs 50
```

### 13.3. Promovendo o melhor modelo para produção

```powershell
cd apps\ml
python src/promote_model.py `
    --tracking-uri http://localhost:5000 `
    --experiment   mlp_metropt3 `
    --metric       test_f1_class1 `
    --dest-dir     models/
```

Após copiar o artefato, faça o hot-swap sem reiniciar:

```powershell
curl -X PUT http://localhost/api/models/active `
  -H "X-Admin-Token: change-me-in-production" `
  -H "Content-Type: application/json" `
  -d '{"model_name": "mlp"}'
```

---

## 14. Testes

### 14.1. Backend (pytest + SQLite em memória)

```powershell
# Via Docker
docker compose exec api pytest tests/ -v

# Localmente (não exige Postgres — fixtures usam SQLite in-memory)
cd apps\backend
pytest tests/ -v
```

Cobertura: rotas (`predict`, `predictions`, `health`, `simulator`, `models`), SSE, WebSocket, `ModelRegistry`, `InferencePipelineService`, exceptions, rate limit.

### 14.2. ML (pytest)

```powershell
cd apps\ml
pytest tests/ -v
```

Cobertura: `MetroPTPreprocessor`, `MetroPTBalancer`, ingestão, treino RF/XGBoost/MLP/sequenciais, `promote_model`.

### 14.3. Frontend (Vitest)

```powershell
docker compose exec frontend pnpm test

# Ou localmente:
cd apps\frontend
pnpm test
```

### 14.4. Frontend E2E (Playwright)

```powershell
cd apps\frontend
pnpm exec playwright install   # primeira vez
pnpm e2e                       # headless
pnpm e2e:headed                # com browser visível
pnpm e2e:ui                    # modo UI interativo
```

Specs: `e2e/dashboard_flow.spec.ts`, `e2e/failure_alert.spec.ts`.

### 14.5. Load Test — SSE sob concorrência (RNF-37)

`locust_streaming.py` (raiz do repo) mede a **latência de conexão SSE**
(handshake + cabeçalhos de `GET /api/stream/sensors`) sob >=100 conexões
concorrentes reais e mantidas abertas — não uma rajada de requests
independentes. Reaproveita o parsing/validação de eventos já existente em
`locust_sse.py` (RNF-28).

**Por que "latência de conexão" e não "tempo até o 1º evento"?** O stream
emite 1 evento/s por design (RF-12) num broadcast compartilhado com clock
próprio — um cliente recém-conectado pode legitimamente esperar até ~1s
pelo próximo tick antes do primeiro dado, o que é uma característica de
produto (cadência de 1Hz), não de performance/infra. Medir isso como
"latência" tornaria RNF-37 estruturalmente impossível de passar
independente de qualquer otimização. Confirmado empiricamente com `curl`/
socket cru: a conexão HTTP (200 OK, stream aberto) responde em poucos ms;
só o evento de dado varia conforme o alinhamento com o tick do broadcast.

```powershell
pip install -r requirements-dev.txt   # locust, psutil

# Stack precisa estar de pé (nginx expõe a porta 80)
docker compose up -d db api nginx

# 100 conexões concorrentes por 75s — via Nginx, mesmo path do frontend
python -m locust -f locust_streaming.py --host http://localhost `
    --headless -u 100 -r 20 --run-time 75s --csv=loadtest_results/run1
```

Variáveis de ambiente opcionais (nenhuma obrigatória — o endpoint hoje é
público, sem auth):

| Variável                | Uso                                                        | Default              |
| ------------------------ | ----------------------------------------------------------- | --------------------- |
| `STREAMING_AUTH_TOKEN`  | Envia `Authorization: Bearer <token>` se o endpoint exigir  | (nenhum)             |
| `SSE_HOLD_SECONDS`      | Quanto tempo cada cliente mantém a conexão aberta           | `45`                  |
| `SSE_PATH`              | Path do endpoint SSE                                        | `/api/stream/sensors` |

O veredito RNF-37 (conexões, p50/p95/p99, PASS/FAIL) é impresso
automaticamente ao final da run headless.

### 14.6. Memory Leak Check (RNF-38)

`check_memory_leak.py` (raiz do repo) mede o **RSS real** do container
`api` (via `docker stats`, de fora — sem instalar nada na imagem) antes e
depois de ciclos de carga SSE, para diferenciar alocação normal (estabiliza
após aquecimento) de vazamento real (cresce ciclo a ciclo). Não usa
`tracemalloc` — ele só enxerga alocações Python e sub-mediria memória
nativa de asyncpg/ONNX Runtime/SQLAlchemy.

```powershell
docker compose up -d db api

python check_memory_leak.py --container api --url http://localhost/api `
    --clients 100 --cycles 3 --load-duration 30 --settle-seconds 10
```

Rodando fora do Docker (`uvicorn` local), use `--pid <PID>` em vez de
`--container` — medição via `psutil`, documentada no output como não
necessariamente representativa do ambiente containerizado de produção.

**Interpretação:** o script imprime RSS por ciclo (baseline frio, pós-
warmup — descartado do cálculo — e N ciclos de carga). O "crescimento"
reportado é sempre `final - pós-warmup` (não `final - baseline frio`),
porque a primeira carga sempre aquece pools/caches e infla a memória uma
única vez; o que importa para detectar leak é se os ciclos SEGUINTES
continuam crescendo.

### 14.7. Configuração de produção — workers & connection pool

Ambos configuráveis via `.env` (mesmo padrão do `ACTIVE_MODEL` — mudar
requer `docker compose up -d --force-recreate api`):

```bash
UVICORN_WORKERS=1     # apps/backend + docker-compose.yml (command do serviço api)
DB_POOL_SIZE=10       # src/core/config.py -> src/core/database.py
DB_MAX_OVERFLOW=20
```

**Workers — testado 1 vs 2, mantido 1** (evidência abaixo). Mais workers
significa mais PROCESSOS independentes, cada um carregando seu próprio
modelo ONNX + seu próprio `SensorStreamService`/broadcast loop + seu
próprio `InferencePipelineService` — não há compartilhamento de estado
entre workers nesta arquitetura. Com 1 worker já atendendo 100 conexões
SSE muito abaixo do SLA, 2 workers só duplicam custo:

| Config | RAM idle | RAM sob 100 SSE | p50 | p95 | p99 |
| ------ | -------: | ---------------: | --: | --: | --: |
| workers=1 (final) | 552 MB | 572 MB | 22 ms | **33 ms** | 51 ms |
| workers=2          | 1097 MB | 1120 MB | 16 ms | **48 ms** | 50 ms |

Workers=2 quase dobra a memória (2 modelos carregados) e piora a cauda
(p95 48ms vs 33ms — provável contenção de CPU entre 2 processos rodando
feature engineering + inferência ONNX a cada 1s, sem ganho de throughput
porque o event loop assíncrono de 1 worker já não é o gargalo para SSE).
**Conclusão baseada em dados: mais workers não ajudou neste workload.**

**Connection pool — mantido em 10/20 (default original), não testado
variações.** `GET /stream/sensors` não usa `Depends(get_db)` — nenhuma
conexão de banco é aberta por cliente SSE conectado; o único escritor no
Postgres é `InferencePipelineService.save_prediction`, uma vez por segundo,
independente de quantos clientes SSE estão conectados. O pool de conexões
é estruturalmente irrelevante para RNF-37/RNF-38 nesta arquitetura — testar
variações só para "ver o número mudar" violaria a instrução de não inflar
o pool sem justificativa real. Se o padrão de acesso ao banco mudar (ex.:
queries por cliente SSE), este ponto deve ser reavaliado.

### 14.8. Resultados — RNF-37 e RNF-38

**RNF-37** (config final: `workers=1`, `DB_POOL_SIZE=10`, ambiente:
Docker Desktop / Windows, via Nginx):

* Conexões simultâneas testadas: **200** (100 usuários × 2 ciclos de
  reconexão em 75s de run, pico observado de 101 conexões concorrentes
  nos logs do `api`)
* p50 = 22ms · **p95 = 33ms** · p99 = 51ms
* **RNF-37: PASS** (100+ conexões, p95 < 200ms — margem de ~6×)

**RNF-38** (3 ciclos de 100 clientes × 30s, assentamento de 10s, métrica:
RSS real do container `api` via `docker stats`):

* Baseline (pós-warmup): 564.5 MB
* Final (após 3 ciclos): 564.6 MB
* Crescimento: **+0.1 MB**
* **RNF-38: PASS** (limite 50 MB — sem sinal de leak)

Resultados brutos (CSV do Locust, logs) em `loadtest_results/` (gerado
localmente, não versionado — reproduza com os comandos acima).

**Limitação do ambiente:** testado em Docker Desktop para Windows numa
máquina de desenvolvimento única, não em infraestrutura de produção real
(sem rede entre hosts, sem múltiplas réplicas, sem load balancer externo).
Os números absolutos podem não se transferir diretamente para produção,
mas a CONCLUSÃO relativa (workers=1 > workers=2 para este workload; pool
de conexões é irrelevante para SSE) depende da arquitetura da aplicação,
não do hardware, e deve se manter.

---

## 15. Demo do Simulador

Sequência sugerida para a defesa do TCC:

```powershell
# 1. Verificar modo atual
curl http://localhost/api/simulator/mode

# 2. Ativar degradação progressiva (~5 min para drift completo)
curl -X PUT http://localhost/api/simulator/mode `
  -H "Content-Type: application/json" `
  -d '{"mode": "DEGRADATION"}'

# 3. Falha iminente — dispara alertas WebSocket
curl -X PUT http://localhost/api/simulator/mode `
  -H "Content-Type: application/json" `
  -d '{"mode": "FAILURE"}'

# 4. Voltar ao normal
curl -X PUT http://localhost/api/simulator/mode `
  -H "Content-Type: application/json" `
  -d '{"mode": "NORMAL"}'
```

O que esperar no dashboard:

| Modo | Comportamento esperado |
|---|---|
| `NORMAL` | Radar estável, polígono "Atual" próximo do "Ótimo"; score < 30 % |
| `DEGRADATION` | Polígono "Atual" diverge gradualmente; score sobe → zona ALERTA |
| `FAILURE` | Score > 65 %; toasts via WebSocket; badge LIVE âmbar/vermelha |

---

## 16. Troubleshooting

### `POST /predict/` retorna 503

**Causa**: artefato do modelo ativo não encontrado.

**Solução**: verifique `apps/ml/models/`. Treine o modelo correspondente ou troque para um disponível:

```powershell
curl -X PUT http://localhost/api/models/active `
  -H "X-Admin-Token: change-me-in-production" `
  -d '{"model_name": "random_forest"}'
```

### `FileNotFoundError: Simulator parquet not found`

**Causa**: o parquet do MetroPT-3 não foi gerado.

**Solução**:

```powershell
cd apps\ml
python -m src.ingest_metropt
```

O backend lê o parquet de `/ml/data/processed/metropt3.parquet` (montagem do volume). Garanta que `apps/ml/data/processed/metropt3.parquet` exista no host.

### Autoencoder retorna score sempre próximo de 0 ou 1

**Causa**: `mse_threshold` em `autoencoder_v1_card.json` desatualizado em relação ao `.onnx`.

**Solução**:

```powershell
cd apps\ml
python src/train_autoencoder.py
```

### `alembic upgrade head` falha com `InvalidPasswordError` ou DNS error em `db`

**Causa**: `apps/backend/.env` aponta para hostname `db` (rede Docker) ao rodar fora do container.

**Solução**: use `localhost` ao rodar Alembic local:

```dotenv
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/tcc_db
```

### `ACTIVE_MODEL` ignorado ao subir o contêiner

**Causa**: variável não exportada antes do build, ou cache de imagem.

**Solução**:

```powershell
docker compose up --build api -d
```

### PowerShell bloqueia ativação do venv

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Frontend devolve 504 Gateway Timeout em rotas novas

**Causa**: Turbopack compila rotas sob demanda no primeiro acesso (>60 s em árvores pesadas).

**Solução**: já mitigado no `infra/nginx/nginx.conf` (`proxy_read_timeout 180s` no bloco `/`). Aguarde o primeiro compile e atualize a página.

### Conexão SSE/WS cai após 60 s

**Causa**: timeout do reverse proxy.

**Solução**: já mitigado no Nginx — `proxy_read_timeout 3600s` para `/api/stream/` e `/ws/`. Troubleshooting completo de SSE/WS (handshake, buffering, headers, CORS): §4.3 e §4.4.

### Frontend lança `NEXT_PUBLIC_API_URL não está definida`

**Causa**: `apps/frontend/.env.local` não existe — este arquivo é gitignorado e **não** é o mesmo que o `.env` da raiz (ver §7).

**Solução**:

```bash
cp apps/frontend/.env.local.example apps/frontend/.env.local
docker compose up -d --build frontend   # ou `pnpm dev` se rodando fora do Docker
```

### Editei um componente do frontend e nada mudou na tela

**Causa**: Turbopack (Next.js dev server) às vezes não detecta mudanças de arquivo através do bind mount do Docker Desktop no Windows — o HMR mostra `[HMR] connected`, mas continua servindo a versão antiga, sem erro. `WATCHPACK_POLLING=true` (já definido no `docker-compose.yml`) não resolve porque essa variável é específica do watcher do webpack, não do Turbopack.

**Solução**: force um reload completo do processo:

```bash
docker compose restart frontend
```

### Docker Desktop com `out of memory` no `api`

**Causa**: limite de 3 GB definido + carga do parquet + ONNX session de modelos sequenciais.

**Solução**: aumente o limite em `docker-compose.yml` (`deploy.resources.limits.memory`) ou suba o Docker Desktop para 8+ GB em **Settings → Resources**.

### `index_manuals.py` loga `Failed to send telemetry event ... capture() takes 1 positional argument but 3 were given`

**Causa**: incompatibilidade entre a versão de `chromadb==0.5.23` e a API do `posthog` (dependência transitiva) instalada — a telemetria anônima do ChromaDB tenta chamar `capture()` com uma assinatura que essa versão do `posthog` não aceita mais.

**Impacto**: nenhum. É só logging de uma tentativa de telemetria (que já falha silenciosamente dentro do próprio ChromaDB) — indexação, busca e persistência funcionam normalmente (validado em `docker compose exec mcp-server python index_manuals.py`, ver §4.7). Se incomodar, `CHROMA_ANONYMIZED_TELEMETRY=false` no ambiente do container desativa a tentativa.

### Known Issues (não corrigidos nesta task — fora do escopo)

Divergências/observações reais encontradas na auditoria (RNF-41/RNF-42), documentadas em vez de corrigidas porque esta é uma task de documentação, não de mudança de comportamento:

- **`proxy_buffering` não está explicitamente desativado no bloco `/api/stream/`** de `infra/nginx/nginx.conf`. Funciona porque `chunked_transfer_encoding off` + `X-Accel-Buffering: no` já bastam nesta versão do Nginx, mas a diretiva "canônica" para SSE (`proxy_buffering off;`) não está presente. Se o comportamento de streaming mudar em uma atualização do Nginx, este é o primeiro lugar a checar.
- **`/api/` (bloco REST genérico) herda `proxy_read_timeout 3600s`**, o mesmo valor usado para SSE/WS — nenhuma rota REST precisa de uma conexão de 1 hora; não é um bug funcional, mas é uma configuração mais permissiva do que o necessário para esse bloco.
- **`ALLOWED_ORIGINS` inclui `http://localhost:8000`** no `.env` atual — não há nenhum serviço exposto diretamente nessa porta neste `docker-compose.yml` (a API só é alcançável via Nginx em `:80`), então essa origem parece vestigial de uma configuração anterior sem proxy.
- **Imagem do `mcp-server` cresceu para ~10 GB (RF-20/RNF-44)**: `pip install sentence-transformers` puxa `torch` da PyPI padrão, que inclui dependências CUDA (`nvidia-*`) mesmo num container CPU-only sem GPU. Funciona (PyTorch cai para CPU automaticamente — `Use pytorch device_name: cpu` no log), mas a imagem fica bem maior que o necessário. Otimização futura: instalar `torch` a partir do índice CPU-only da PyTorch (`--index-url https://download.pytorch.org/whl/cpu`) no `Dockerfile`.
- **PDF sem texto extraível é reprocessado a cada execução** (RF-20): como nenhum `file_hash` é gravado para um arquivo que gerou zero chunks, ele nunca é marcado como "já visto" — cada run tenta extrair de novo (custo: só a extração via pypdf, nenhum embedding é gerado). Decisão deliberada: se o PDF ganhar texto extraível depois (ex.: substituído por uma versão não-escaneada), a próxima execução já pega automaticamente, sem precisar de nenhuma ação manual.
- **Trocar `EMBEDDING_MODEL` não invalida embeddings antigos automaticamente**: cada chunk registra `embedding_model` nos metadados, mas a lógica de skip/reindex compara só `file_hash` — se você trocar de modelo sem tocar nos PDFs, os embeddings antigos (gerados pelo modelo anterior) continuam no ChromaDB, agora "misturados" com um `embedding_model` diferente do `EMBEDDING_MODEL` atual. Fora do escopo desta task (RF-20 pede consistência do modelo *dentro* de uma mesma indexação, não migração entre modelos); se for trocar de modelo, apague `data/chroma/` antes de reindexar.
- **`tests/test_simulator.py` (backend) falha na coleta** com `IndexError: 3` em `Path(__file__).resolve().parents[3]` — pré-existente, não introduzido nem corrigido pela RF-22 (confirmado via `git stash` antes desta task). Contorno usado para rodar a suíte: `pytest --ignore=tests/test_simulator.py`.
- **Llama 3.2 3B nem sempre segue a instrução de branch "Limitações" do System Prompt (RF-22)**: quando o MCP não retorna nenhum trecho relevante, o modelo às vezes mantém a estrutura completa do template (preenchendo "Não especificado nos trechos recuperados" nas seções, o que é seguro) em vez de trocar para a seção alternativa "## Limitações" exatamente como instruído — e uma vez chegou a citar a própria query do usuário como se fosse uma referência de manual. Validado que o modelo **nunca inventa** procedimento/peça/valor técnico nesses casos (a regra de segurança central se mantém), mas o *formato* exato da branch não é 100% determinístico com um modelo de 3B rodando localmente — limitação conhecida de modelos pequenos, não um bug de código.
- **`apps/backend/requirements.txt` puxa `nvidia-nccl-cu12` (~340 MB) como dependência transitiva de `mcp` (RF-22)**, mesmo o backend sendo só um *cliente* MCP (nunca roda modelos de ML locais via essa lib). Não foi investigado a fundo qual extra do `mcp`/`opentelemetry` trafega isso — funciona normalmente (a lib nunca é importada em runtime), mas infla a imagem sem necessidade real. Mesma classe de problema do torch/CUDA no mcp-server (RF-20).
- **`anyio`/`idna`/`typing_extensions` estavam com pin exato (`==`) em `apps/backend/requirements.txt`**, congelados de um `pip freeze` anterior, e conflitavam com as versões mínimas exigidas por `mcp` (RF-22). Resolvido removendo o pin exato dessas três dependências transitivas de baixo nível (deixando o pip resolver a versão compatível) — os pins de bibliotecas de topo (`fastapi`, `httpx`, `pydantic`, etc.) foram mantidos intactos.
- **Rate limit de alertas críticos (RF-24) usa Postgres, não Redis** — decisão deliberada (nenhum cache existia no projeto; ver §4.10 para a análise completa). Funciona corretamente no volume atual (1 linha por ativo simulado), mas expiração é verificada na leitura (não expurgada em background) e não há índice dedicado em `expires_at` — não escalaria da mesma forma para milhares de equipamentos sem ajustes.
- **Telegram (RF-24) não foi testado contra a API pública real** — nenhuma credencial real (`TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`) está disponível neste ambiente de desenvolvimento. Toda a validação de HTTP usa mock; a lógica de negócio (threshold, rate limit, concorrência, orquestração) foi validada com infraestrutura real (Postgres real, classes reais), só a chamada de rede ao Telegram em si é simulada.
- **E-mail via Resend (RF-25) não foi testado contra a API real** — mesma limitação do Telegram acima, sem `RESEND_API_KEY` real neste ambiente. `EmailNotificationAdapter` é o primeiro canal de e-mail do projeto (nenhuma integração existia antes desta task).
- **Configuração de alertas (RF-25) é global, não por usuário** — o projeto não possui sistema de usuários; ver §4.11 para a decisão completa.
- **Fila de notificações (RNF-50/51) sem retry automático e sem result backend** — decisões deliberadas de simplicidade; ver §4.12 para o raciocínio completo.

---

## 17. Autores & Licença

### Autores

| Nome |
|---|
| Lucas de Moraes Silveira |
| Raphael Nobuyuki Haga Okuyama |
| Ronaldo Simeone Antonio |

### Licença

Projeto desenvolvido **exclusivamente para fins acadêmicos** (TCC — Indústria 4.0). O dataset MetroPT-3 é de uso público sob os termos da UCI Machine Learning Repository.
