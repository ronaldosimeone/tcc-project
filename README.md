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

Crie o arquivo `.env` na **raiz do projeto** antes de subir os serviços. Esse arquivo é consumido por **todos** os contêineres (`api`, `frontend`, `ml`, `db`, `mlflow`) via `env_file: .env`.

```dotenv
# projeto-tcc/.env

# ── PostgreSQL ────────────────────────────────────────────────────────────
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_DB=tcc_db

# ── Backend (hostname "db" = nome do serviço na rede Docker) ──────────────
DATABASE_URL=postgresql+asyncpg://user:password@db:5432/tcc_db

# ── Modelo ativo na inicialização ────────────────────────────────────────
# Valores válidos: random_forest | xgboost | mlp | random_forest_v2 |
#                  xgboost_v2    | tcn     | bilstm | patchtst | autoencoder
ACTIVE_MODEL=random_forest

# ── Token de admin (necessário para PUT /models/active e GET /models) ────
ADMIN_API_TOKEN=change-me-in-production

# ── Frontend ──────────────────────────────────────────────────────────────
NEXT_PUBLIC_API_URL=http://localhost

# ── Ollama (próxima fase — assistente RAG local) ─────────────────────────
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

> **Atenção**: o backend é configurado para ler o parquet do simulador em `/ml/data/processed/metropt3.parquet`, conforme `SIMULATOR_PARQUET_PATH` no `docker-compose.yml`. Você precisa **gerar o parquet antes da primeira execução** (ver §12).

---

## 8. Como Rodar (Docker Compose)

O projeto é orquestrado **exclusivamente via Docker Compose**. Todos os 6 serviços sobem com um único comando.

### 8.1. Subindo tudo

```powershell
# Na raiz do projeto (.env já deve existir)
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
| `GET` | `/docs` · `/redoc` · `/openapi.json` | — | Documentação |

### 10.2. Streaming

| Tipo | Caminho | Descrição |
|---|---|---|
| **SSE** | `GET /api/stream/sensors` | `text/event-stream` — emite `sensor_reading` 1× por segundo com timestamp + 12 sensores |
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

**Solução**: já mitigado no Nginx — `proxy_read_timeout 3600s` para `/api/stream/` e `/ws/`.

### Docker Desktop com `out of memory` no `api`

**Causa**: limite de 3 GB definido + carga do parquet + ONNX session de modelos sequenciais.

**Solução**: aumente o limite em `docker-compose.yml` (`deploy.resources.limits.memory`) ou suba o Docker Desktop para 8+ GB em **Settings → Resources**.

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
