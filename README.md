# PredictIQ — Sistema de Manutenção Preditiva Industrial

> **TCC · Indústria 4.0 · v3.0** — Detecção de falhas em tempo real para o compressor MetroPT-3, com streaming de dados reais, pipeline de Deep Learning sequencial, Autoencoder não-supervisionado, dashboard interativo e assistente inteligente via LLM local + MCP.

---

## Visão Geral das Tecnologias

| Camada | Tecnologia | Responsabilidade |
|---|---|---|
| **Frontend** | Next.js 15, TypeScript strict, Tailwind CSS v4, shadcn/ui, Recharts | Dashboard em tempo real, SSE 1 Hz, alertas via WebSocket, radar chart com normalização absoluta |
| **Backend** | FastAPI, SQLAlchemy 2 (async), Pydantic v2, Alembic, structlog | API REST, persistência, injeção de dependência, Model Registry com hot-swap atômico |
| **Banco de dados** | PostgreSQL 15 (Docker) | Histórico de predições |
| **Machine Learning** | Scikit-learn, XGBoost, PyTorch Lightning, ONNX Runtime, Optuna, MLflow | RF, XGBoost, MLP, TCN, BiLSTM, PatchTST e Conv1D Autoencoder servidos via ONNX em produção |
| **MLOps** | MLflow 2.x (Docker), PostgreSQL backend store, promoção automatizada | Rastreamento de experimentos, registro de artefatos e promoção do melhor modelo |
| **Motor de Simulação** | NumPy, PyArrow, Parquet | Streaming sequencial do dataset oficial MetroPT-3 a 1 Hz com modos NORMAL / DEGRADATION / FAILURE |
| **Infraestrutura** | Docker Desktop, Docker Compose, Nginx | Orquestração unificada, reverse proxy com suporte a SSE/WebSocket e mapeamento de volumes ML |
| **IA Generativa** | Ollama (Llama 3.2 3B), ChromaDB, MCP | Assistente RAG de sugestões de reparo _(próxima fase)_ |

---

## Arquitetura do Monorepo

```
projeto-tcc/
├── apps/
│   ├── backend/                        # API FastAPI — Clean Architecture
│   │   ├── alembic/                    # Migrations do banco de dados
│   │   │   └── versions/
│   │   │       └── 0001_create_predictions_table.py
│   │   ├── src/
│   │   │   ├── core/                   # config, database, exceptions, logging, rate limit
│   │   │   ├── models/                 # ORM SQLAlchemy (Prediction)
│   │   │   ├── routers/                # I/O, REST, SSE, WebSocket, injeção de dependência
│   │   │   ├── schemas/                # DTOs Pydantic v2
│   │   │   └── services/
│   │   │       ├── model_registry.py   # Hot-swap atômico (RNF-25)
│   │   │       ├── model_service.py    # Adapter pattern — interface unificada para todos os modelos
│   │   │       ├── inference_pipeline.py  # Background loop: SSE → ML → DB → WebSocket
│   │   │       ├── feature_buffer.py   # Deque thread-safe para janelas temporais
│   │   │       ├── preprocessing.py    # MetroPTPreprocessor (std/ma/lag/roc/min/max)
│   │   │       ├── simulator.py        # Data Streamer real do MetroPT-3 (.parquet)
│   │   │       ├── mlp_adapter.py      # ONNX MLP adapter
│   │   │       ├── onnx_tree_adapter.py   # ONNX Random Forest / XGBoost V2 adapter
│   │   │       ├── onnx_sequence_adapter.py  # ONNX TCN / BiLSTM / PatchTST adapter
│   │   │       └── onnx_autoencoder_adapter.py  # Conv1D Autoencoder adapter (MSE → sigmoid)
│   │   ├── tests/
│   │   ├── alembic.ini
│   │   └── requirements.txt
│   │
│   ├── frontend/                       # Dashboard Next.js
│   │   ├── app/                        # App Router (layout, page, globals.css)
│   │   ├── components/
│   │   │   └── dashboard/
│   │   │       ├── FleetDashboard.tsx  # Orquestrador principal do dashboard
│   │   │       ├── AssetRadarChart.tsx # Radar com normalização absoluta + modo LIVE
│   │   │       ├── AssetTable.tsx      # Tabela de ativos com risco em tempo real
│   │   │       └── AssetEfficiencyChart.tsx
│   │   ├── hooks/                      # use-sensor-data, use-alert-websocket, use-sse
│   │   ├── lib/                        # api-client, utils
│   │   └── __tests__/
│   │
│   └── ml/                             # Pipelines de Machine Learning
│       ├── data/raw/                   # Dataset CSV (ignorado pelo git)
│       ├── data/processed/             # Parquet pré-processado (ignorado pelo git)
│       ├── models/                     # Artefatos treinados (.joblib, .onnx, *_card.json)
│       ├── src/
│       │   ├── ingest_metropt.py       # CSV → Parquet
│       │   ├── preprocessing.py        # MetroPTPreprocessor (idêntico ao backend)
│       │   ├── train_random_forest.py
│       │   ├── train_xgboost.py
│       │   ├── train_mlp.py
│       │   ├── train_autoencoder.py    # Conv1D Autoencoder não-supervisionado
│       │   ├── models/autoencoder.py   # Arquitetura Conv1D Lightning Module
│       │   └── datamodule_unsupervised.py  # DataModule treina apenas em janelas saudáveis
│       └── tests/
│
├── infra/
│   └── nginx/
│       └── nginx.conf                  # Reverse proxy: SSE/WS timeout 3600s
│
├── docs/
│   └── model_comparison.*              # Comparação e validação dos modelos
│
├── docker-compose.yml
└── README.md
```

---

## Arquitetura de Inteligência Artificial

### Pipeline de Inferência em Tempo Real

O sistema implementa um pipeline de inferência contínuo que conecta o motor de streaming ao modelo ML sem intervenção manual:

```
MetroPT-3 Parquet
      ↓  (1 Hz, streaming sequencial)
SensorSimulator  →  SensorStreamService  →  SSE /api/stream/sensors  →  Frontend
                              ↓
                  InferencePipelineService (background asyncio.Task)
                              ↓
                      SensorBuffer (deque thread-safe, window=30)
                              ↓
                  MetroPTPreprocessor (std / ma / lag / roc / min / max)
                              ↓
                    ModelService.predict_from_features()
                              ↓
                  PostgreSQL (RF-09)  +  AlertService  →  WebSocket /ws/alerts
```

> **Warmup:** Nos primeiros 15 ticks o buffer ainda não tem dados suficientes para as features rolling. O sistema faz fallback para inferência stateless (`predict(request)`) e não bloqueia o startup.

---

### Modelos Disponíveis

O sistema suporta 8 modelos com hot-swap atômico via `PUT /models/active` — nenhum reinicia o servidor:

| `model_name` | Tipo | Artefato | Observação |
|---|---|---|---|
| `random_forest` | Supervisionado (clássico) | `.joblib` | Baseline do projeto |
| `xgboost` | Supervisionado (clássico) | `.joblib` | Otimizado com Optuna |
| `mlp` | Supervisionado (DL) | `.onnx` | PyTorch Lightning → ONNX |
| `random_forest_v2` | Supervisionado (clássico) | `.onnx` | RF exportado para ONNX Runtime |
| `xgboost_v2` | Supervisionado (clássico) | `.onnx` | XGBoost exportado para ONNX Runtime |
| `tcn` | Sequencial (DL) | `.onnx` | Temporal Convolutional Network |
| `bilstm` | Sequencial (DL) | `.onnx` | Bidirectional LSTM |
| `patchtst` | Sequencial (DL) | `.onnx` | PatchTST Transformer |
| `autoencoder` | **Não-supervisionado (DL)** | `.onnx` | Conv1D — detecção por erro de reconstrução |

---

### Autoencoder Conv1D — Detecção Não-Supervisionada

O Autoencoder é o modelo mais avançado do sistema. Não requer exemplos de falha no treino — aprende apenas o padrão de operação saudável e deteta anomalias como desvios da reconstrução esperada.

**Arquitetura:**

```
Entrada: (B, T=60, C=12)           ← janela de 60 ticks × 12 sensores

Encoder:
  Conv1D(12→32, k=4, s=2) + BN + GELU   → (B, 30, 32)
  Conv1D(32→64, k=4, s=2) + BN + GELU   → (B, 15, 64)
  Conv1D(64→128, k=4, s=2) + BN + GELU  → (B,  8, 128)

Decoder:
  ConvTranspose1D(128→64, k=4, s=2)     → (B, 15, 64)
  ConvTranspose1D( 64→32, k=4, s=2)     → (B, 30, 32)
  ConvTranspose1D( 32→12, k=4, s=2)     → (B, 60, 12)

Saída: (B, T=60, C=12)                  ← reconstrução da janela de entrada
```

**Score de anomalia:**

O erro de reconstrução MSE é normalizado via sigmoid com midpoint exato no threshold de treino:

```
score = sigmoid( (mse − mse_threshold) / (mse_threshold / 3) )
```

- `mse = mse_threshold` → `score = 0.50` (limiar neutro)
- `mse = 2 × mse_threshold` → `score ≈ 0.95` (anomalia clara)
- `mse ≪ mse_threshold` → `score ≈ 0.02` (operação saudável)

O `mse_threshold` é o **percentil 99** do MSE calculado sobre as janelas saudáveis do conjunto de validação e é persistido no `autoencoder_v1_card.json`.

---

### Motor de Simulação — Streaming Real MetroPT-3

O simulador foi completamente reescrito. A geração de dados sintéticos com `np.random` foi substituída por um **Data Streamer** que consome o dataset oficial MetroPT-3 diretamente do ficheiro `.parquet` em memória.

**Modos disponíveis:**

| Modo | Comportamento |
|---|---|
| `NORMAL` | Streaming sequencial de linhas do conjunto **saudável** (label=0) do MetroPT-3 |
| `FAILURE` | Streaming sequencial de linhas dos **4 eventos de falha** reais do dataset |
| `DEGRADATION` | Interpolação vetorizada (`lerp`) entre linha normal e linha de falha ao longo de 300 ciclos |

**Janelas de falha reais utilizadas:**

```
2020-04-18 00:00 → 2020-04-18 23:59
2020-05-29 23:30 → 2020-05-30 06:00
2020-06-05 10:00 → 2020-06-07 14:30
2020-07-15 14:30 → 2020-07-15 19:00
```

**Fórmula de degradação (drift vetorizado):**

```python
drift   = min(step / 300, 1.0)          # 0.0 → 1.0 ao longo de 300 ticks
blended = normal_row + drift * (failure_row - normal_row)
```

> O índice de streaming é circular (`idx = (idx + 1) % len(array)`), garantindo operação contínua independentemente do tamanho do dataset.

---

## Dashboard Frontend — Modo LIVE e Radar de Perfil Operacional

### Radar Chart com Normalização Absoluta

O `AssetRadarChart` foi refatorado para utilizar **normalização absoluta** em vez de min-max:

```
score_normalizado = min(100, max(0, (valor_raw / teto_físico) × 100))
```

Isso permite que sensores de grandezas físicas diferentes (pressão em bar, temperatura em °C, corrente em A) sejam comparados na mesma forma geométrica sem distorção de escala:

| Sensor | Teto físico | Referência saudável (% de capacidade) |
|---|---|---|
| TP2 | 12 bar | 84.2 % |
| TP3 | 12 bar | 84.2 % |
| H1 | 12 bar | 70.8 % |
| Motor_current | 10 A | 38.0 % |
| Oil_temperature | 80 °C | 80.0 % |
| Reservoirs | 12 bar | 58.3 % |

O polígono verde ("Ótimo") representa as medianas de operação saudável do MetroPT-3. O polígono azul ("Atual") é recalculado via `useMemo([sensorData])` a cada tick de 1 Hz.

### Modo LIVE — APU-Trem-042

Quando o ativo `APU-Trem-042` é selecionado na tabela da frota, o dashboard entra em **modo LIVE**:

- 🟢 **Badge LIVE** com ponto pulsante (`animate-pulse`) — confirma que os dados são em tempo real
- **Badge de Anomaly Score** codificada por cor: `NORMAL` (verde) / `ALERTA` (âmbar) / `CRÍTICO` (vermelho), com percentagem calculada pelo modelo ativo
- **Strip de valores raw instantâneos**: TP2 (bar), Temperatura (°C), Corrente Motor (A)
- O radar é atualizado frame a frame à medida que o SSE entrega novas leituras

Para ativos estáticos, o dashboard exibe a badge `IDLE` e o polígono usa os valores mockados da tabela de frota.

---

## Pré-requisitos

Antes de começar, certifique-se de ter instalado:

| Ferramenta | Versão mínima | Como instalar |
|---|---|---|
| **Docker Desktop** | 4.x | [docker.com](https://www.docker.com/products/docker-desktop/) |
| **Python** | 3.12+ | Necessário apenas para Alembic e ML localmente |

Verificação rápida:

```bash
docker --version        # Docker version 24.x.x ou superior
docker compose version  # v2.x.x
```

---

## Configuração de Variáveis de Ambiente

Crie o arquivo `.env` na **raiz do projeto** antes de subir os serviços:

```dotenv
# projeto-tcc/.env

# ── PostgreSQL ────────────────────────────────────────────────────────────
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_DB=tcc_db

# ── Backend (dentro do Docker — hostname "db" = nome do serviço na rede) ─
DATABASE_URL=postgresql+asyncpg://user:password@db:5432/tcc_db

# ── Modelo ativo na inicialização ────────────────────────────────────────
# Valores válidos: random_forest | xgboost | mlp | random_forest_v2 |
#                  xgboost_v2 | tcn | bilstm | patchtst | autoencoder
ACTIVE_MODEL=random_forest

# ── Ollama (LLM local) ────────────────────────────────────────────────────
OLLAMA_BASE_URL=http://host.docker.internal:11434

# ── Frontend ──────────────────────────────────────────────────────────────
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Subindo o Sistema Completo

O projeto é orquestrado **exclusivamente via Docker Compose**. Um único comando sobe todos os serviços:

```bash
# Na raiz do projeto — o arquivo .env deve existir
docker compose up --build
```

Para rodar em background:

```bash
docker compose up --build -d
```

| Serviço | URL | Descrição |
|---|---|---|
| **Frontend** | `http://localhost:3000` | Dashboard Next.js |
| **Backend / API** | `http://localhost:8000` | API FastAPI (via Nginx) |
| **Swagger UI** | `http://localhost:8000/docs` | Documentação interativa da API |
| **MLflow** | `http://localhost:5000` | UI de experimentos e artefatos |
| **PostgreSQL** | `localhost:5432` | Banco de dados (interno) |

### Parar os serviços

```bash
# Para e remove os contêineres (dados persistidos no volume Docker)
docker compose down

# Para, remove contêineres E apaga todos os dados (reset completo)
docker compose down -v
```

---

## Rodando as Migrations (Alembic)

As migrations criam e versionam o schema do PostgreSQL. A tabela `predictions` é criada pela migration `0001`.

> **Importante:** As migrations **não são executadas automaticamente** ao subir os serviços. Execute-as manualmente após o `docker compose up`.

### Opção 1 — Via docker exec (recomendada)

```bash
docker compose exec api alembic upgrade head
```

Saída esperada:

```
INFO  [alembic.runtime.migration] Context impl PostgreSQLImpl.
INFO  [alembic.runtime.migration] Running upgrade  -> 0001, create predictions table — RF-09
```

### Opção 2 — Localmente (requer Python 3.12+)

Com o contêiner do banco rodando:

```bash
docker compose up db -d
```

Crie `apps/backend/.env` com `localhost`:

```dotenv
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/tcc_db
ACTIVE_MODEL=random_forest
```

Instale e execute:

```bash
cd apps/backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
alembic upgrade head
```

### Referência de comandos Alembic

```bash
alembic current                              # Estado atual no banco
alembic history --verbose                    # Histórico completo
alembic downgrade -1                         # Reverter última migration
alembic revision --autogenerate -m "desc"    # Gerar nova migration
```

---

## Endpoints da API

| Endpoint | Método | Auth | Descrição |
|---|---|---|---|
| `/health` | `GET` | — | Probe de liveness e conectividade com o banco |
| `/predict/` | `POST` | — | Inferência + persiste o resultado — 100 req/min por IP |
| `/api/v1/predictions` | `GET` | — | Histórico paginado (`?page=1&size=20`) |
| `/api/stream/sensors` | `GET` | — | Stream SSE de leituras a 1 Hz |
| `/ws/alerts` | `WebSocket` | — | Canal push de alertas críticos (probability > 0.70) |
| `/api/simulator/mode` | `GET/PUT` | — | Consulta/altera o modo do simulador |
| `/models` | `GET` | `X-Admin-Token` | Lista modelos registrados |
| `/models/active` | `PUT` | `X-Admin-Token` | Hot-swap do modelo ativo sem reiniciar o servidor |
| `/docs` | `GET` | — | Swagger UI interativo |

**Destaques de infraestrutura:**

- **Atomic Hot-Swap:** o `ModelRegistry` usa um protocolo de duas fases (carregamento fora do lock + swap atômico dentro do lock) para garantir zero downtime durante a troca de modelos.
- **ONNX Runtime:** todos os modelos DL são exportados para ONNX (opset 17, eixo de batch dinâmico) e servidos via `onnxruntime.InferenceSession` — sem dependência de PyTorch em produção.
- **Rate limiting** via `slowapi`: 100 req/min por IP em `POST /predict/`.
- **Logging estruturado** via `structlog` (JSON em produção).

---

## Ambiente de Machine Learning

O diretório `apps/ml/` possui seu **próprio ambiente virtual isolado**.

### Configurando o ambiente local

```bash
cd apps/ml
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # Windows PowerShell
pip install -r requirements.txt
```

### Baixando o dataset

Faça o download do **MetroPT-3 Air Compressor Dataset** na [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/791/metropt-3+dataset) e salve em:

```
apps/ml/data/raw/MetroPT3(AirCompressor).csv
```

### Pipeline de treinamento

Execute a partir de `apps/ml/` com o venv ativo:

```bash
# Passo 1 — Ingestão e geração do Parquet
python -m src.ingest_metropt

# Passo 2a — Random Forest (baseline)
python -m src.train_random_forest

# Passo 2b — XGBoost com Optuna
python src/train_xgboost.py --n-trials 100
# Use --n-trials 5 para smoke run rápido (~2 min)

# Passo 2c — MLP com PyTorch Lightning
python src/train_mlp.py --max-epochs 50
# Use --max-epochs 10 para smoke run rápido (~5 min)

# Passo 2d — Autoencoder Conv1D (não-supervisionado)
python src/train_autoencoder.py --max-epochs 50
# Treina apenas em janelas saudáveis; calcula MSE threshold no percentil 99
```

Artefatos gerados em `apps/ml/models/`:

```
random_forest_final.joblib      ← ACTIVE_MODEL=random_forest
xgboost_v1.joblib               ← ACTIVE_MODEL=xgboost
mlp_v1.onnx + mlp_scaler.joblib ← ACTIVE_MODEL=mlp
random_forest_v2.onnx           ← ACTIVE_MODEL=random_forest_v2
xgboost_v2.onnx                 ← ACTIVE_MODEL=xgboost_v2
autoencoder_v1.onnx             ← ACTIVE_MODEL=autoencoder
autoencoder_scaler.joblib       ← StandardScaler do Autoencoder (fitado só em saudável)
*_card.json                     ← métricas, threshold e metadados de cada modelo
```

### Rodando os testes do ML

```bash
pytest tests/ -v
```

---

## MLOps — MLflow + Promoção de Modelos

O serviço `mlflow` roda como contêiner dedicado. O PostgreSQL atua como backend store; os artefatos ficam em volume Docker nomeado (`mlflow_artifacts`).

```bash
docker compose up mlflow db -d
# Acesse: http://localhost:5000
```

### Promovendo um modelo para produção

```bash
# Com o venv do ML ativo, na pasta apps/ml
python src/promote_model.py \
    --tracking-uri http://localhost:5000 \
    --experiment   mlp_metropt3 \
    --metric       test_f1_class1 \
    --dest-dir     models/
```

Após a promoção, use o hot-swap para carregar o novo modelo sem reiniciar:

```bash
curl -X PUT http://localhost:8000/models/active \
  -H "X-Admin-Token: $ADMIN_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "autoencoder"}'
```

---

## Simulador de Sensores — Demo

O simulador consome o MetroPT-3 real. O fluxo de demonstração sugerido para a defesa:

```bash
# 1. Verificar modo atual
curl http://localhost/api/simulator/mode

# 2. Ativar degradação progressiva (~5 min para drift completo)
curl -X PUT http://localhost/api/simulator/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "DEGRADATION"}'

# 3. Ativar falha iminente — dispara alertas WebSocket
curl -X PUT http://localhost/api/simulator/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "FAILURE"}'

# 4. Retornar ao normal
curl -X PUT http://localhost/api/simulator/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "NORMAL"}'
```

**O que observar no dashboard:**

1. `NORMAL` — radar estável próximo ao polígono "Ótimo"; Anomaly Score < 30%.
2. `DEGRADATION` — polígono "Atual" deriva gradualmente; score sobe para a zona ALERTA.
3. `FAILURE` — badge LIVE fica âmbar/vermelha; toasts de alerta via WebSocket; score > 65%.

---

## Rodando os Testes

### Backend

```bash
# Via Docker
docker compose exec api pytest tests/ -v

# Localmente (não requer PostgreSQL — usa SQLite em memória)
pytest tests/ -v
```

### Frontend

```bash
# Via Docker
docker compose exec frontend pnpm test

# Localmente
pnpm test
```

---

## Troubleshooting

### `alembic upgrade head` falha com `InvalidPasswordError`

**Causa:** `DATABASE_URL` usa `@db:5432`. O hostname `db` só resolve dentro do Docker.

**Solução:** Verifique `apps/backend/.env` e confirme que usa `localhost`:

```dotenv
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/tcc_db
```

---

### `POST /predict/` retorna HTTP 503

**Causa:** Artefato do modelo ativo não encontrado no caminho configurado.

**Solução:** Execute o treinamento correspondente ou troque para um modelo disponível:

```bash
docker compose exec api curl -X PUT http://localhost:8000/models/active \
  -H "X-Admin-Token: $ADMIN_API_TOKEN" \
  -d '{"model_name": "random_forest"}'
```

---

### Autoencoder retorna score sempre próximo de 0 ou 1

**Causa:** O `mse_threshold` no `autoencoder_v1_card.json` está desatualizado em relação ao modelo treinado.

**Solução:** Re-execute o treinamento para regenerar o card com o threshold correto:

```bash
cd apps/ml && python src/train_autoencoder.py
```

---

### `ACTIVE_MODEL=xgboost` mas a API usa Random Forest

**Causa:** Variável de ambiente não lida pelo processo.

**Solução:** Adicione ao `.env` da raiz e recrie o contêiner:

```bash
docker compose up --build api -d
```

---

### PowerShell bloqueia ativação do venv

**Causa:** Política de execução do Windows impede scripts `.ps1`.

**Solução:** Execute como Administrador:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Autores

| Nome |
|---|
| Lucas de Moraes Silveira |
| Raphael Nobuyuki Haga Okuyama |
| Ronaldo Simeone Antonio |

---

## Licença

Este projeto é desenvolvido exclusivamente para fins acadêmicos.
