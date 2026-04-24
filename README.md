# PredictIQ — Sistema de Manutenção Preditiva Industrial

> **TCC · Indústria 4.0** — Detecção de falhas em tempo real para o compressor MetroPT-3, com dashboard interativo, histórico persistido e assistente inteligente via LLM local + MCP.

---

## Visão Geral das Tecnologias

| Camada               | Tecnologia                                                             | Responsabilidade                                                                |
| -------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Frontend**         | Next.js 16, TypeScript strict, Tailwind CSS v4, shadcn/ui, Recharts    | Dashboard em tempo real, SSE 1 Hz, alertas via WebSocket                        |
| **Backend**          | FastAPI, SQLAlchemy 2 (async), Pydantic v2, Alembic                    | API REST, persistência, injeção de dependência, Model Registry                  |
| **Banco de dados**   | PostgreSQL 15 (Docker)                                                 | Histórico de predições                                                          |
| **Machine Learning** | Scikit-learn, XGBoost, Optuna, PyTorch Lightning, ONNX Runtime, MLflow | RF, XGBoost e MLP com hot-swap atômico via `ACTIVE_MODEL`                       |
| **MLOps**            | MLflow 2.x (Docker), PostgreSQL backend store, promoção automatizada   | Rastreamento de experimentos, registro de artefatos e promoção do melhor modelo |
| **Infraestrutura**   | Docker Desktop, Docker Compose                                         | Orquestração unificada de todos os serviços                                     |
| **IA Generativa**    | Ollama (Llama 3.2 3B), ChromaDB, MCP                                   | Assistente RAG de sugestões de reparo _(próxima fase)_                          |

---

## Arquitetura do Monorepo

```
projeto-tcc/
├── apps/
│   ├── backend/                  # API FastAPI — Clean Architecture
│   │   ├── alembic/              # Migrations do banco de dados
│   │   │   └── versions/
│   │   │       └── 0001_create_predictions_table.py
│   │   ├── src/
│   │   │   ├── core/             # config, database, exceptions, logging, rate limit
│   │   │   ├── models/           # ORM SQLAlchemy (Prediction)
│   │   │   ├── routers/          # I/O, REST, SSE, WebSocket, injeção de dependência
│   │   │   ├── schemas/          # DTOs Pydantic v2
│   │   │   └── services/         # Regras de negócio
│   │   ├── tests/
│   │   ├── alembic.ini
│   │   └── requirements.txt
│   │
│   ├── frontend/                 # Dashboard Next.js
│   │   ├── app/                  # App Router (layout, page, globals.css)
│   │   ├── components/           # header, sidebar, sensor-monitor, alert-panel, connection-status
│   │   ├── hooks/                # use-sensor-data, use-alert-websocket, use-sse
│   │   ├── lib/                  # api-client, utils
│   │   └── __tests__/
│   │
│   └── ml/                       # Pipelines de Machine Learning
│       ├── data/raw/             # Dataset CSV (ignorado pelo git)
│       ├── data/processed/       # Parquet pré-processado (ignorado pelo git)
│       ├── models/               # Artefatos treinados (.joblib, .onnx, model_card.json)
│       ├── notebooks/            # EDA e validação visual
│       ├── src/                  # ingest_metropt, preprocessing, balancing, train
│       └── tests/
│
├── infra/
│   └── nginx/
│       └── nginx.conf            # Nginx reverse proxy: entrada única, SSE/WS, timeout 3600s
│
├── docs/
│   └── model_comparison.*        # Comparação e validação dos modelos de Machine Learning
│
├── docker-compose.yml            # Orquestração de todos os serviços
└── README.md
```

---

## Pré-requisitos

Antes de começar, certifique-se de ter instalado em seu sistema:

| Ferramenta         | Versão mínima | Como instalar                                                 |
| ------------------ | ------------- | ------------------------------------------------------------- |
| **Docker Desktop** | 4.x           | [docker.com](https://www.docker.com/products/docker-desktop/) |
| **Python**         | 3.12+         | Necessário apenas para rodar Alembic localmente               |

Verificação rápida:

```bash
docker --version   # Docker version 24.x.x ou superior
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

# ── Ollama (LLM local) ────────────────────────────────────────────────────
OLLAMA_BASE_URL=http://host.docker.internal:11434

# ── Frontend ──────────────────────────────────────────────────────────────
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Subindo o Sistema Completo

O projeto é orquestrado **exclusivamente via Docker Compose**. Um único comando sobe todos os serviços (nginx, backend, frontend, banco de dados, MLflow):

```bash
# Na raiz do projeto — o arquivo .env deve existir
docker compose up --build
```

Para rodar em background:

```bash
docker compose up --build -d
```

| Serviço           | URL                          | Descrição                      |
| ----------------- | ---------------------------- | ------------------------------ |
| **Frontend**      | `http://localhost:3000`      | Dashboard Next.js              |
| **Backend / API** | `http://localhost:8000`      | API FastAPI (via Nginx)        |
| **Swagger UI**    | `http://localhost:8000/docs` | Documentação interativa da API |
| **MLflow**        | `http://localhost:5000`      | UI de experimentos e artefatos |
| **PostgreSQL**    | `localhost:5432`             | Banco de dados (interno)       |

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

Com os serviços rodando, execute as migrations diretamente no contêiner do backend:

```bash
docker compose exec api alembic upgrade head
```

Saída esperada:

```
INFO  [alembic.runtime.migration] Context impl PostgreSQLImpl.
INFO  [alembic.runtime.migration] Running upgrade  -> 0001, create predictions table — RF-09
```

### Opção 2 — Localmente (requer Python 3.12+)

Útil quando os serviços ainda não estão rodando ou para desenvolvimento local.

**Pré-requisito:** O contêiner do banco deve estar rodando:

```bash
docker compose up db -d
```

**Configure o ambiente local:**

Crie `apps/backend/.env` com `localhost` (não `db`):

```dotenv
# apps/backend/.env — usado apenas pelo Alembic e FastAPI rodando LOCALMENTE
#
# IMPORTANTE: use 'localhost', NÃO '@db:5432'. O hostname 'db' só é
# resolvível dentro da rede interna do Docker Compose.
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/tcc_db
ACTIVE_MODEL=random_forest
```

**Instale e execute:**

```bash
cd apps/backend

# Crie e ative o venv (apenas na primeira vez)
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # Windows PowerShell
# source .venv/bin/activate        # macOS / Linux

pip install -r requirements.txt
alembic upgrade head
```

### Referência de comandos Alembic

```bash
# Estado atual das migrations no banco
alembic current

# Histórico completo de migrations
alembic history --verbose

# Reverter a última migration aplicada
alembic downgrade -1

# Gerar nova migration (detecta mudanças nos modelos ORM automaticamente)
alembic revision --autogenerate -m "descricao da mudanca"
```

---

## Endpoints da API

| Endpoint              | Método      | Auth            | Descrição                                              |
| --------------------- | ----------- | --------------- | ------------------------------------------------------ |
| `/health`             | `GET`       | —               | Probe de liveness e conectividade com o banco          |
| `/predict/`           | `POST`      | —               | Inferência + persiste o resultado — 100 req/min por IP |
| `/api/v1/predictions` | `GET`       | —               | Histórico paginado (`?page=1&size=20`)                 |
| `/api/stream/sensors` | `GET`       | —               | Stream SSE de leituras a 1 Hz                          |
| `/ws/alerts`          | `WebSocket` | —               | Canal push de alertas críticos (probability > 0.70)    |
| `/api/simulator/mode` | `GET/PUT`   | —               | Consulta/altera o modo do simulador                    |
| `/models`             | `GET`       | `X-Admin-Token` | Lista modelos registrados                              |
| `/models/active`      | `PUT`       | `X-Admin-Token` | Hot-swap do modelo ativo (RF ↔ XGBoost ↔ MLP)          |
| `/docs`               | `GET`       | —               | Swagger UI interativo                                  |

**Destaques de infraestrutura:**

- **Model Registry com Atomic Hot-Swap:** troque o modelo ativo sem reiniciar o servidor via `PUT /models/active`.
- **ONNX Runtime:** o MLP é servido via `OnnxMlpAdapter`, mantendo a mesma interface que RF e XGBoost.
- **Rate limiting** via `slowapi`: 100 req/min por IP em `POST /predict/`.
- **Logging estruturado** via `structlog` (JSON em produção).

---

## Rodando os Testes

### Backend

```bash
# Com o sistema rodando via Docker
docker compose exec api pytest tests/ -v

# OU localmente (com venv ativo, na pasta apps/backend)
# Não requer PostgreSQL — usa SQLite em memória
pytest tests/ -v
```

### Frontend

```bash
# Com o sistema rodando
docker compose exec frontend pnpm test

# OU localmente (na pasta apps/frontend)
# Não requer backend rodando
pnpm test
```

---

## Ambiente de Machine Learning

O diretório `apps/ml/` possui seu **próprio ambiente virtual isolado**. Rode os scripts de treinamento localmente (fora do Docker) ou dentro do contêiner `jupyter`.

### Configurando o ambiente local

```bash
cd apps/ml

python -m venv .venv
.\.venv\Scripts\Activate.ps1    # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux

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
# Passo 1 — Ingestão
python -m src.ingest_metropt

# Passo 2a — Random Forest (modelo padrão)
python -m src.train_random_forest

# Passo 2b — XGBoost com Optuna
python src/train_xgboost.py --n-trials 100
# Use --n-trials 5 para smoke run rápido (~2 min)

# Passo 2c — MLP com PyTorch Lightning
python src/train_mlp.py --max-epochs 50
# Use --max-epochs 10 para smoke run rápido (~5 min)
```

Artefatos gerados em `apps/ml/models/`:

```
random_forest_final.joblib   ← padrão (ACTIVE_MODEL=random_forest)
xgboost_v1.joblib            ← ACTIVE_MODEL=xgboost
mlp_v1.onnx                  ← ACTIVE_MODEL=mlp (ONNX Runtime)
mlp_scaler.joblib            ← StandardScaler do MLP
*_card.json                  ← métricas e metadados de cada modelo
```

### Rodando os testes do ML

```bash
# Com o venv do ML ativo, na pasta apps/ml
pytest tests/ -v
```

---

## MLOps — MLflow + Promoção de Modelos

O serviço `mlflow` roda como contêiner dedicado. O PostgreSQL atua como backend store; os artefatos ficam em volume Docker nomeado (`mlflow_artifacts`).

```bash
# Acessar a UI de experimentos
# Com os serviços rodando: http://localhost:5000
docker compose up mlflow db -d
```

### Promovendo um modelo para produção

```bash
# Com o venv do ML ativo, na pasta apps/ml
python src/promote_model.py
```

Opções disponíveis:

```bash
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
  -d '{"model_name": "mlp"}'
```

---

## Simulador de Sensores

O backend inclui um simulador configurável com distribuições estatísticas baseadas no MetroPT-3. Útil para demonstrações sem hardware real.

### Modos disponíveis

| Modo          | Comportamento                                      |
| ------------- | -------------------------------------------------- |
| `NORMAL`      | Gaussianas estáveis em torno dos valores nominais  |
| `DEGRADATION` | Deriva progressiva ao longo de ~300 ciclos         |
| `FAILURE`     | Picos e quedas bruscas — dispara alertas WebSocket |

### Endpoints

```bash
# Consultar modo atual
curl http://localhost/api/simulator/mode

# Ativar modo de falha
curl -X PUT http://localhost/api/simulator/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "FAILURE"}'

# Retornar ao normal
curl -X PUT http://localhost/api/simulator/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "NORMAL"}'
```

### Fluxo de demonstração sugerido

1. Inicie com `NORMAL` — dashboard exibe leituras estáveis.
2. Troque para `DEGRADATION` — observe Motor Current e Oil Temperature subindo gradualmente.
3. Troque para `FAILURE` — alertas WebSocket são disparados (probability > 0.70) e toasts aparecem no dashboard.
4. Retorne a `NORMAL` para encerrar a demonstração.

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

**Causa:** Artefato do modelo ativo não encontrado.

**Solução:** Execute o treinamento correspondente ou use hot-swap para um modelo disponível:

```bash
# Via docker exec
docker compose exec api curl -X PUT http://localhost:8000/models/active \
  -H "X-Admin-Token: $ADMIN_API_TOKEN" \
  -d '{"model_name": "random_forest"}'
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

| Nome                          |
| ----------------------------- |
| Lucas de Moraes Silveira      |
| Raphael Nobuyuki Haga Okuyama |
| Ronaldo Simeone Antonio       |

---

## Licença

Este projeto é desenvolvido exclusivamente para fins acadêmicos.
