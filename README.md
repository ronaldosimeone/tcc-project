# PredictIQ — Sistema de Manutenção Preditiva Industrial

> **TCC · Indústria 4.0** — Detecção de falhas em tempo real para o compressor MetroPT-3, com dashboard interativo, histórico persistido e assistente inteligente via LLM local + MCP.

---

## Visão Geral das Tecnologias

| Camada | Tecnologia | Responsabilidade |
|---|---|---|
| **Frontend** | Next.js 16, TypeScript strict, Tailwind CSS v4, shadcn/ui, Recharts | Dashboard em tempo real, polling 5 s, alertas visuais |
| **Backend** | FastAPI, SQLAlchemy 2 (async), Pydantic v2, Alembic | API REST, persistência, injeção de dependência, Model Registry |
| **Banco de dados** | PostgreSQL 15 (Docker) | Histórico de predições |
| **Machine Learning** | Scikit-learn, XGBoost, Optuna, PyTorch Lightning, ONNX Runtime, MLflow | RF, XGBoost e MLP com hot-swap atômico via `ACTIVE_MODEL` |
| **Infraestrutura** | Docker Desktop, Docker Compose | Orquestração dos serviços |
| **IA Generativa** | Ollama (Llama 3.2 3B), ChromaDB, MCP | Assistente RAG de sugestões de reparo *(próxima fase)* |

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
│   │   ├── components/           # header, sidebar, sensor-monitor, alert-panel
│   │   ├── hooks/                # use-sensor-data, use-prediction-history
│   │   ├── lib/                  # api-client, utils
│   │   └── __tests__/
│   │
│   └── ml/                       # Pipelines de Machine Learning
│       ├── data/raw/             # Dataset CSV (ignorado pelo git)
│       ├── data/processed/       # Parquet pré-processado (ignorado pelo git)
│       ├── models/               # Artefatos treinados (.joblib, model_card.json)
│       ├── notebooks/            # EDA e validação visual
│       ├── src/                  # ingest_metropt, preprocessing, balancing, train
│       └── tests/
│
├── infra/
│   └── nginx/
│       └── nginx.conf            # Nginx reverse proxy (S6): entrada única, SSE/WS, timeout 3600s
│
├── docs/
│   └── model_comparison.*        # Comparação e validação dos modelos de Machine Learning
│
├── docker-compose.yml            # Orquestração dos serviços (nginx, backend, frontend, ml, db)
├── CLAUDE.md                     # Diretrizes de código do projeto
└── README.md

```

---

## Pré-requisitos

Antes de começar, certifique-se de ter instalado em seu sistema:

| Ferramenta | Versão mínima | Como instalar |
|---|---|---|
| **Python** | 3.12+ | [python.org](https://www.python.org/downloads/) |
| **Node.js** | 20 LTS | [nodejs.org](https://nodejs.org/) |
| **pnpm** | 9+ | `npm install -g pnpm` |
| **Docker Desktop** | 4.x | [docker.com](https://www.docker.com/products/docker-desktop/) |

Verificação rápida:

```bash
python --version    # Python 3.12.x ou superior
node --version      # v20.x.x
pnpm --version      # 9.x.x
docker --version    # Docker version 24.x.x ou superior
```

---

## Configuração de Variáveis de Ambiente

O projeto usa **dois arquivos `.env` distintos**, cada um com um contexto de execução diferente.

### Por que dois arquivos?

Dentro do Docker, os serviços se comunicam usando o nome do serviço como hostname (ex: `db`). Quando você roda o Alembic ou o FastAPI **diretamente no Windows** (fora do contêiner), o acesso ao banco precisa ser via `localhost`. Os dois arquivos resolvem essa diferença de contexto de rede.

---

### Arquivo 1 — `.env` na raiz do projeto

Lido pelo **Docker Compose** para configurar o contêiner PostgreSQL e os demais serviços.

Crie `projeto-tcc/.env`:

```dotenv
# ── PostgreSQL — cria o banco ao subir o contêiner ────────────────────────
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_DB=tcc_db

# ── Backend dentro do Docker (hostname 'db' = nome do serviço na rede) ────
DATABASE_URL=postgresql+asyncpg://user:password@db:5432/tcc_db

# ── Ollama (LLM local) ────────────────────────────────────────────────────
OLLAMA_BASE_URL=http://host.docker.internal:11434

# ── Frontend ──────────────────────────────────────────────────────────────
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

### Arquivo 2 — `apps/backend/.env`

Lido pelo **Alembic e pelo FastAPI quando executados diretamente no Windows** (fora do Docker). A diferença crítica está no hostname: use `localhost`, não `db`.

Crie `projeto-tcc/apps/backend/.env`:

```dotenv
# ── IMPORTANTE: use 'localhost', NÃO '@db' ────────────────────────────────
#
# Usar '@db:5432' fora de um contêiner Docker causa "InvalidPasswordError"
# ou falha de DNS, pois o hostname 'db' só é resolvível dentro da rede
# interna do Docker Compose.
#
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/tcc_db

# ── RF-10: Selecção do modelo ativo (sem redeploy) ────────────────────────
# Valores: random_forest (padrão) | xgboost
ACTIVE_MODEL=random_forest

# ── Caminhos alternativos para os artefatos ───────────────────────────────
# MODEL_PATH=../../ml/models/random_forest_final.joblib
# XGBOOST_MODEL_PATH=../../ml/models/xgboost_v1.joblib

# ── Habilita logs SQL do SQLAlchemy (desenvolvimento) ────────────────────
DEBUG=false
```

---

## Subindo o Banco de Dados

O PostgreSQL 15 é provisionado via Docker. Você não precisa instalar o PostgreSQL localmente.

### 1. Inicie o serviço de banco de dados

```bash
# Na raiz do projeto
docker compose up db -d
```

### 2. Aguarde o banco ficar saudável

```bash
docker compose ps

# Saída esperada:
# NAME   IMAGE         STATUS
# db     postgres:15   Up (healthy)
```

O serviço está pronto quando o status for `(healthy)`. O `healthcheck` executa `pg_isready -U user -d tcc_db` a cada 5 segundos.

### Parar o banco

```bash
# Para e remove os contêineres (dados persistidos no volume Docker)
docker compose down

# Para, remove contêineres E apaga todos os dados (reset completo)
docker compose down -v
```

---

## Rodando as Migrations (Alembic)

As migrations criam e versionam o schema do PostgreSQL. A tabela `predictions` (RF-09) é criada pela migration `0001`.

### 1. Entre na pasta do backend e ative o ambiente virtual

```bash
cd apps/backend

# Crie o venv (apenas na primeira vez)
python -m venv .venv
```

Ativação:

```bash
# Windows — PowerShell
.\.venv\Scripts\Activate.ps1

# Windows — Prompt de Comando (CMD)
.\.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

O prompt deve exibir `(.venv)` na frente, confirmando a ativação.

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Aplique as migrations

```bash
# O arquivo apps/backend/.env deve existir com localhost:5432
# O contêiner do banco deve estar rodando (docker compose up db -d)

alembic upgrade head
```

Saída esperada:

```
INFO  [alembic.runtime.migration] Context impl PostgreSQLImpl.
INFO  [alembic.runtime.migration] Running upgrade  -> 0001, create predictions table — RF-09
```

### Referência de comandos Alembic

```bash
# Estado atual da migration no banco
alembic current

# Histórico completo de migrations
alembic history --verbose

# Reverter a última migration aplicada
alembic downgrade -1

# Gerar nova migration (detecta mudanças nos modelos ORM automaticamente)
alembic revision --autogenerate -m "descricao da mudanca"
```

---

## Rodando a API (Backend)

### 1. Ative o venv (se não estiver ativo)

```bash
cd apps/backend
.\.venv\Scripts\Activate.ps1      # Windows PowerShell
```

### 2. Inicie o servidor

```bash
uvicorn src.main:app --reload --port 8000
```

O servidor inicia em **`http://localhost:8000`**.

| Endpoint | Método | Auth | Descrição |
|---|---|---|---|
| `/health` | `GET` | — | Probe de liveness e conectividade com o banco |
| `/predict/` | `POST` | — | Inferência + persiste o resultado — 100 req/min por IP (RNF-19) |
| `/api/v1/predictions` | `GET` | — | Histórico paginado (`?page=1&size=20`) |
| `/models` | `GET` | `X-Admin-Token` | Lista modelos registrados e disponibilidade dos artefatos (RF-11) |
| `/models/active` | `PUT` | `X-Admin-Token` | Hot-swap atômico do modelo ativo em tempo de execução (RNF-25) |
| `/docs` | `GET` | — | Swagger UI interativo |

**Destaques de infraestrutura:**
- **RNF-25 — Model Registry com Atomic Hot-Swap:** troque o modelo ativo (RF ↔ XGBoost ↔ MLP) sem reiniciar o servidor via `PUT /models/active`. O carregamento ocorre em background thread; apenas o ponteiro é trocado sob `asyncio.Lock`.
- **RNF-24 — Deep Learning com ONNX Runtime:** o MLP é exportado em ONNX e serve via `OnnxMlpAdapter`, mantendo a mesma interface que RF e XGBoost — latência p95 < 500 ms.
- **RNF-18** — Handler global: qualquer erro não tratado retorna HTTP 500 com JSON seguro (sem stack trace).
- **RNF-19** — Rate limiting via `slowapi`: 100 req/min por IP em `POST /predict/`.
- **Logging estruturado** via `structlog` (JSON em produção, legível em desenvolvimento).

### Exemplo — predição

```bash
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "TP2": 5.02, "TP3": 9.21, "H1": 8.97,
    "DV_pressure": 2.10, "Reservoirs": 8.85,
    "Motor_current": 4.5, "Oil_temperature": 72.3,
    "COMP": 1.0, "DV_eletric": 0.0, "Towers": 1.0,
    "MPG": 1.0, "Oil_level": 1.0
  }'
```

Resposta:

```json
{
  "predicted_class": 0,
  "failure_probability": 0.042371,
  "timestamp": "2024-06-01T12:00:00.000000+00:00"
}
```

### Exemplo — histórico paginado

```bash
curl "http://localhost:8000/api/v1/predictions?page=1&size=10"
```

```json
{
  "items": [ { "id": 42, "timestamp": "...", "TP2": 5.02, "failure_probability": 0.04 } ],
  "total": 42,
  "page": 1,
  "size": 10,
  "pages": 5
}
```

### Rodando os testes do backend

```bash
# Com o venv ativo, na pasta apps/backend
# Não requer PostgreSQL — usa SQLite em memória
pytest tests/ -v
# Inclui testes de integração do ModelRegistry, autenticação e endpoints /models.
```

---

## Ambiente de Machine Learning

O diretório `apps/ml/` possui seu **próprio ambiente virtual isolado**, separado do backend, para evitar que dependências pesadas de ciência de dados (Pandas, Scikit-learn, imbalanced-learn) contaminem o servidor de produção.

### 1. Configure o ambiente virtual do ML

```bash
cd apps/ml

# Crie o venv (apenas na primeira vez)
python -m venv .venv
```

Ativação:

```bash
# Windows — PowerShell
.\.venv\Scripts\Activate.ps1

# Windows — CMD
.\.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Baixe e posicione o dataset

Faça o download do **MetroPT-3 Air Compressor Dataset** na [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/791/metropt-3+dataset) e salve o arquivo CSV em:

```
apps/ml/data/raw/MetroPT3(AirCompressor).csv
```

### 4. Execute o pipeline na ordem correta

Todos os comandos devem ser executados a partir de `apps/ml/` com o venv ativo:

```bash
# Passo 1 — Ingestão
# Lê o CSV, mapeia os 4 intervalos de falha real para a coluna 'anomaly'
# e salva o resultado como Parquet em data/processed/
python -m src.ingest_metropt
```

```bash
# Passo 2a — Random Forest (modelo padrão)
# Pré-processa (rolling features), divide (80/20 estratificado),
# executa GridSearchCV com SMOTE anti-leakage dentro de cada fold,
# faz refit com best_params_ e salva os artefatos em models/
python -m src.train_random_forest
```

```bash
# Passo 2b — XGBoost com Optuna (RF-10, RNF-23)
# 100 trials de hiperparâmetros (retomável — SQLite persiste o estudo).
# Use --n-trials 5 para um smoke run rápido (~2 min).
python src/train_xgboost.py --n-trials 100
```

Após o treinamento, os seguintes artefatos são gerados:

```bash
# Passo 2c — MLP com PyTorch Lightning (RNF-24)
# Treina uma rede [256→128→64], exporta ONNX e rastreia no MLflow local.
# Use --max-epochs 10 para um smoke run rápido (~5 min).
python src/train_mlp.py --max-epochs 50
```

Após o treinamento, os seguintes artefatos são gerados:

```
apps/ml/models/
├── random_forest_final.joblib   ← padrão (ACTIVE_MODEL=random_forest)
├── model_card.json              ← métricas RF
├── xgboost_v1.joblib            ← ativado com ACTIVE_MODEL=xgboost
├── xgboost_v1_card.json         ← métricas XGBoost + melhores hiperparâmetros
├── mlp_v1.onnx                  ← ativado com ACTIVE_MODEL=mlp (ONNX Runtime)
├── mlp_scaler.joblib            ← StandardScaler ajustado no X_train
└── mlp_v1_card.json             ← métricas MLP + arquitetura

apps/ml/data/optuna/
└── xgboost_study.db             ← estudo Optuna persistido (RNF-23)

apps/ml/mlruns/                  ← experimentos MLflow (RNF-24)
└── mlp_metropt3/                   abra com: mlflow ui --backend-store-uri apps/ml/mlruns
```

> Veja `docs/model_comparison.md` para a tabela completa de métricas (F1, AUC, latência p50/p95).

### Detalhes do pipeline

| Etapa | Módulo | Descrição |
|---|---|---|
| **Ingestão** | `ingest_metropt.py` | Lê CSV, cria coluna `anomaly` com 4 intervalos de falha real, salva Parquet |
| **Pré-processamento** | `preprocessing.py` | `MetroPTPreprocessor`: delta, std_5, ma_5, ma_15 nos 7 sensores analógicos |
| **Balanceamento** | `balancing.py` | `MetroPTBalancer`: SMOTE aplicado **somente** dentro de cada fold (anti-leakage) |
| **Treinamento RF** | `train_random_forest.py` | `GridSearchCV` sobre `imblearn.Pipeline([SMOTE → RF])` com `cv=3, scoring='f1'` |
| **Treinamento XGBoost** | `train_xgboost.py` | Optuna (100 trials, SQLite) + `scale_pos_weight` (sem SMOTE), `tree_method=hist` |
| **Treinamento MLP** | `train_mlp.py` | PyTorch Lightning `[256→128→64]`, AdamW, EarlyStopping; exporta ONNX + StandardScaler; rastreado no MLflow (RNF-24) |

### Rodando os testes do ML

```bash
# Com o venv do ML ativo, na pasta apps/ml
pytest tests/ -v
# 137 testes: preprocessing, balancing, ingestão, RF, XGBoost, MLP (ONNX)
# Os testes MLP são ignorados automaticamente se os artefatos ainda não existirem.
```

---

## Rodando o Dashboard (Frontend)

### 1. Crie o arquivo de configuração

Crie `apps/frontend/.env.local`:

```dotenv
# URL da API — deve apontar para o backend em execução
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 2. Instale as dependências

```bash
cd apps/frontend
pnpm install
```

### 3. Inicie o servidor de desenvolvimento

```bash
pnpm dev
```

O dashboard estará disponível em **`http://localhost:3000`**.

> **Pré-condição:** O backend precisa estar rodando em `localhost:8000` antes de abrir o dashboard. Sem ele, o painel exibe o badge "Backend offline" e o polling de sensores não funciona.

### Rodando os testes do frontend

```bash
# Na pasta apps/frontend (não requer backend rodando)
pnpm test
```

---

## Subindo Tudo via Docker

Para orquestrar todos os serviços de uma vez:

```bash
# Na raiz do projeto — o arquivo .env deve existir
docker compose up --build
```

| Serviço | URL | Descrição |
|---|---|---|
| **Frontend** | `http://localhost:3000` | Dashboard Next.js |
| **Backend** | `http://localhost:8000` | API FastAPI |
| **Jupyter** | `http://localhost:8888` (token: `admin`) | Notebooks de ML |
| **PostgreSQL** | `localhost:5432` | Banco de dados |

> **Lembre-se:** Com Docker, as migrations **não são executadas automaticamente**. Após `docker compose up`, aplique as migrations com o venv do backend ativo e o arquivo `apps/backend/.env` configurado com `localhost:5432`.

---

## Troubleshooting

### `InvalidPasswordError` ou falha de conexão no `alembic upgrade head`

**Causa:** O `DATABASE_URL` usa `@db:5432`. O hostname `db` só existe dentro da rede Docker.

**Solução:** Verifique `apps/backend/.env` e confirme que está usando `localhost`:

```dotenv
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/tcc_db
```

---

### `POST /predict/` retorna HTTP 503

**Causa:** O artefato do modelo ativo não foi encontrado no caminho esperado.

**Solução:** Confirme qual modelo está ativo e execute o treinamento correspondente:

```bash
# Verificar qual modelo está configurado
echo $ACTIVE_MODEL  # default: random_forest

# Treinar o Random Forest (se necessário)
cd apps/ml && python -m src.train_random_forest

# Treinar o XGBoost (se ACTIVE_MODEL=xgboost)
cd apps/ml && python src/train_xgboost.py --n-trials 5

# Treinar o MLP (se ACTIVE_MODEL=mlp)
cd apps/ml && python src/train_mlp.py --max-epochs 10
```

Alternativamente, com o servidor rodando, use o hot-swap para carregar um modelo disponível sem reiniciar:

```bash
curl -X PUT http://localhost:8000/models/active \
  -H "X-Admin-Token: $ADMIN_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "random_forest"}'
```

---

### `ACTIVE_MODEL=xgboost` mas a API continua usando o Random Forest

**Causa:** A variável de ambiente não está sendo lida pelo processo FastAPI.

**Solução:** Adicione ao `apps/backend/.env`:

```dotenv
ACTIVE_MODEL=xgboost
```

E reinicie o servidor (`uvicorn src.main:app --reload`). O log de startup exibirá `model_loaded model=xgboost`.

> **Alternativa sem reiniciar:** use `PUT /models/active` para trocar o modelo em tempo de execução (RNF-25).

---

### `pnpm dev` exibe erro `NEXT_PUBLIC_API_URL não está definida`

**Causa:** O arquivo `apps/frontend/.env.local` não foi criado.

**Solução:**

```bash
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > apps/frontend/.env.local
```

---

### PowerShell bloqueia a ativação do venv

**Causa:** A política de execução do Windows impede scripts `.ps1`.

**Solução:** Execute o PowerShell como Administrador e rode:

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

---
