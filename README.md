# TCC: Sistema Inteligente para Previsão de Falhas e Suporte à Manutenção

[![CI Pipeline](https://github.com/ronaldosimeone/tcc-project/actions/workflows/ci.yml/badge.svg)](https://github.com/ronaldosimeone/tcc-project/actions/workflows/ci.yml)

## Descrição

Projeto de Trabalho de Conclusão de Curso (Engenharia de Computação - FACENS) focado em uma plataforma integrada para gestão de ativos industriais no contexto da Indústria 4.0. O sistema une manutenção preditiva ao suporte técnico automatizado, utilizando uma arquitetura de monorepo.

O ecossistema é composto por:
- **Backend:** FastAPI
- **Frontend:** Next.js
- **Machine Learning & IA:** Python (Scripts, Notebooks, LLM via MCP)
- **Banco de dados:** PostgreSQL

---

## Arquitetura

O projeto segue o padrão monorepo, no qual múltiplas aplicações e serviços coexistem no mesmo repositório, cada um com suas responsabilidades bem definidas e orquestrados de forma unificada.

### Tecnologias utilizadas

- **Backend:** Python, FastAPI, SQLAlchemy, Pydantic, Celery, Redis
- **Frontend:** Next.js 15 (App Router), React 19, TypeScript, Tailwind CSS, Shadcn/ui
- **Machine Learning:** Scikit-learn, XGBoost, ONNX Runtime, ChromaDB, Anthropic SDK
- **Infraestrutura:** Docker e Docker Compose, Nginx

### Estrutura do Monorepo

```
projeto-tcc/
├── apps/
│   ├── backend/                 # API FastAPI (Clean Architecture)
│   │   ├── src/
│   │   │   ├── core/            # config, database, exceptions
│   │   │   ├── routers/         # I/O e injeção de dependência
│   │   │   ├── schemas/         # DTOs Pydantic v2
│   │   │   ├── services/        # Regras de negócio
│   │   │   └── models/          # Modelos SQLAlchemy
│   │   └── tests/
│   ├── frontend/                # Dashboard Next.js 15
│   └── ml/                      # Pipelines de Machine Learning
│       ├── data/
│       │   ├── raw/             # Datasets originais (ignorado pelo git)
│       │   └── processed/       # Parquet prontos para uso (ignorado pelo git)
│       ├── notebooks/           # Análise exploratória (EDA)
│       ├── src/                 # Scripts e classes de ML
│       └── tests/
├── .env.example
├── CLAUDE.md                    # Diretrizes de arquitetura e padrões de código
└── docker-compose.yml
```

---

## Pré-requisitos

Antes de executar o projeto, é necessário ter instalado:

- Docker e Docker Compose
- Node.js + pnpm
- Python 3.11 ou superior

---

## Como rodar o projeto

### 1. Clone o repositório

```bash
git clone https://github.com/ronaldosimeone/tcc-project.git
cd tcc-project
```

### 2. Configure as variáveis de ambiente

```bash
cp .env.example .env
# Ajuste as variáveis (credenciais de banco, URLs, etc.)
```

### 3. Suba os serviços com Docker

```bash
docker-compose up --build
```

---

## Serviços disponíveis

Após a inicialização:

| Serviço | URL |
|---|---|
| Frontend (Dashboard) | http://localhost:3000 |
| Backend (API) | http://localhost:8000 |
| Documentação Swagger | http://localhost:8000/docs |
| PostgreSQL | localhost:5432 |

---

## Backend — Setup FastAPI (Sprint 1, Task 4)

API REST construída com FastAPI, SQLAlchemy assíncrono e Pydantic v2, seguindo Clean Architecture.

### Rodando o backend localmente

```bash
cd apps/backend
pip install -r requirements.txt

# Inicia o servidor de desenvolvimento
uvicorn src.main:app --reload --port 8000
```

### Health check

```bash
curl http://localhost:8000/health
```

Resposta esperada:
```json
{
  "status": "ok",
  "version": "0.1.0",
  "database": {
    "connected": true,
    "latency_ms": 1.23
  }
}
```

Quando o banco não está acessível, `status` retorna `"degraded"` sem derrubar a API.

### Variáveis de ambiente do backend

| Variável | Padrão | Descrição |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/tcc_db` | Connection string do PostgreSQL |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Endpoint do LLM local (Ollama) |
| `ALLOWED_ORIGINS` | `["http://localhost:3000"]` | CORS — origens permitidas |
| `DEBUG` | `false` | Ativa echo SQL e logs verbosos |

### Testes do backend

```bash
cd apps/backend
pytest tests/ -v
```

Os testes usam SQLite in-memory via `aiosqlite` — não exigem PostgreSQL rodando.

---

## Machine Learning — Dataset MetroPT-3 (Sprint 1, Task 3)

Pipeline de ingestão do dataset [MetroPT-3](https://archive.ics.uci.edu/dataset/791/metropt-3+dataset) (UCI ML Repository): compressor de ar de um trem do metrô do Porto, utilizado para detecção de falhas preditivas.

### Ingestão dos dados

O script baixa o ZIP da UCI, valida o schema e persiste como `.parquet`.
**É idempotente**: se o `.parquet` já existir, a execução é ignorada.

```bash
# A partir da raiz do monorepo
python -m apps.ml.src.ingest_metropt

# Ou a partir de apps/ml/
cd apps/ml
python src/ingest_metropt.py
```

Saída esperada:
```
[INFO] Data directories are ready: …/data/raw | …/data/processed
[INFO] Downloading MetroPT-3 dataset from https://…
[INFO] Download complete: … (XX.XX MB)
[INFO] Loaded NNNNNN rows × 16 columns.
[INFO] Schema validation passed – all 16 expected columns present.
[INFO] Preprocessing complete. Final shape: NNNNNN rows × 16 columns.
[INFO] Saved Parquet to …/data/processed/metropt3.parquet
[INFO] Ingestion pipeline finished successfully.
```

### Análise Exploratória (EDA)

```bash
cd apps/ml
jupyter lab notebooks/01_eda_metropt.ipynb
```

O notebook cobre:
- Estatísticas descritivas e verificação de nulos
- Distribuição de classes (labels de falha)
- Séries temporais dos sensores (TP2, TP3, H1, DV_pressure, Motor_current, Oil_temperature)
- Histogramas de distribuição individual
- Matriz de correlação com anotações numéricas
- Detecção de outliers via IQR

---

## Machine Learning — Feature Engineering (Sprint 2, RF-02)

### Por que essas features?

O compressor opera em ciclos de carga/descarga a ~1 Hz. Falhas mecânicas não se manifestam como limiares violados em um único instante — elas aparecem como **mudanças na dinâmica** do sinal ao longo do tempo. As quatro transformações abaixo capturam essa dinâmica com custo computacional mínimo.

#### 1. Imputação de Nulos (ffill → bfill)

Sensores industriais produzem leituras ausentes por perda de pacote, reinicialização do CLP ou ruído elétrico. A **propagação da última leitura válida** (`ffill`) é semanticamente correta para séries temporais físicas: o compressor não muda de estado instantaneamente. O `bfill` subsequente cobre lacunas no início da série onde não há valor anterior.

#### 2. Delta de Pressão — Δp(t)

```
Δp(t) = p(t) − p(t−1)
```

A derivada discreta da pressão de saída (`TP2`) é um indicador de **eventos de válvula**. Um `Δp` negativo abrupto indica abertura inesperada da válvula de alívio; um `Δp` positivo fora do perfil de carga sinaliza bloqueio na linha de descarga.

#### 3. Desvio Padrão Janelado — std_w(t)

```
std_w(t) = σ( x[t−w+1 : t+1] )    (w = 5 por padrão)
```

A **volatilidade local** captura instabilidade mecânica crescente. Uma janela de 5 amostras cobre exatamente um ciclo de carga/descarga a 1 Hz. Um aumento em `Motor_current_std_5` antecede falhas de rolamentos; um aumento em `TP2_std_5` antecede falhas de anel de pistão (documentadas no paper original do MetroPT-3).

#### 4. Médias Móveis — MA_5 e MA_15

```
MA_k(t) = mean( x[t−k+1 : t+1] )    k ∈ {5, 15}
```

- **Suavização de ruído**: `MA_5` remove oscilações de alta frequência sem atraso excessivo.
- **Detecção de tendência**: quando `MA_5` cruza `MA_15` para cima em `Oil_temperature`, indica aquecimento acelerado antes de um desligamento térmico — sinal de manutenção preventiva.

### Uso do `MetroPTPreprocessor`

```python
from apps.ml.src.preprocessing import MetroPTPreprocessor
import pandas as pd

df_raw = pd.read_parquet("apps/ml/data/processed/metropt3.parquet")

preprocessor = MetroPTPreprocessor(
    window_std=5,
    window_ma_short=5,
    window_ma_long=15,
)
df_features = preprocessor.fit_transform(df_raw)
```

Por herdar `BaseEstimator` + `TransformerMixin` do scikit-learn, pode ser encadeado diretamente em um `Pipeline`:

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest

pipe = Pipeline([
    ("features", MetroPTPreprocessor()),
    ("model",    IsolationForest(contamination=0.01)),
])
pipe.fit(df_raw)
```

### Testes do módulo ML

```bash
# A partir da raiz do monorepo
pytest apps/ml/tests/ -v

# Com relatório de cobertura
pytest apps/ml/tests/ -v --cov=apps.ml.src --cov-report=term-missing
```

Os testes utilizam a fixture `tmp_path` do pytest — nenhum arquivo é gravado em `data/` durante a execução.

---

## Qualidade e padrões de código

### Backend (Python)
- **Ruff** — linting
- **Black** — formatação
- **Mypy** — tipagem estática

### Frontend (TypeScript)
- **ESLint** — linting e regras de React
- **Prettier** — formatação

### Automação (Pre-commit)

```bash
pip install pre-commit
pre-commit install
```

O pipeline de CI/CD (GitHub Actions) barra pull requests que não passem nos testes ou linters.

---

## Observações importantes

- Cada aplicação (`backend` e `ml`) possui seu próprio `requirements.txt`.
- Os diretórios `venv/` e `node_modules/` estão no `.gitignore` e não devem ser versionados.
- A pasta `apps/ml/data/` está no `.gitignore` — **nunca commite os datasets**.
- Modelos treinados devem ser exportados para **ONNX** antes de serem servidos pela API (`CLAUDE.md §4`).

---

## Autores

- Lucas de Moraes Silveira
- Raphael Nobuyuki Haga Okuyama
- Ronaldo Simeone Antonio

---

## Licença

Este projeto é desenvolvido exclusivamente para fins acadêmicos.
