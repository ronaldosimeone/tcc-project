# TCC: Sistema Inteligente para Previsão de Falhas e Suporte à Manutenção

[![CI Pipeline](https://github.com/ronaldosimeone/tcc-project/actions/workflows/ci.yml/badge.svg)](https://github.com/ronaldosimeone/tcc-project/actions/workflows/ci.yml)

## Descrição

Projeto de Trabalho de Conclusão de Curso focado em uma plataforma integrada para gestão de ativos industriais no contexto da Indústria 4.0. O sistema une manutenção preditiva ao suporte técnico automatizado, utilizando uma arquitetura de monorepo.

O ecossistema é composto por:
- **Backend:** FastAPI
- **Frontend:** Next.js
- **Machine Learning Preditivo:** Python (Scikit-learn, ONNX)
- **Assistente Inteligente (IA Generativa):** LLM Local via Ollama + RAG (ChromaDB)
- **Banco de dados:** PostgreSQL

---

## 🧠 Inteligência Artificial e Suporte Técnico via MCP

O grande diferencial deste sistema é a capacidade de não apenas prever a falha através de Machine Learning clássico, mas também orientar o operador na sua resolução prática utilizando Inteligência Artificial Generativa e o **Model Context Protocol (MCP)**.

### 1. O Agente de Manutenção Assistida
Quando o modelo preditivo detecta uma anomalia (ex: aumento súbito no desvio padrão da vibração), o sistema aciona automaticamente nosso **Agente de Suporte**:
* **Ollama (Cérebro):** Executa o LLM (Llama 3.2 de 3B) 100% localmente para processar o diagnóstico, garantindo privacidade e funcionamento offline.
* **MCP (A Ponte de Conhecimento):** O Agente utiliza um servidor MCP customizado para conectar o LLM aos manuais de instrução técnicos da máquina armazenados no banco de dados vetorial.
* **RAG (Retrieval-Augmented Generation):** Através do MCP, o Agente realiza uma busca semântica no **ChromaDB**, extrai o capítulo exato do manual sobre a falha iminente e entrega ao técnico um plano de ação passo a passo.

### 2. Fluxo de Resolução
1. **Detecção:** O sensor reporta anomalia (análise via Isolation Forest/XGBoost).
2. **Consulta via MCP:** O Agente de IA é acionado e chama a ferramenta de busca vetorial nos manuais técnicos.
3. **Suporte:** O técnico recebe no Dashboard do Next.js a causa provável e o procedimento de reparo extraído diretamente da documentação oficial do equipamento, eliminando alucinações.

---

## Arquitetura e Tecnologias

O projeto segue o padrão monorepo, no qual múltiplas aplicações e serviços coexistem no mesmo repositório, cada um com suas responsabilidades bem definidas.

### Stack Tecnológico

- **Backend:** Python, FastAPI, SQLAlchemy, Pydantic v2, Celery, Redis
- **Frontend:** Next.js 15 (App Router), React 19, TypeScript, Tailwind CSS, Shadcn/ui
- **Machine Learning (Clássico):** Scikit-learn, XGBoost, ONNX Runtime
- **IA Generativa & Agentes:** Ollama, ChromaDB, Model Context Protocol (MCP)
- **Infraestrutura:** Docker e Docker Compose, Nginx

### Estrutura do Monorepo

```text
projeto-tcc/
├── apps/
│   ├── backend/                 # API FastAPI (Clean Architecture)
│   │   ├── src/
│   │   │   ├── core/            # config, database, exceptions
│   │   │   ├── routers/         # I/O e injeção de dependência
│   │   │   ├── schemas/         # DTOs Pydantic v2
│   │   │   ├── services/        # Regras de negócio
│   │   │   └── models/          # Modelos SQLAlchemy
│   │   └── tests/
│   ├── frontend/                # Dashboard Next.js 15
│   └── ml/                      # Pipelines de Machine Learning
│       ├── data/
│       │   ├── raw/             # Datasets originais (ignorado pelo git)
│       │   └── processed/       # Parquet prontos para uso (ignorado pelo git)
│       ├── notebooks/           # Análise exploratória (EDA)
│       ├── src/                 # Scripts e classes de ML
│       └── tests/
├── .env.example
├── CLAUDE.md                    # Diretrizes de arquitetura e padrões de código
├── AGENTS.md                    # Definição de MCPs e Personas de IA utilizadas no dev
└── docker-compose.yml
```

---

## Pré-requisitos

Antes de executar o projeto, é necessário ter instalado:

- Docker e Docker Compose
- Node.js + pnpm
- Python 3.11 ou superior
- [Ollama](https://ollama.com/) (para o módulo de Assistente de Manutenção)

---

## Como rodar o projeto

### 1. Clone o repositório

```bash
git clone [https://github.com/ronaldosimeone/tcc-project.git](https://github.com/ronaldosimeone/tcc-project.git)
cd tcc-project
```

### 2. Configure as variáveis de ambiente

```bash
cp .env.example .env
# Ajuste as variáveis (credenciais de banco, URLs do Ollama, etc.)
```

### 3. Suba os serviços com Docker

```bash
docker-compose up --build
```

---

## Serviços disponíveis

Após a inicialização, os serviços estarão operando nas seguintes portas:

| Serviço | URL |
|---|---|
| Frontend (Dashboard) | http://localhost:3000 |
| Backend (API) | http://localhost:8000 |
| Documentação Swagger | http://localhost:8000/docs |
| PostgreSQL | localhost:5432 |
| Ollama (LLM Local) | localhost:11434 |

---

## Backend — Setup FastAPI

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

---

## Machine Learning — Dataset MetroPT-3

Pipeline de ingestão e processamento do dataset [MetroPT-3](https://archive.ics.uci.edu/dataset/791/metropt-3+dataset) (compressor de ar de um trem do metrô do Porto).

O script baixa o ZIP da UCI, valida o schema e persiste como `.parquet`. A ingestão é idempotente.

```bash
# A partir da raiz do monorepo
python -m apps.ml.src.ingest_metropt
```

---

## Machine Learning — Feature Engineering

Para alimentar a IA preditiva, o sistema converte os dados brutos ruidosos em indicativos dinâmicos de falha através da classe `MetroPTPreprocessor`. O compressor opera em ciclos de carga/descarga a ~1 Hz, logo, falhas mecânicas manifestam-se como mudanças na dinâmica do sinal.

As transformações incluem:

1. **Imputação de Nulos:** `ffill` seguido de `bfill` para cobrir perdas de pacote sem gerar saltos irreais.
2. **Delta de Pressão (Δp):** Indicador de eventos anômalos de válvula.
3. **Desvio Padrão Janelado (std_5):** Captura a instabilidade mecânica e desgaste de rolamentos (volatilidade do sinal).
4. **Médias Móveis (MA_5 e MA_15):** Suaviza ruídos (alta frequência) e acusa tendências de degradação térmica (crossover).

Essa engenharia é 100% compatível com a API do Scikit-Learn e entra direto no Pipeline de inferência.

### Uso do `MetroPTPreprocessor`

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from apps.ml.src.preprocessing import MetroPTPreprocessor
import pandas as pd

df_raw = pd.read_parquet("apps/ml/data/processed/metropt3.parquet")

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
```

Os testes utilizam a fixture `tmp_path` do pytest — nenhum arquivo é gravado em `data/` durante a execução.

---

## Qualidade e Padrões de Código

O repositório é governado por regras estritas descritas no arquivo `CLAUDE.md`.

### Backend (Python)
- **Tipagem Estrita:** Pydantic v2 e Type Hints.
- **Formatação e Linting:** Ruff e Black.
- **Análise Estática:** Mypy.

### Frontend (Next.js/TypeScript)
- **Tipagem:** TypeScript estrito (proibido `any`).
- **Arquitetura:** Prioridade para Server Components; Client Components isolados para streaming/SSE.
- **UI:** Componentização com Shadcn/ui e TailwindCSS.

### Automação (Pre-commit)

O pipeline de CI/CD barra pull requests que não passem nos testes ou linters. Para instalar os ganchos locais:

```bash
pip install pre-commit
pre-commit install
```

---

## Observações Arquiteturais

- Cada aplicação (`backend` e `ml`) possui seu próprio `requirements.txt`.
- A pasta `apps/ml/data/` está no `.gitignore` — **nunca commite os datasets**.
- Modelos preditivos treinados devem ser exportados para **ONNX** antes de serem servidos pela API para garantir máxima performance de inferência.

---

## Autores

- Lucas de Moraes Silveira
- Raphael Nobuyuki Haga Okuyama
- Ronaldo Simeone Antonio

---

## Licença

Este projeto é desenvolvido exclusivamente para fins acadêmicos.