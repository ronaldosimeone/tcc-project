# TCC: Sistema Inteligente para Previsão de Falhas e Suporte à Manutenção

[](https://github.com/ronaldosimeone/tcc-project/actions/workflows/ci.yml)

## Descrição

Projeto de Trabalho de Conclusão de Curso focado em uma plataforma integrada para gestão de ativos industriais no contexto da Indústria 4.0. O sistema une manutenção preditiva ao suporte técnico automatizado, utilizando uma arquitetura de monorepo.

O ecossistema é composto por:

  - **Backend:** FastAPI
  - **Frontend:** Next.js
  - **Machine Learning Preditivo:** Python (Scikit-learn, ONNX)
  - **Assistente Inteligente (IA Generativa):** LLM Local via Ollama + RAG (ChromaDB)
  - **Banco de dados:** PostgreSQL

-----

## 🧠 Inteligência Artificial e Suporte Técnico via MCP

O grande diferencial deste sistema é a capacidade de não apenas prever a falha através de Machine Learning clássico, mas também orientar o operador na sua resolução prática utilizando Inteligência Artificial Generativa e o **Model Context Protocol (MCP)**.

### 1\. O Agente de Manutenção Assistida

Quando o modelo preditivo detecta uma anomalia (ex: aumento súbito no desvio padrão da vibração), o sistema aciona automaticamente nosso **Agente de Suporte**:

  * **Ollama (Cérebro):** Executa o LLM (Llama 3.2 de 3B) 100% localmente para processar o diagnóstico, garantindo privacidade e funcionamento offline.
  * **MCP (A Ponte de Conhecimento):** O Agente utiliza um servidor MCP customizado para conectar o LLM aos manuais de instrução técnicos da máquina armazenados no banco de dados vetorial.
  * **RAG (Retrieval-Augmented Generation):** Através do MCP, o Agente realiza uma busca semântica no **ChromaDB**, extrai o capítulo exato do manual sobre a falha iminente e entrega ao técnico um plano de ação passo a passo.

### 2\. Fluxo de Resolução

1.  **Detecção:** O sensor reporta anomalia (análise via Random Forest/XGBoost).
2.  **Consulta via MCP:** O Agente de IA é acionado e chama a ferramenta de busca vetorial nos manuais técnicos.
3.  **Suporte:** O técnico recebe no Dashboard do Next.js a causa provável e o procedimento de reparo extraído diretamente da documentação oficial do equipamento, eliminando alucinações.

-----

## Arquitetura e Tecnologias

O projeto segue o padrão monorepo, no qual múltiplas aplicações e serviços coexistem no mesmo repositório, cada um com suas responsabilidades bem definidas.

### Stack Tecnológico

  - **Backend:** Python, FastAPI, SQLAlchemy, Pydantic v2, Celery, Redis
  - **Frontend:** Next.js 15 (App Router), React 19, TypeScript, Tailwind CSS, Shadcn/ui
  - **Machine Learning:** Scikit-learn, Imbalanced-learn (SMOTE), Pandas, Jupyter
  - **IA Generativa & Agentes:** Ollama, ChromaDB, Model Context Protocol (MCP)
  - **Infraestrutura:** Docker e Docker Compose, Nginx

### Estrutura do Monorepo

```text
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
│       ├── notebooks/           # Validação visual (analise_dados.ipynb)
│       ├── src/                 # Scripts (Ingestão, Preprocessing, Balancing, Train)
│       └── tests/
├── .env.example
└── docker-compose.yml
```

-----

## Pré-requisitos

Antes de executar o projeto, é necessário ter instalado:

  - Docker e Docker Compose
  - Node.js + pnpm
  - Python 3.11 ou superior
  - [Ollama](https://ollama.com/) (para o módulo de Assistente de Manutenção)

-----

## Como rodar o projeto

### 1\. Clone o repositório

```bash
git clone https://github.com/ronaldosimeone/tcc-project.git
cd tcc-project
```

### 2\. Configure as variáveis de ambiente

```bash
cp .env.example .env
# Ajuste as variáveis (credenciais de banco, URLs do Ollama, etc.)
```

### 3\. Suba os serviços com Docker

```bash
docker-compose up --build
```

-----

## Machine Learning — Setup e Pipeline de Dados

O módulo de ML é isolado do sistema global para garantir que as dependências pesadas de dados (Pandas, Scikit-Learn) não afetem o backend.

### Configuração do Ambiente Virtual (Local)

Para desenvolver ou rodar análises localmente sem sujar o Python global:

```bash
# Na raiz do projeto, crie e ative a venv
python -m venv .venv
.\.venv\Scripts\activate  # No Windows

# Instale as dependências isoladas do módulo de Machine Learning
pip install -r apps/ml/requirements.txt
```

### 1\. Ingestão de Dados (MetroPT-3)

Pipeline de ingestão do dataset [MetroPT-3](https://archive.ics.uci.edu/dataset/791/metropt-3+dataset) (compressor de ar de um trem do metrô do Porto). O script baixa o ZIP da UCI, valida o schema e persiste como `.parquet`.

```bash
python -m apps.ml.src.ingest_metropt
```

### 2\. Engenharia de Features (Suavização de Sinal)

Para alimentar a IA preditiva, os dados brutos ruidosos são convertidos em indicativos dinâmicos de falha através da classe `MetroPTPreprocessor`.
As transformações incluem:

  - **Imputação de Nulos:** `ffill` e `bfill` para cobrir perdas de pacote.
  - **Delta de Pressão (Δp):** Indicativo de eventos anômalos de válvula.
  - **Média Móvel (MA) e Desvio Padrão (std):** Suaviza ruídos e captura a volatilidade do sinal (desgaste mecânico).

### 3\. Balanceamento de Classes (SMOTE)

Datasets industriais são naturalmente desbalanceados (poucas falhas para muitos dados normais). Utilizamos a classe `MetroPTBalancer`, baseada na técnica SMOTE (Synthetic Minority Over-sampling Technique), para gerar amostras sintéticas de falha exclusivas para o conjunto de treinamento, garantindo que o modelo aprenda os padrões de degradação sem viés.

### 4\. Validação Visual (Jupyter)

O pipeline ponta a ponta é validado visualmente através do notebook `apps/ml/notebooks/analise_dados.ipynb`, que comprova a eficácia do balanceamento de classes e a suavização física dos sinais dos sensores.

### Testes Unitários do Módulo ML

Garantem a integridade das matrizes matemáticas e evitam vazamento de dados (*data leakage*):

```bash
pytest apps/ml/tests/ -v
```

-----

## Backend — Setup FastAPI

API REST construída com FastAPI, SQLAlchemy assíncrono e Pydantic v2, seguindo Clean Architecture.

### Rodando o backend localmente

```bash
cd apps/backend

# Crie e ative uma venv exclusiva para o backend
python -m venv .venv
.\.venv\Scripts\activate  # No Windows

# Instale as dependências
pip install -r requirements.txt

# Inicie o servidor
uvicorn src.main:app --reload --port 8000
```

-----

## Frontend — Setup Next.js

Dashboard interativo construído com Next.js 15, React 19 e Tailwind CSS.

### Rodando o frontend localmente

```bash
cd apps/frontend

# Instale as dependências usando pnpm
pnpm install

# Inicie o servidor de desenvolvimento
pnpm dev
```

O painel estará disponível em `http://localhost:3000`.

-----

## Qualidade e Padrões de Código

O repositório é governado por regras estritas descritas no arquivo `CLAUDE.md`.

### Backend & ML (Python)

  - **Tipagem Estrita:** Pydantic v2 e Type Hints.
  - **Formatação e Linting:** Ruff e Black.
  - **Isolamento:** Uso mandatório de `.venv` e arquivos `requirements.txt` independentes.

### Frontend (Next.js/TypeScript)

  - **Tipagem:** TypeScript estrito (proibido `any`).
  - **Arquitetura:** Prioridade para Server Components; Client Components isolados.
  - **UI:** Componentização com Shadcn/ui e TailwindCSS.

-----

## Autores

  - Lucas de Moraes Silveira
  - Raphael Nobuyuki Haga Okuyama
  - Ronaldo Simeone Antonio

-----

## Licença

Este projeto é desenvolvido exclusivamente para fins acadêmicos.

-----