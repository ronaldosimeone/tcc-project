Aqui está o código completo do seu `README.md`, já personalizado com o título oficial do seu TCC e os nomes completos dos autores. 

É só copiar o bloco abaixo e colar no arquivo na raiz do seu projeto:

```markdown
# 🚀 TCC: Sistema Inteligente para Previsão de Falhas e Suporte à Manutenção

[![CI Pipeline](https://github.com/ronaldosimeone/tcc-project/actions/workflows/ci.yml/badge.svg)](https://github.com/ronaldosimeone/tcc-project/actions/workflows/ci.yml)

## 📌 Descrição

Projeto de Trabalho de Conclusão de Curso (Engenharia de Computação - FACENS) focado em uma plataforma integrada para gestão de ativos industriais no contexto da Indústria 4.0. O sistema une manutenção preditiva ao suporte técnico automatizado, utilizando uma arquitetura de monorepo.

O ecossistema é composto por:
- **Backend:** FastAPI
- **Frontend:** Next.js
- **Machine Learning & IA:** Python (Scripts, Notebooks, LLM via MCP)
- **Banco de dados:** PostgreSQL

---

## 🏗️ Arquitetura

O projeto segue o padrão monorepo, no qual múltiplas aplicações e serviços coexistem no mesmo repositório, cada um com suas responsabilidades bem definidas e orquestrados de forma unificada.

### Tecnologias utilizadas

- **Backend:** Python, FastAPI, SQLAlchemy, Pydantic, Celery, Redis.
- **Frontend:** Next.js 15 (App Router), React 19, TypeScript, Tailwind CSS, Shadcn/ui.
- **Machine Learning:** Scikit-learn, XGBoost, ONNX Runtime, ChromaDB, Anthropic SDK.
- **Infraestrutura:** Docker e Docker Compose, Nginx.

---

## ⚙️ Pré-requisitos

Antes de executar o projeto, é necessário ter instalado:

- Docker
- Docker Compose
- Node.js
- pnpm
- Python 3.11 ou superior

---

## ▶️ Como rodar o projeto

### 1️⃣ Clone o repositório

```bash
git clone [https://github.com/ronaldosimeone/tcc-project.git](https://github.com/ronaldosimeone/tcc-project.git)
```

### 2️⃣ Acesse a pasta do projeto

```bash
cd tcc-project
```

### 3️⃣ Configure as variáveis de ambiente

Copie o arquivo de exemplo e ajuste as variáveis (como chaves de API e credenciais de banco) conforme necessário para o seu ambiente:

```bash
cp .env.example .env
```

### 4️⃣ Suba os serviços com Docker

```bash
docker-compose up --build
```

---

## 🌐 Serviços disponíveis

Após a inicialização, os serviços estarão acessíveis nos seguintes endereços:

- **Frontend (Dashboard):** http://localhost:3000
- **Backend (API):** http://localhost:8000
- **Documentação da API (Swagger):** http://localhost:8000/docs
- **Banco de dados (PostgreSQL):** localhost:5432

---

## 🧪 Qualidade e padrões de código

O projeto adota boas práticas rigorosas de desenvolvimento e automação de qualidade de código.

### Backend (Python)
- **Ruff:** Linting ultra-rápido.
- **Black:** Formatação de código.
- **Mypy:** Verificação de tipagem estática.

### Frontend (TypeScript)
- **ESLint:** Linting e regras de React.
- **Prettier:** Formatação de código.

### Automação (Pre-commit)
Para garantir a execução automática de todos os checks antes de cada commit, instale o pre-commit localmente após clonar o repositório:
```bash
pip install pre-commit
pre-commit install
```

---

## 🧠 Observações importantes

- Cada aplicação (`backend` e `ml`) possui seu próprio ambiente virtual (`venv`) e arquivo `requirements.txt`.
- As dependências de Python não são versionadas diretamente, apenas listadas nos arquivos de requisitos.
- Os diretórios `venv/` e `node_modules/` estão no `.gitignore` e não devem ser versionados.
- O pipeline de CI/CD está configurado via GitHub Actions para barrar pull requests que não passem nos testes ou linters.

---

## 👨‍💻 Autores

- Lucas de Moraes Silveira
- Raphael Nobuyuki Haga Okuyama
- Ronaldo Simeone Antonio  

---

## 📄 Licença

Este projeto é desenvolvido exclusivamente para fins acadêmicos.
```