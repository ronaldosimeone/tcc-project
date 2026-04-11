# 🚀 Projeto TCC - Monorepo

## 📌 Descrição

Projeto de TCC utilizando arquitetura de monorepo com:

- Backend: FastAPI
- Frontend: Next.js
- ML: Python (Jupyter / scripts)
- Banco de dados: PostgreSQL

---

## ⚙️ Pré-requisitos

- Docker e Docker Compose
- Node.js + pnpm
- Python 3.11+

---

## ▶️ Como rodar o projeto

### 1. Clone o repositório

git clone <url-do-repositorio>

### 2. Acesse a pasta

cd tcc-project

### 3. Configure as variáveis de ambiente

Copie o arquivo:
cp .env.example .env

### 4. Suba os serviços

docker-compose up --build

---

## 🌐 Serviços disponíveis

- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- Banco de dados: localhost:5432

---

## 🧪 Padrões do projeto

- Python: Ruff, Black, Mypy
- TypeScript: ESLint, Prettier
- Hooks automáticos com pre-commit

---

## 📁 Estrutura

apps/
backend/
frontend/
ml/
docs/

---

## 👨‍💻 Autor

- Ronaldo Simeone Antonio
- Lucas Silveira
- Raphael Okuyama
