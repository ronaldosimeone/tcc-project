# 🚀 Projeto TCC - Monorepo

## 📌 Descrição

Projeto de TCC utilizando arquitetura de monorepo com:

- Backend: FastAPI
- Frontend: Next.js
- ML: Python (Jupyter / scripts)
- Banco de dados: PostgreSQL

---

## 🏗️ Arquitetura

O projeto segue o padrão monorepo, no qual múltiplas aplicações e serviços coexistem no mesmo repositório, cada um com suas responsabilidades bem definidas.

### Tecnologias utilizadas

- Backend: FastAPI (Python)
- Frontend: Next.js (React + TypeScript)
- Machine Learning: Python (scripts e notebooks)
- Banco de dados: PostgreSQL
- Infraestrutura: Docker e Docker Compose

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

git clone https://github.com/ronaldosimeone/tcc-project.git

### 2️⃣ Acesse a pasta do projeto

cd tcc-project

### 3️⃣ Configure as variáveis de ambiente

Copie o arquivo de exemplo:

cp .env.example .env

Ajuste as variáveis conforme necessário para o seu ambiente.

### 4️⃣ Suba os serviços com Docker

docker-compose up --build

---

## 🌐 Serviços disponíveis

Após a inicialização, os serviços estarão acessíveis nos seguintes endereços:

- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- Banco de dados (PostgreSQL): localhost:5432

---

## 🧪 Qualidade e padrões de código

O projeto adota boas práticas de desenvolvimento e automação de qualidade de código.

### Backend (Python)
- Ruff (linting)
- Black (formatação)
- Mypy (verificação de tipos)

### Frontend (TypeScript)
- ESLint (linting)
- Prettier (formatação)

### Automação
- pre-commit para execução automática de checks antes dos commits

---

## 🧠 Observações importantes

- Cada aplicação (backend e ml) possui seu próprio ambiente virtual (venv) e arquivo requirements.txt.
- Dependências não são versionadas diretamente, apenas listadas.
- Os diretórios venv/ não são versionados e devem ser criados localmente.
- O projeto está preparado para execução local e futura expansão para ambientes de produção.

---

## 👨‍💻 Autores

- Ronaldo Simeone Antonio  
- Lucas Silveira  
- Raphael Okuyama  

---

## 📄 Licença

Este projeto é desenvolvido exclusivamente para fins acadêmicos.
