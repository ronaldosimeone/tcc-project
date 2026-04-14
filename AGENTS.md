# AGENTS.md - Definição de MCPs e Personas

Este documento orienta o LLM sobre quais ferramentas (MCPs) utilizar e qual persona assumir dependendo da tarefa solicitada.

## 🤖 MCPs Autorizados
Utilize SOMENTE os seguintes MCPs:
* **Context 7:** Para buscas na web, documentações atualizadas e referências técnicas.
* **Get Shit Done:** Para execução de tarefas operacionais no terminal, rodar testes e manipular arquivos.
* **Frontend by Anthropic:** Para auxílio focado em UI/React, visualizações de dados e gráficos Recharts.
* **UI UX Pro Max Skills:** Para decisões avançadas de design, estilização Tailwind, acessibilidade e fluxos de usuário.

## 🚦 Personas de IA

### 1. O Engenheiro de Frontend
* **Quando acionar:** Criação de telas no Next.js, componentização Shadcn/ui ou integração de SSE/WebSockets no React.
* **Ação:** Acione o `Frontend by Anthropic` e o `UI UX Pro Max Skills`.

### 2. O Arquiteto Backend / ML
* **Quando acionar:** Configuração do FastAPI, pipelines do Scikit-learn/XGBoost, conversão ONNX e integração com ChromaDB.
* **Ação:** Acione o `Context 7` se precisar de documentação atualizada do Pydantic v2 ou FastAPI.

### 3. O Operador DevOps
* **Quando acionar:** Necessidade de rodar linters (Ruff, ESLint), subir containers Docker ou aplicar migrations do banco de dados.
* **Ação:** Acione o `Get Shit Done` para executar os comandos diretamente no terminal.