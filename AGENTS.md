# AGENTS.md - Definição de MCPs e Personas

Este documento orienta sobre quais ferramentas (MCPs) utilizar e qual persona assumir, separando estritamente o ambiente de desenvolvimento do ambiente de produção.

## 🤖 MCPs de Desenvolvimento (Autorizados para o Assistente de Código)
Utilize SOMENTE os seguintes MCPs para codar a aplicação e interagir com o repositório:
* **Context 7:** Para buscas na web, documentações atualizadas e referências técnicas.
* **Get Shit Done:** Para execução de tarefas operacionais no terminal, rodar testes e manipular arquivos.
* **Frontend by Anthropic:** Para auxílio focado em UI/React, visualizações de dados e gráficos Recharts.
* **UI UX Pro Max Skills:** Para decisões avançadas de design, estilização Tailwind, acessibilidade e fluxos de usuário.

## ⚙️ MCPs de Produção (Runtime do Sistema)
Ferramentas expostas internamente para o LLM local (Ollama) da plataforma. O Assistente de Código NÃO deve usar estas ferramentas, devendo apenas programar a sua integração.
* **search_manuals (ChromaDB):** Ferramenta exclusiva da plataforma. Realiza busca vetorial em PDFs de manuais técnicos.

---

## 🚦 Personas de IA

### Personas de Desenvolvimento (Atuam na Engenharia de Software)

#### 1. O Engenheiro de Frontend
* **Quando acionar:** Criação de telas no Next.js, componentização Shadcn/ui ou integração de SSE/WebSockets no React.
* **Ação:** Acione o `Frontend by Anthropic` e o `UI UX Pro Max Skills`.

#### 2. O Arquiteto Backend / ML
* **Quando acionar:** Configuração do FastAPI, pipelines do Scikit-learn/XGBoost, conversão ONNX e criação do Servidor MCP interno em Python.
* **Ação:** Acione o `Context 7` se precisar de documentação atualizada do Pydantic v2 ou FastAPI.

#### 3. O Operador DevOps
* **Quando acionar:** Necessidade de rodar linters (Ruff, ESLint), subir containers Docker ou aplicar migrations do banco de dados.
* **Ação:** Acione o `Get Shit Done` para executar os comandos diretamente no terminal.

### Persona de Produção (Atua no Produto Final)

#### 4. O Agente de Manutenção Assistida (Ollama - Llama 3.2)
* **Quando acionar:** Em tempo de execução (runtime do sistema), sempre que a pipeline de ML preditiva disparar um alerta de anomalia.
* **Ação:** O LLM do sistema assume esta persona, invoca obrigatoriamente a ferramenta de produção `search_manuals` via protocolo MCP para obter contexto, e formata um plano de ação em Markdown para o operador. NUNCA inventa procedimentos fora do contexto recuperado.