# CLAUDE.md - Contexto e Diretrizes do TCC

## 1. Visão Geral do Sistema
Plataforma full-stack focada em manutenção preditiva para equipamentos industriais (Indústria 4.0). O sistema ingere dados de sensores, realiza predições de falha, transmite leituras em tempo real e aciona um Assistente Inteligente (LLM Local via Ollama + MCP) para sugerir planos de reparo baseados em manuais técnicos.

## 2. Padrões de Código
### Python (Backend & ML)
* **Tipagem:** SEMPRE use Type Hints estritos. NUNCA omita os tipos de retorno.
* **Nomenclatura:** `snake_case` para variáveis/funções/arquivos, `PascalCase` para classes.
* **Validação:** SEMPRE use Pydantic v2 para DTOs (Request/Response).

### TypeScript (Frontend)
* **Tipagem:** PROIBIDO `any`. Prefira `interface` a `type`.
* **Funções:** Use arrow functions. Se >2 parâmetros, use objeto. Prefira *early returns*.
* **Nomenclatura:** `kebab-case` para arquivos. `PascalCase` para Componentes/Classes. `camelCase` para funções.
* **Datas:** EXCLUSIVO `dayjs`.

## 3. Backend (FastAPI & Clean Arch)
* **Arquitetura:** `routers/` (Apenas I/O, Auth e injeção) -> `services/` (Regras de negócio) -> `models/` (SQLAlchemy).
* **Routers:** NUNCA coloque regras de negócio nas rotas. Use o `Depends` do FastAPI para injetar serviços.
* **Erros:** Trate exceções nos `services` lançando erros customizados. Use middlewares globais no FastAPI para retornar JSON padronizado.
* **Tempo Real:** Para SSE e WebSockets, isole a lógica de conexão no `ConnectionManager` ou `SensorStreamService`.

## 4. Machine Learning e IA Local
* **Inferência Preditiva:** Em produção, a API SEMPRE deve carregar os modelos clássicos no formato `ONNX` (Singleton no lifespan do FastAPI). NUNCA treine o modelo na rota da API.
* **Assistente RAG (Ollama):** O sistema utiliza o Ollama rodando localmente (modelo sugerido: `llama3.2` de 3B) para inferência do LLM. O backend gerencia a comunicação com a API do Ollama e com a ferramenta MCP (ChromaDB) para injetar o contexto dos manuais, fazendo o *streaming* da resposta em Markdown para o front.

## 5. Frontend (Next.js 15 & Tempo Real)
* **Server/Client:** PRIORIZE Server Components. Use Client Components apenas quando houver interatividade (gráficos, alertas, streaming).
* **Streaming & Polling:** SEMPRE use hooks customizados (ex: `useSSE`, `useAlertWebSocket`). Otimize renders com `React.memo`.
* **UI & Estilização:**
  * PRIORIZE **Shadcn/ui** (um componente por arquivo). Botões SEMPRE `<Button>`.
  * PROIBIDO cores hard-coded (`bg-black`, `text-white`). Use as variáveis do tema no `@app/globals.css`.
* **Estado de Erro/Loading:** Sempre implemente `Skeleton`, `EmptyState` e `ErrorState`.

## 6. Versionamento e Git
* **Commits:** Use Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`).
* **Ação:** NUNCA commite o código sem a permissão explícita do usuário. Apenas prepare os arquivos.