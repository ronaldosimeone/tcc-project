# RNF-62 / RNF-63 — Remoção do bloqueio de segurança do `apps/mcp-server`

Continuação direta do RNF-62/63. O relatório anterior fechou em **PASS COM RESSALVAS**
por um bloqueio objetivo: `apps/mcp-server` ainda tinha vulnerabilidades HIGH em
`transformers`. Esta task **elimina** de fato os High/Critical de produção do MCP —
sem `--ignore-vuln`, sem `continue-on-error`, sem "é serviço interno" —, preservando o
pipeline `Backend → MCP Server → vector store → Sentence Transformers → modelo`.

---

## 1. Auditoria inicial (AUDIT-FIRST)

`pip-audit -r apps/mcp-server/requirements.txt` numa imagem `python:3.11-slim` limpa
(mesmo Python do Dockerfile). Requirements originais:
`mcp==2.2.0`, `pytest==9.1.1`, `pypdf==5.1.0`, `sentence-transformers==3.3.1`, `chromadb==0.5.23`.

### Cadeia de dependências

```
chromadb==0.5.23        ->  tokenizers>=0.13.2,<=0.20.3   (TETO — travava o upgrade de transformers)
transformers==4.46.3    ->  tokenizers>=0.20,<0.21
  └─ sentence-transformers==3.3.1  ->  transformers>=4.41.0,<5.0.0
pypdf==5.1.0
(torch, huggingface-hub, safetensors, numpy  — transitivos)
```

### Vulnerabilidades encontradas (pip-audit)

| Pacote | Versão | Qtd | Severidade | Correção disponível |
| --- | --- | --- | --- | --- |
| **chromadb** | 0.5.23 | 3 | **1 Critical + 2 High (8.8)** — `CVE-2026-45833` / `CVE-2026-45830` / `CVE-2026-45831` (code injection / RBAC no servidor HTTP) | **NENHUMA.** `patched: None`. Faixa afetada `>= 0.4.17, <= 1.5.9` — **todas** as versões publicadas. A última (1.5.9) ainda ADICIONA `CVE-2026-45829` (Critical, pré-autenticação). |
| **transformers** | 4.46.3 | 26 | RCE via arquivo de modelo malicioso (High), ReDoS no tokenizer (Medium), … — 8 sem correção publicada no 4.46.3 | `5.17.0` = 0 CVEs |
| **pypdf** | 5.1.0 | 41 | 2 High (loop infinito em imagem inline não terminada) + ~39 DoS Medium/Low | todas com fix (`>= 6.0.0`) |

Os 4 CVEs do `chromadb` estão **exclusivamente no componente servidor HTTP** (`chroma run`
/ FastAPI / `SimpleRBACAuthorizationProvider`). O mcp-server usa só o cliente embutido
(`PersistentClient`), que nunca sobe servidor. Ainda assim o RNF-62 exige removê-los da
árvore de produção — não argumentar alcançabilidade.

### O conflito informado antes NÃO existe mais

`chromadb 1.5.9` exige só `tokenizers>=0.13.2` (sem teto). Sem `chromadb`, a cadeia
moderna resolve limpa (`pip check` OK): `sentence-transformers 6.0.1` + `transformers
5.17.0` + `tokenizers 0.23.2` + `torch 2.14.0`.

### Estratégias (§3)

- **A — upgrade compatível**: impossível. Não existe versão do `chromadb` sem High/Critical.
- **B — upgrade em cascata**: resolve transformers/tokenizers/torch, mas `chromadb` continua
  Critical+High (subir para 1.5.9 piora). Não elimina o bloqueio.
- **C — substituir dependência**: única saída para `pip-audit` = 0 High/0 Critical sem
  `--ignore-vuln`. Adotada para o `chromadb`; a cadeia `transformers` foi resolvida com o
  upgrade em cascata (agora sem conflito).

---

## 2. Remediação

| Pacote | Antes | Depois | Vulnerabilidade | Ação |
| --- | --- | --- | --- | --- |
| `chromadb` | `0.5.23` | **removido** | `CVE-2026-45833` (Critical), `CVE-2026-45830` / `CVE-2026-45831` (High 8.8) — sem correção em nenhuma versão | Substituído por [`vector_store.py`](vector_store.py) — `sqlite3` (stdlib) + `numpy` (já presente via `sentence-transformers`), **zero dependência nova**. Mesmo contrato (`PersistentClient` / `get_or_create_collection` / `add` / `upsert` / `query(include=[...,"embeddings"])` / `get(where=)` / `count` / `delete`), mesma dimensão 384, mesma distância L2 para candidatos. |
| `sentence-transformers` | `3.3.1` | `6.0.1` | (puxava `transformers` 4.46.3) | upgrade em cascata |
| `transformers` | `4.46.3` | `5.17.0` | 26 CVEs (RCE modelo malicioso High, ReDoS Medium, …) | upgrade — `5.17.0` = 0 CVEs |
| `tokenizers` | `0.20.3` | `0.23.2` | — | acompanha transformers 5.x |
| `torch` | transitivo | `2.14.0` (pin) | `< 2.14` tinha High | pin — 0 CVEs |
| `pypdf` | `5.1.0` | `6.18.0` | 2 High (DoS) + ~39 Medium/Low | upgrade — 0 CVEs |
| `pytest` | em `requirements.txt` | movido p/ `requirements-dev.txt` | — | tira teste da imagem de produção (paralelo ao backend) |

**Modelo de embeddings inalterado**: `paraphrase-multilingual-MiniLM-L12-v2`, 384 dim,
CPU-only, `trust_remote_code` nunca ligado, nenhum modelo vindo de entrada de usuário.
Env `CHROMA_DB_PATH` e diretório `data/chroma` mantidos (compat); o diretório agora
guarda `vector_store.sqlite3`.

### Reindexação (§6)

`data/chroma/` (formato ChromaDB 0.5.23) apagado e reconstruído por `python
index_manuals.py` com a stack nova. Validado:

| Métrica | Antes (chromadb 0.5.23) | Depois (vector_store.py) |
| --- | --- | --- |
| chunks | 9 (3/manual) | **9 (3/manual)** |
| dimensão | 384 | **384** |
| idempotência (2ª execução) | skip 3 / indexed 0 | **skip 3 / indexed 0** |
| "vazamento na bomba centrifuga" | 0.71 / 0.70 / 0.63 | **0.7108 / 0.7006 / 0.6337** |
| `avg_results/query` (24 queries) | 0.75 | **0.75** |

Os scores e o comportamento do RAG são **idênticos** — a troca do armazenamento não muda
o resultado da busca.

---

## 3. Validação

Ambiente: `docker compose build mcp-server` + `docker compose up -d mcp-server` (imagem de
produção, `requirements.txt`, `python:3.11-slim`).

| Item | Resultado | Evidência |
| --- | --- | --- |
| **pip-audit — mcp-server** (`-r requirements.txt`) | **PASS** | 82 dependências resolvidas, `No known vulnerabilities found`. Sem `--ignore-vuln`. |
| **pip-audit — backend** (`-r requirements.txt`) | **PASS** | `No known vulnerabilities found, 1 ignored` — `PYSEC-2026-3740` (nltk, Medium, sem fix, não alcançável, já documentado; inalterado por esta task). |
| **pnpm audit — frontend** (`--prod --audit-level high`) | **PASS** | `No known vulnerabilities found`. |
| **pip check** (mcp-server) | **PASS** | `No broken requirements found`. |
| **pytest** (`cd apps/mcp-server && pytest`) | **PASS** | `48 passed` (test_semantic_search + test_mcp_server + test_index_manuals). |
| **Startup do MCP** (Docker) | **PASS** | `Uvicorn running on http://0.0.0.0:8100`, `StreamableHTTP session manager started`, sem erro. |
| **Carga do modelo de embeddings** | **PASS** | `paraphrase-multilingual-MiniLM-L12-v2`, 384 dim, `device: cpu`, cold-start ~48 s (< `MCP_CLIENT_TIMEOUT_SECONDS=60`). |
| **Vector store / indexação** | **PASS** | 9 chunks, 384 dim, 3 por manual, backend = `vector_store`. |
| **Idempotência** | **PASS** | 2ª execução: `indexed=0 skipped=3`, count continua 9. |
| **Busca semântica (queries determinísticas)** | **PASS** | "vazamento na bomba centrifuga" → bomba `0.71/0.70/0.63` (== baseline); "cavitacao e ruido na succao" → bomba `0.72/0.71/0.68`; "vibracao e temperatura do enrolamento do motor" → motor `0.68/0.62`; "trocar o oleo do compressor CX-500" → compressor `0.68`; "receita de bolo de chocolate" → `[]`. |
| **MCP protocol over HTTP** | **PASS** | `curl` real na rede Docker: `initialize` → 200 + `mcp-session-id`; `tools/list` → schema `search_maintenance_manual(query: string)`; `tools/call` → `isError:false`, `structuredContent` com 3 resultados. |
| **Backend → MCP (chamada real, sem mock)** | **PASS** | `MCPSearchClient` real (código do backend, `mcp==2.2.0`, transporte `streamable-http`) → mcp-server real → 3 resultados, scores idênticos; query irrelevante → `[]`. |
| **Fluxo de manutenção** (falha → sugestão → MCP → referências) | **PASS** | `MaintenanceSuggestionService.suggest()` real com `failure_probability=0.92` → MCP real → 2 context chunks → prompt contendo os trechos do manual → 2 `ManualReference` (file_name, page, score). Passo Ollama com dublê (Ollama não está de pé neste ambiente; a task não toca esse passo). |
| **Benchmark RNF-45** | **PASS** | p50 22 ms / p95 38 ms / p99 47 ms (alvo < 500 ms), `avg_results/query = 0.75` (== baseline). |
| **CI — security audit** | **PASS** | `continue-on-error` removido do step do mcp-server; agora `pip-audit -r apps/mcp-server/requirements.txt` sem flags → quebra o build em qualquer High/Critical, igual a backend/frontend. |
| **pytest — backend (regressão)** | **PASS (com 1 falha pré-existente não relacionada)** | `376 passed, 1 failed, 1 xfailed, 34 skipped`. A falha (`test_drift_monitor.py::test_task_runs_directly_without_http`, RF-27/RNF-55, `insufficient_data` — seeding de linhas no SQLite do teste de drift) **reproduz idêntica no HEAD limpo** (`git stash` de todas as mudanças) — nenhuma relação com o mcp-server. O xfail é o `xgboost_v1` pré-existente já documentado em `PENDENCIAS.md`. |
| **frontend — `pnpm test` / `pnpm build`** | não re-executado | **Zero arquivos de frontend alterados nesta task** — validado no RNF-62/63 anterior. |
| **OWASP ZAP baseline** | 0 High / 0 Critical | RNF-62/63 anterior. O mcp-server é interno (`docker-compose.yml`: `expose`, sem `ports:`, não roteado pelo Nginx) — a superfície que o ZAP varre (Nginx → API/Frontend) **não mudou** nesta task. |

**Estado final exigido:**

```
Backend produção:     0 High, 0 Critical   ✅
Frontend produção:    0 High, 0 Critical   ✅
MCP Server produção:  0 High, 0 Critical   ✅
OWASP ZAP:            0 High, 0 Critical   ✅
```

---

## 4. Alterações

| Arquivo | Tipo | O quê |
| --- | --- | --- |
| `apps/mcp-server/requirements.txt` | M | remove `chromadb` e `pytest`; adiciona `transformers`/`tokenizers`/`torch` pinados; `pypdf` 5.1.0 → 6.18.0; `sentence-transformers` 3.3.1 → 6.0.1; comentário RNF-62 |
| `apps/mcp-server/requirements-dev.txt` | **NEW** | `-r requirements.txt` + `pytest==9.1.1` |
| `apps/mcp-server/vector_store.py` | **NEW** | store embutido `sqlite3` + `numpy` (~330 linhas) com o contrato do chromadb usado pelo mcp-server |
| `apps/mcp-server/index_manuals.py` | M | `import chromadb` → `import vector_store`; `chromadb.PersistentClient` → `vector_store.PersistentClient`; docstrings/comentários |
| `apps/mcp-server/semantic_search.py` | M | docstring (métrica L2/cosine agora no `vector_store.py`; scores re-medidos) |
| `apps/mcp-server/server.py` | M | comentário/docstring |
| `apps/mcp-server/benchmark_semantic_search.py` | M | docstring |
| `apps/mcp-server/semantic_search_benchmark.json` | M | re-gerado (p95 18 ms → 38 ms; ainda 13× abaixo do alvo) |
| `apps/mcp-server/data/chroma/` | (gitignored) | reindexado — `vector_store.sqlite3` |
| `.github/workflows/ci.yml` | M | remove `continue-on-error: true` do audit do mcp-server; ajusta comentário da política |
| `README.md` | M | §17.1 (tabela de remediação + política), §17.5 (CI), §4.6/§4.7/§4.8/§4.13 (nomenclatura vector store), §16 (remove item de troubleshooting obsoleto do posthog/chromadb; ajusta tamanho da imagem) |
| `docker-compose.yml` | M | 1 comentário (`ChromaDB` → `vector store`) |

Fora de escopo, **não tocados**: frontend, Storybook, React, DVC, treino de ML, Drift
Monitoring, fila médica, notificações, arquitetura do backend, thresholds de ML, regras
de negócio.

---

## 5. Ressalvas

1. **`nltk` `PYSEC-2026-3740` (Medium, backend)** — sem correção publicada; path traversal
   só em `nltk.data.load`/`download` com caminho controlado pelo atacante; dataset
   MetroPT-3 100% numérico, `evidently` nunca chama o caminho de texto. `--ignore-vuln`
   explícito no CI (não esconde High/Critical). **Inalterado por esta task** — herdado do
   RNF-62/63 anterior, documentado em `PENDENCIAS.md` e README §17.
2. **Imagem do mcp-server ~6–7 GB** — `torch` da PyPI padrão traz dependências CUDA
   (`nvidia-*`, ~2 GB) num container CPU-only. Não é vulnerabilidade (`pip-audit` limpo).
   A remoção do `chromadb` já cortou ~40 pacotes transitivos (`onnxruntime`, `kubernetes`,
   `grpcio`, `opentelemetry-*`, `posthog`, …). Otimização CPU-only do `torch` documentada
   como follow-up (README §16) — não feita para não re-validar toda a stack.
3. **`vector_store.query` abre conexão sqlite + `CREATE TABLE IF NOT EXISTS` por chamada**
   → p95 subiu de ~18 ms para ~38 ms. 13× abaixo do alvo RNF-45 (500 ms). Cache de conexão
   é a otimização óbvia se o corpus algum dia crescer muito.
4. **1 teste de backend falhando** (`test_drift_monitor.py::test_task_runs_directly_without_http`,
   RF-27/RNF-55) — **pré-existente**, reproduz idêntico no HEAD limpo, sem relação com o
   mcp-server (é seeding de linhas de sensor no SQLite do próprio teste). Não introduzido
   nem corrigido aqui (fora de escopo).
5. **Passo Ollama do fluxo de manutenção** validado com dublê — Ollama não está de pé
   neste ambiente. Todo o resto do fluxo (threshold → MCP real → embeddings reais →
   vector store real → referências → montagem do prompt) é real. A task não toca o passo
   Ollama.

---

## 6. Veredito

# PASS

Os 4 componentes (Backend, Frontend, MCP Server, OWASP ZAP) estão com **0 High / 0
Critical** em produção. O `pip-audit` do mcp-server passa **sem `--ignore-vuln` e sem
`continue-on-error`**. O MCP sobe, o modelo carrega, o vector store funciona, a busca
semântica é coerente (scores **idênticos** ao baseline), a chamada real Backend→MCP
funciona e o fluxo de manutenção monta as referências corretamente. Os testes do
mcp-server passam (48/48). O CI passou a falhar o build em qualquer High/Critical do MCP.

O blocker do RNF-62 foi **efetivamente eliminado**, não mascarado.

> **Não commitado / não pushado** (conforme §17 do enunciado). Todas as mudanças estão
> na working tree.
