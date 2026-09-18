# RNF-62 / RNF-63 — FECHAMENTO DEFINITIVO

**Data:** 2026-09-18. Continuação da re-auditoria do mesmo dia — o relatório anterior
fechou em **PASS COM RESSALVAS** por um único blocker real: `evidently → nltk →
PYSEC-2026-3740` (High, sem correção publicada). Esta rodada **elimina esse blocker
do runtime de produção** (não o mascara) e reobtém `PASS` real nos dois requisitos.

---

## A. Blocker original — como foi eliminado

```
evidently==0.7.21 (apps/backend/requirements.txt, único uso: RF-27/RNF-55 — drift_monitor.py)
  ↓ dependência OBRIGATÓRIA e IMEDIATA (evidently/__init__.py importa nltk no import mais simples)
nltk>=3.6.7
  ↓
PYSEC-2026-3740 / GHSA-8mgp-746c-j5xp / CVE-2026-81726 — High (CVSS 3.1: 7.0, CVSS 4.0: 8.3)
  sem correção publicada em NENHUMA versão do nltk (3.10.3, a mais recente do PyPI, ainda afetada)
```

**Investigação (audit-first, nada assumido)**

1. `grep -rl "evidently"` em todo o repo → **só 2 arquivos**: `src/services/drift_monitor.py`
   (produção — chamado por `src/tasks/drift_tasks.py`, uma task Celery Beat real, e por
   `src/routers/monitoring.py`, `GET /monitoring/drift`) e `tests/test_drift_monitor.py`.
   `apps/ml` não usa `evidently`/`nltk` em nada.
2. API real usada em produção: só `evidently.Dataset`, `DataDefinition`, `Report`,
   `evidently.metrics.ValueDrift(method="psi")` — cálculo de **PSI numérico**, nada de
   NLP/texto.
3. **Tentativa de isolamento (manter `evidently`, remover só `nltk` do runtime)**:
   testado empiricamente — `pip uninstall nltk` seguido de `from evidently import
   Dataset` produz `ModuleNotFoundError: No module named 'nltk'`. Causa raiz:
   `evidently/__init__.py` importa, na cadeia, `evidently.legacy.features.
   OOV_words_percentage_feature`, que faz `from nltk.corpus import words` — **import
   incondicional, executado mesmo que a aplicação nunca use nenhuma feature de
   texto**. Confirmado: **não é possível manter `evidently` instalado sem `nltk`.**
   Estratégia de isolamento descartada.
4. **Substituição da funcionalidade (adotada)**: a única coisa que `evidently` fazia
   aqui — PSI por feature — foi reimplementada com `numpy`/`pandas` (zero dependência
   nova). Fonte real do algoritmo do `evidently` inspecionada
   (`evidently.legacy.calculations.stattests.psi._psi()` +
   `.utils.get_binned_data()`): bins pela regra de Sturges
   (`np.histogram_bin_edges(reference+current combinados, bins="sturges")`),
   preenchimento de bins vazios por epsilon, `PSI = Σ (ref% - cur%) · ln(ref%/cur%)`.
   Reimplementada em `apps/backend/src/services/drift_monitor.py::
   _population_stability_index` (~35 linhas, função pura).
5. **Validação de equivalência** — comparação direta `evidently` real vs. reimplementação
   em 3 cenários sintéticos (distribuições iguais, deslocadas, assimétricas tipo gama),
   7 features cada:

   ```
   same_dist:  TP2 evidently=0.109185  mine=0.109185  diff=0.00000000
   shifted:    TP2 evidently=9.632074  mine=9.632074  diff=0.00000000
   gamma:      TP2 evidently=0.215129  mine=0.215129  diff=0.00000000
   (mesma igualdade nas 7 features × 3 cenários — MAX DIFF = 0.0000000000)
   ```

6. `evidently==0.7.21` removido de `apps/backend/requirements.txt`.

**Confirmação na imagem real de produção** (não só no `requirements.txt`):

```bash
$ docker compose build api celery-worker celery-beat && docker compose up -d ...
$ docker compose exec api pip list | grep -iE "nltk|evidently"
(sem saída — AUSENTES)

$ docker compose exec api pip-audit          # modo ambiente real, não -r arquivo
No known vulnerabilities found

$ pip-audit -r apps/backend/requirements.txt  # modo arquivo, resolução completa
No known vulnerabilities found                # RC=0 — SEM --ignore-vuln

$ pipdeptree | grep -i nltk
(sem saída — nltk não entra por NENHUM outro caminho transitivo)
```

**Validação funcional real (produção, sem mock)**: task Celery disparada manualmente
contra Postgres/Redis reais do `docker compose` — `current_rows=356`,
`psi=7.284299057867684`, `drift_detected=true`, `features={"TP2": 0.228, "TP3": 0.197,
"H1": 0.425, "DV_pressure": 0.109, "Reservoirs": 0.199, "Oil_temperature": 7.28,
"Motor_current": 0.65}`, persistido em `drift_reports` (INSERT...ON CONFLICT DO UPDATE
real, COMMIT real), refletido em `GET /api/monitoring/drift` através do Nginx real.
`celery-worker` registra `[tasks] . monitoring.daily_drift_analysis` normalmente;
`celery-beat` inicia normalmente.

---

## B. Dependências finais

| Componente | Audit | High | Critical | Status |
| --- | ---: | ---: | ---: | --- |
| Backend | `pip-audit -r requirements.txt` (sem `--ignore-vuln`) | 0 | 0 | **PASS** |
| Backend (ambiente real, `docker compose exec api pip-audit`) | modo ambiente | 0 | 0 | **PASS** |
| Frontend produção | `pnpm audit --prod --audit-level high` | 0 | 0 | **PASS** |
| MCP | `pip-audit -r requirements.txt` (82 deps, 0 vulns de qualquer severidade) | 0 | 0 | **PASS** |

Nenhum `--ignore-vuln` em nenhum dos 3 audits de produção. Nenhum `continue-on-error`
no CI.

---

## C. ZAP

Resultado da varredura desta sessão (antes da remoção do `evidently` — a mudança é
100% backend/lógica de negócio, não toca rotas HTTP, headers, autenticação nem
frontend, então o resultado HTTP-level é inalterado; não refeito após a remoção por
não haver superfície nova a varrer):

```
FAIL-NEW: 0
High: 0
Critical: 0
Medium: 12   (4 variantes da mesma limitação de CSP unsafe-inline/unsafe-eval/wildcard
              já documentada — next dev; ver README §17.4)
Low: 11      (Cross-Origin-Embedder-Policy ausente ×4; Dangerous JS Functions ×2;
              Timestamp Disclosure ×5 — todos em chunks minificados do Next.js/
              Turbopack, não em código da aplicação)
Info: 30     (comentários/cache-headers em assets estáticos do build — nenhum segredo real)
```

---

## D. RNF-63

```
Sanitizer:     apps/backend/src/core/log_sanitizer.py — processor structlog
               (redact_sensitive), plugado antes do renderer em src/core/logging.py
Testes:        apps/backend/tests/test_log_privacy.py — 17/17 PASS
PII exposta:   NÃO — e-mail, telefone, endereço, IP tratado, Telegram chat_id
               mascarados/hasheados na linha JSON efetivamente escrita
Secrets expostos: NÃO — Authorization/Bearer, API key, senha, cookie, DSN com
               credencial, Telegram bot token — todos redigidos ("***" ou mascarado)
```

Inalterado nesta rodada (não regrediu — reverificado).

---

## E. Testes (resultados reais desta sessão)

```
Backend pytest:         376 passed, 1 failed (pré-existente, ver F), 1 xfailed
                         (pré-existente, documentado), 34 skipped — idêntico ao
                         baseline ANTES da remoção do evidently
MCP pytest:              48 passed
Frontend tests:          299 passed, 23 failed (pré-existente, ver F)
Frontend build:          ✓ Compiled successfully
pip-audit (backend):     No known vulnerabilities found — RC=0, SEM --ignore-vuln
pip-audit (mcp-server):  No known vulnerabilities found — RC=0
pnpm audit (frontend):   No known vulnerabilities found
ZAP:                     FAIL-NEW 0 · WARN-NEW 7 · PASS 60 (0 High/Critical)
ruff (backend):          195 erros pré-existentes em arquivos não tocados (EXE002 —
                         metadado de execução do Git/FS — e imports pré-existentes;
                         ferramenta não pinada no CI, ruleset evolui com o tempo).
                         Arquivos desta e da sessão anterior (log_sanitizer.py,
                         logging.py, auth.py, main.py, test_log_privacy.py,
                         drift_monitor.py, test_drift_monitor.py) — 0 erros de
                         código real; restam só EXE002 (metadado, repo inteiro) e
                         1 noqa de convenção usada em 5 arquivos do projeto
mypy (backend):          32 erros pré-existentes — todos import-untyped (stubs
                         ausentes para pandas/joblib/onnxruntime/sklearn/celery/
                         pyarrow, ambiente) + 1 erro pré-existente em config.py.
                         Nenhum nos arquivos desta task
import-linter (backend): 1 kept, 0 broken (RNF-56 — intacto)
```

---

## F. Regressões — comparação HEAD limpo vs. HEAD alterado

| | HEAD limpo (`git stash`, confirmado nesta sessão e na anterior) | HEAD com RNF-62/63 completo |
| --- | --- | --- |
| Backend pytest | 1 falha (`test_drift_monitor.py::test_task_runs_directly_without_http` — `insufficient_data`/seeding, sem relação com `evidently`/PSI) | **A MESMA falha, idêntica** — 376 passed / 1 failed / 1 xfailed |
| Frontend tests | 23 falhas (5 arquivos: `loading-states`, `sensor-monitor`, `connection-resilience`, `use-alert-websocket`, `use-sensor-data`) | **As mesmas 23 falhas, idênticas** — 299 passed / 23 failed |
| **Novas falhas introduzidas** | — | **0** |

A falha de `test_task_runs_directly_without_http` foi re-executada explicitamente
ANTES e DEPOIS de remover `evidently` (mesmo comando, mesmo container) — resultado
idêntico nos dois casos, confirmando que a causa é o seeding de linhas do próprio
teste, não o cálculo de PSI.

---

## G. Arquivos alterados (sessão completa RNF-62/RNF-63, incluindo o fechamento de hoje)

```
.env.example
.github/workflows/ci.yml                (remove --ignore-vuln PYSEC-2026-3740; comentário atualizado)
.gitignore                              (apps/backend/.audit_tmp/ ignorado)
README.md                               (§4.15 nova subseção "PSI sem evidently"; §17.1/§17.5 corrigidos)
PENDENCIAS.md                           (item nltk/evidently marcado RESOLVIDO com evidência completa)
docker-compose.yml

apps/backend/Dockerfile
apps/backend/.dockerignore              (NOVO)
apps/backend/requirements.txt           (evidently==0.7.21 REMOVIDO)
apps/backend/requirements-dev.txt       (NOVO)
apps/backend/src/core/auth.py
apps/backend/src/core/logging.py
apps/backend/src/core/log_sanitizer.py  (NOVO)
apps/backend/src/main.py
apps/backend/src/services/drift_monitor.py  (evidently -> _population_stability_index própria)
apps/backend/tests/test_alert_settings.py
apps/backend/tests/test_drift_monitor.py    (2 testes renomeados: evidently_real -> psi_real; docstrings)
apps/backend/tests/test_log_privacy.py  (NOVO, 17 testes)

apps/frontend/.dockerignore             (NOVO)
apps/frontend/package.json
apps/frontend/pnpm-lock.yaml

apps/mcp-server/requirements.txt
apps/mcp-server/requirements-dev.txt    (NOVO)
apps/mcp-server/vector_store.py         (NOVO — substitui chromadb)
apps/mcp-server/index_manuals.py
apps/mcp-server/semantic_search.py
apps/mcp-server/server.py
apps/mcp-server/benchmark_semantic_search.py
apps/mcp-server/semantic_search_benchmark.json
apps/mcp-server/RNF-62-mcp-remediacao.md
infra/nginx/nginx.conf
```

---

## H. Veredito

# RNF-62: PASS
# RNF-63: PASS

**RNF-62** — todos os critérios objetivos atendidos, sem exceção e sem mascaramento:

```
Backend  → 0 High / 0 Critical  — pip-audit SEM --ignore-vuln (nltk saiu da árvore)
Frontend → 0 High / 0 Critical  — pnpm audit limpo
MCP      → 0 High / 0 Critical  — 0 vulnerabilidades de qualquer severidade
ZAP      → 0 High / 0 Critical  — baseline real contra a stack rodando
```

Nenhum `--ignore-vuln`, nenhum `continue-on-error`, nenhuma exceção de scanner, em
nenhum dos 3 audits de produção. `evidently`/`nltk` não estão presentes na imagem
real de produção (confirmado via `pip list` dentro do container e via `pip-audit`
rodado no ambiente real, não só contra o arquivo de requisitos). A funcionalidade de
drift (RF-27/RNF-55) continua funcionando de ponta a ponta, validada contra
Postgres/Redis/Celery reais, com resultado numericamente idêntico ao `evidently`
(diff < 1e-9).

**RNF-63** — sanitizer ativo, 17/17 testes reais passando, nenhuma PII/secret na
saída efetiva do `structlog`.

As 24 falhas de teste remanescentes (1 backend + 23 frontend) são **pré-existentes,
não-relacionadas, e comprovadamente inalteradas** por esta task (mesma contagem,
mesmos testes, reproduzidas de forma idêntica em HEAD limpo antes e depois da
remoção do `evidently`).

**Nenhum commit, nenhum push foi feito.** Todas as alterações estão na working tree
para revisão.
