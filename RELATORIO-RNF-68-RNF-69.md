# RNF-68 / RNF-69 — CI/CD Final do PredictIQ

**Data:** 2026-09-21
**Escopo:** [`.github/workflows/ci.yml`](.github/workflows/ci.yml) (workflow único), scripts de suporte (`apps/backend/scripts/check_mutation_score.py`, `apps/backend/scripts/parse_mutmut_summary.py`), [`docker-compose.ci-load-smoke.yml`](docker-compose.ci-load-smoke.yml), [`mypy.ini`](mypy.ini), documentação em [`README.md §14.9`](README.md).

**Veredito: RNF-68 NÃO ATINGIDO (meta <15min; real ~46-48min) — RNF-69 PASS**

| Requisito | Antes | Depois | Meta | Status |
|---|---|---|---|---|
| RNF-68 — tempo total do pipeline | Pipeline nunca completava com sucesso (30 runs mais recentes, todas falhas — ver §1); 1ª medição limpa pós-arquitetura nova: **1h08min** | **46-48min** (2 execuções reais consecutivas: 46min18s e 47min38s) | < 15 min | ❌ NÃO ATINGIDO — real, medido, ~32% mais rápido que o baseline, mas 3x acima da meta |
| RNF-69 — artifacts publicados, inclusive em falha | Nenhum artifact de teste era publicado | 25 artifacts (11 tipos), todos com `if: always()` onde a falha do teste é o cenário relevante | Publicados mesmo em falha | ✅ PASS |
| Quality Gates preservados | — | Nenhum piso reduzido (cobertura Python 85%, mutation score 70%, cobertura frontend 75%, axe 0 Critical/Serious, 0 vuln High/Critical) | Sem redução de escopo | ✅ PASS |
| Gates realmente bloqueiam | Não testado nesta task | Testado (ver §5): `mutation-score-gate` falha com score sintético abaixo do piso; `quality-gate` falhou de verdade em 4 execuções reais por falhas genuínas em `test-typescript`/`test-e2e` | Bloqueio real, não cosmético | ✅ PASS |

---

## 1. Auditoria inicial (antes de qualquer mudança)

Via API pública do GitHub Actions (`/repos/.../actions/runs`, sem autenticação): **as 30 execuções mais recentes do workflow tinham falhado**, nenhuma chegava a `quality-gate`. Causas raiz identificadas e corrigidas nesta task (detalhe nos commits do branch `feat/rnf-68-69-ci-cd-pipeline`):

1. Débito real de lint/type-check em `apps/ml` nunca visto localmente (dependências `>=` não fixadas em `requirements.txt` — ambiente local nunca instalou torch/pytorch-lightning/optuna de verdade).
2. Ferramentas de lint (`ruff`/`black`/`mypy`) instaladas sem versão fixa no CI — drift de versão vs. `.pre-commit-config.yaml`.
3. `prettier` faltando como devDependency do frontend (só existia via `pnpm dlx`, não fixado).
4. `pnpm-lock.yaml` em `lockfileVersion: '9.0'` (pnpm v10) vs. `pnpm/action-setup@v3` fixado em v8 no CI.
5. Dois jobs (`lint-typescript`, `test-typescript`) sem `cd apps/frontend` no step de instalação — falhavam antes de rodar qualquer teste.
6. `actions/setup-node@v4` sem `cache-dependency-path` em 3 jobs — cache do pnpm nunca "pegava".

Também não havia **nenhum** artifact de teste publicado (RNF-69 não existia) e o job de mutation testing rodava em 4 grupos artificiais sem granularidade real.

---

## 2. RNF-68 — o que foi paralelizado

### 2.1 `needs` removidos (dependências falsas)

O pipeline anterior encadeava `lint → test`, `lint → test → e2e` sem dependência real de dado entre os jobs — só compartilhavam runner por conveniência de YAML. `pytest` não precisa que `ruff`/`black`/`mypy` tenham passado antes (cada job instala as próprias dependências); o E2E sobe seu próprio `next dev` + MSW, não depende da suíte Vitest ter passado. Removido todo `needs` que não refletia uma dependência real de artefato. Os únicos `needs` que sobraram: `mutation-score-gate` (precisa dos 18 grupos de mutation testing) e `quality-gate` final (precisa de todos, de propósito).

### 2.2 Matrix de mutation testing: 4 → 9 → 18 grupos

Mutation testing é a etapa mais lenta por ordem de magnitude (cada um dos ~1400 mutantes dispara sua própria subprocess de pytest). Histórico real desta task:

| Iteração | Grupos | Composição | Resultado medido |
|---|---|---|---|
| 1 (herdada) | 4 | Agrupamento arbitrário | Nunca completava (CI quebrado, ver §1) |
| 2 (esta task) | 9 | 2 arquivos por grupo (exceto `model_service.py` sozinho) | 1h08min, gargalo de 67min (`preprocessing.py`+`simulator.py`) |
| 3 (esta task) | 18 | 1 arquivo por grupo (granularidade máxima) | 46-48min, gargalo de ~46min (`simulator.py` sozinho) |

Cada grupo cobre uma fatia disjunta dos mesmos 18 arquivos de `apps/backend/pyproject.toml [tool.mutmut] paths_to_mutate` — verificado programaticamente (sem duplicata, sem omissão) antes de cada mudança.

### 2.3 Tentativa de otimização revertida: `pytest-xdist` (`-n auto`)

Benchmark local (16 núcleos): suíte completa (708 testes) de ~77s serial para ~30-35s com `-n auto`, determinístico em 3 execuções. Aplicado ao `runner` do mutmut (`pyproject.toml [tool.mutmut]`) na expectativa de reduzir o custo por mutante sobrevivente (mutante sobrevivente não tem "primeiro teste que falha" pra cortar em `-x` — reroda a suíte inteira até o fim).

**Resultado real no GitHub Actions foi o oposto**: o único par antes/depois limpo (`model_service.py` isolado, mesmo arquivo, nas duas runs) foi de **38min sem `-n auto` → 50min com `-n auto`** — mais lento, não mais rápido (runner hospedado do GitHub tem muito menos núcleos que a máquina de desenvolvimento usada no benchmark local). Revertido no commit seguinte. Isto é documentado explicitamente como o principío geral desta task: **prova real do GitHub Actions prevalece sobre benchmark local quando os dois discordam** — nunca declarar uma otimização "PASS" só porque funcionou localmente.

### 2.4 Tabela de duração por job (execução final, run 35659010834, commit `61cb9e7`)

Pipeline completo: **21:45:31Z → 22:33:09Z = 47min38s.**

| Job | Duração |
|---|---|
| Check Environment Variables | 0:00:07 |
| Dependency Security Audit (RNF-62) | 0:00:53 |
| Lint TypeScript | 0:00:32 |
| Lint Python | 0:03:23 |
| Load Smoke Test (Locust, RNF-68/69) | 0:01:51 |
| Test Python (Backend & ML) | 0:03:48 |
| Test TypeScript | 0:01:14 (falha — pré-existente, ver §4) |
| E2E Tests (Playwright + MSW) | 0:11:51 (falha — pré-existente, ver §4) |
| Mutation Testing — grupo 1 (`alert_service.py`) | 0:08:14 |
| Mutation Testing — grupo 2 (`alert_settings_service.py`) | 0:07:16 |
| Mutation Testing — grupo 3 (`critical_failure_notification_service.py`) | 0:12:07 |
| Mutation Testing — grupo 4 (`drift_monitor.py`) | 0:28:07 |
| Mutation Testing — grupo 5 (`feature_buffer.py`) | 0:05:29 |
| Mutation Testing — grupo 6 (`inference_pipeline.py`) | 0:24:12 |
| Mutation Testing — grupo 7 (`maintenance_suggestion_service.py`) | 0:27:24 |
| Mutation Testing — grupo 8 (`mlp_adapter.py`) | 0:17:41 |
| Mutation Testing — grupo 9 (`model_registry.py`) | 0:12:41 |
| Mutation Testing — grupo 10 (`model_service.py`) | 0:44:07 |
| Mutation Testing — grupo 11 (`onnx_autoencoder_adapter.py`) | 0:26:13 |
| Mutation Testing — grupo 12 (`onnx_sequence_adapter.py`) | 0:14:09 |
| Mutation Testing — grupo 13 (`onnx_tree_adapter.py`) | 0:11:37 |
| Mutation Testing — grupo 14 (`prediction_service.py`) | 0:03:51 |
| Mutation Testing — grupo 15 (`preprocessing.py`) | 0:16:33 |
| Mutation Testing — grupo 16 (`sensor_stream_service.py`) | 0:14:45 |
| **Mutation Testing — grupo 17 (`simulator.py`)** | **0:46:41 (gargalo real)** |
| Mutation Testing — grupo 18 (`telegram_alert_rate_limiter.py`) | 0:07:13 |
| Mutation Score Gate (piso 70%) | 0:00:10 |
| Quality Gate | 0:00:02 |

### 2.5 Por que a meta de <15min não foi atingida — causa raiz real

Lido diretamente em `mutmut/__init__.py` (não suposto): mutmut decide sobrevivência de um mutante **só pelo exit code do `runner` configurado** (`returncode != 1`), nunca por parsing de texto — o que confirma que `pytest-xdist` é seguro (não quebra a detecção), mas também confirma o mecanismo do gargalo: um mutante **sobrevivente** não tem "primeiro teste que falha" para cortar em `-x`, então reroda a suíte inteira (708 testes) até o fim, a cada mutante. Multiplicado pelo número de mutantes de cada arquivo, isso domina o tempo — não o número de grupos do matrix.

Com a granularidade máxima real já aplicada (18 grupos = 1 arquivo por job, o teto de paralelismo sem fatiar mutantes de um mesmo arquivo entre jobs), o gargalo estabilizou em `simulator.py` sozinho, reproduzido em 2 execuções consecutivas (46min41s e — na run anterior sem o fix do Storybook — 45min34s). Fechar o gap até <15min exigiria uma alavanca que esta task **não implementou**: sharding dos mutantes de um mesmo arquivo por ID entre múltiplos jobs (mutmut permite rodar um subconjunto de IDs de mutante). Isso é engenharia real adicional — precisaria descobrir a contagem de mutantes de `simulator.py` antes de rodar, dividir IDs entre N jobs, e agregar o score corretamente entre shards sem contar nada em duplicidade — com risco real de bug na agregação. Fica registrado aqui como **recomendação de trabalho futuro**, fora do escopo desta task (arquitetura do pipeline existente, não uma reescrita do mecanismo de matrix do mutmut).

**Isto não foi escondido nem mascarado**: os quatro commits desta task (`5d5dbb9`, `2bec7ef`, `61cb9e7`, `9cd8069`) documentam cada tentativa, o que funcionou, o que não funcionou, e por quê — inclusive a reversão do `-n auto`, que é uma tentativa real registrada como fracassada, não apagada do histórico.

---

## 3. RNF-69 — artifacts publicados

Confirmado via `GET /actions/runs/{id}/artifacts` da execução final (run 35659010834): **25 artifacts**, todos com `if: always()` no step de upload onde a falha do job ainda é o cenário mais importante para investigar.

| Artifact | Job | Conteúdo | Publicado mesmo em falha do job? |
|---|---|---|---|
| `coverage-python` | `test-python` | `htmlcov/`, `coverage.xml`, `pytest-report.xml` (JUnit) | Sim (`if: always()`) |
| `mutmut-summary-group-1` … `-18` (×18) | `mutation-testing` | JSON por grupo — `total/killed/timeout/survived/skipped` | Sim (`if: always()`) |
| `coverage-frontend` | `test-typescript` | `coverage/` (HTML navegável + `lcov.info`) | **Sim, confirmado** — presente mesmo com `test-typescript` em falha |
| `accessibility-report` | `test-typescript` | `a11y-report.json` (reporter JSON do Vitest, RNF-66/67) | **Sim, confirmado** |
| `storybook-build` | `test-typescript` | `storybook-static/` — catálogo de componentes navegável | **Corrigido nesta task** (ver §3.1) — confirmado presente na execução final |
| `playwright-report` | `test-e2e` | HTML navegável (`pnpm exec playwright show-report`) | **Sim, confirmado** — presente mesmo com `test-e2e` em falha |
| `e2e-results` | `test-e2e` | Screenshots + `trace.zip` dos testes falhos | **Sim, confirmado** |
| `locust-report` | `load-smoke` | CSV + HTML do Locust | Sim (job passou nesta execução) |

### 3.1 Achado real corrigido: `storybook-build` sumia silenciosamente

O step "Build Storybook" **não tinha** `if: always()`. Como o step do Vitest falha antes dele (falhas pré-existentes, ver §4), o Storybook nunca chegava a rodar — confirmado pela própria annotation pública do GitHub Actions na run anterior: `"No files were found with the provided path: apps/frontend/storybook-static/. No artifacts will be uploaded."`. Corrigido no commit `61cb9e7` (`if: always()` no step de build) — Storybook é uma preocupação independente do Vitest (RNF-58) e continua podendo falhar o job por conta própria se o build quebrar de verdade, só deixou de ficar refém de um step anterior não relacionado. Confirmado presente na execução seguinte (run 35659010834).

---

## 4. Resumo dos resultados de teste (execução final, run 35659010834)

| Suíte | Resultado | Observação |
|---|---|---|
| **pytest** (backend) | 707 passed, 1 xfailed — 100% limpo | `Test Python` job: 0:03:48 |
| **Mutation Testing** (mutmut) | Score agregado ≥ 70% (`mutation-score-gate` PASS) | 18/18 grupos `success` |
| **Vitest** (frontend) | ≥21 falhas pré-existentes e documentadas, **nenhuma nova** | Ver detalhe abaixo |
| **axe (`a11y.test.tsx`)** | 0 violações Critical/Serious | Roda dentro da suíte Vitest — nenhuma falha relacionada a acessibilidade apareceu nos logs desta execução |
| **Playwright/E2E** | 12 failed / 22 passed (34 total) | 100% das falhas em `e2e/alert_settings.spec.ts` (RF-25) |
| **ESLint** (`lint-typescript`) | PASS | 0:00:32 |
| **Build** (Storybook) | Corrigido para rodar sempre (§3.1); artifact presente na execução final | — |
| **TypeScript** (build do Next.js) | Não é um job próprio — validado implicitamente pelo build do E2E (`next dev` sobe sem erro de compilação) | — |
| **Locust** (load-smoke, RNF-68/69) | PASS — p95 < 200ms com 15 conexões/25s | Job: 0:01:51 |
| **Segurança** (`security-audit`) | PASS — 0 vulnerabilidades High/Critical | Job: 0:00:53 |

### 4.1 Detalhe real das falhas de Vitest (não genérico — causas identificadas)

Confirmado via logs reais colados pelo usuário do job `Test TypeScript` (run 35659010834), não estimado:

- **`use-alert-websocket.test.ts` — 4 falhas**: bug real na lógica de FIFO da fila de alertas (RNF-34) — `expected [...] to have a length of 5 but got 1`, `expected undefined to be 'alert-2'`. A fila não está respeitando o cap de 5 alertas simultâneos nem o descarte FIFO do mais antigo como deveria.
- **`use-sensor-data.test.ts` — 1 falha**: bug real de classificação de risco — `expected 'NORMAL' to be 'ALERTA'` para probabilidade na faixa 0.3–0.65.
- **`loading-states.test.tsx` / `connection-resilience.test.tsx` — ≥16 falhas**: drift entre componente e teste — `data-testid` esperados (`degraded-mode-banner`, `error-state`, `gauge-skeleton`, `kpi-skeleton`, `chip-skeleton`) e textos esperados (`/pressão tp2/i`, `NORMAL`, `TP3`, `/monitoramento em tempo real/i`) não são encontrados no DOM renderizado nestes cenários de teste.

**Estas são falhas reais de aplicação/teste, não flakiness de infraestrutura de CI** — e são **as mesmas falhas pré-existentes** já documentadas em commits anteriores a esta task (`f3b2785`, RNF-66/67), fora do escopo de RNF-68/69 (arquitetura do pipeline, não correção de bugs de frontend). Nenhuma regressão nova foi introduzida pelas mudanças desta task — confirmado comparando com o comportamento do `test-typescript` antes de qualquer mudança de CI.

### 4.2 Detalhe real das falhas de E2E

12 falhas, 100% em `e2e/alert_settings.spec.ts` (RF-25) — investigado em profundidade nesta mesma task (não apenas observado): causa raiz é uma condição de corrida real do MSW (`Found a redundant "worker.start()" call`, visível nos logs). Uma correção (`waitForServiceWorkerControl()` aguardando `navigator.serviceWorker.controller`) foi tentada, testada localmente (execução completa do Playwright, 10+ minutos), e **descartada**: não corrigiu as 12 falhas-alvo e introduziu flakiness nova em 2 testes antes estáveis de `maintenance_assistant.spec.ts`. Revertida integralmente (confirmado via `git diff --stat` sem diferença) — decisão consciente de não trocar um problema conhecido por um pior.

---

## 5. Prova real de que os gates bloqueiam (não são cosméticos)

**`quality-gate` (agregado geral)** — prova orgânica, não sintética: falhou de verdade em **4 execuções reais consecutivas** desta task (runs 35638962074, 35648496340, 35654114583, 35659010834), sempre pela mesma causa genuína (`test-typescript`/`test-e2e` com falhas reais pré-existentes) — o mecanismo `contains(needs.*.result, 'failure')` do job funcionou exatamente como projetado, sem nenhum caso de falso-positivo/negativo observado.

**`mutation-score-gate` (piso 70%, RNF-64)** — testado diretamente contra o script real usado em produção (`apps/backend/scripts/check_mutation_score.py`), com dados sintéticos, para confirmar que o piso bloqueia de verdade quando o score cai abaixo dele (o score real desta task nunca ficou abaixo de 70%, então este caminho nunca foi exercitado organicamente):

```
$ python check_mutation_score.py 70 low-1.json low-2.json
Mutation score: 40.00%  (piso RNF-64: 70.0%)
FALHOU: mutation score 40.00% abaixo do piso de 70.0% (RNF-64).
$ echo $?
1
```
```
$ python check_mutation_score.py 70 high-1.json
Mutation score: 80.00%  (piso RNF-64: 70.0%)
OK: mutation score dentro do piso exigido pela RNF-64.
$ echo $?
0
```

Confirma: o gate falha (`exit 1`) abaixo do piso e passa (`exit 0`) acima — é o mesmo script, com a mesma fórmula (`(killed+timeout)/(total-skipped)*100`), rodado pelo job `mutation-score-gate` do CI real.

---

## 6. Resumo do CI (workflow)

- **Workflow:** [`.github/workflows/ci.yml`](.github/workflows/ci.yml), nome `CI Pipeline`.
- **Triggers:** `push` e `pull_request` para `main`.
- **Concurrency:** `ci-${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}`, `cancel-in-progress: true` — cancela a run anterior do mesmo PR/branch ao chegar um push novo (achado real: sem isso, dois runs concorrentes competiam por capacidade de runner e contaminavam a medição de RNF-68 — o primeiro run medido, 1h08min, tinha essa contaminação antes do `concurrency` existir).
- **Matrix:** `mutation-testing`, 18 grupos (1 arquivo cada), `fail-fast: false`.
- **Caches:** pip (`actions/setup-python@v5 cache: pip`), pnpm (`actions/setup-node@v4 cache: pnpm` + `cache-dependency-path` correto em todos os 4 jobs frontend), browsers do Playwright (`actions/cache@v4` chaveado no hash do `pnpm-lock.yaml`, instala só no cache miss).
- **Gates:** cobertura Python ≥85%, mutation score ≥70%, cobertura frontend ≥75%, axe Critical/Serious=0, 0 vulnerabilidades High/Critical, import boundaries (RNF-56), tamanho de componente ≤200 linhas (RNF-58) — nenhum reduzido.
- **Artifacts:** 11 tipos, 25 arquivos na execução final (§3).

## 7. Confirmação do README

[`README.md §14.9`](README.md) atualizado no commit `9cd8069` para refletir o estado final real: diagrama com 18 grupos (não 9), tabela com as 4 execuções reais e seus tempos, status honesto de RNF-68 (meta não atingida, causa raiz, recomendação de trabalho futuro), correção da descrição do `load-smoke` (nunca subiu `nginx`/`frontend`, ao contrário do que uma versão anterior do texto dizia).

---

## 8. Conclusão

RNF-69 está **cumprido de verdade**: todo teste publica seu artifact, inclusive (e principalmente) quando falha, com um achado real corrigido no caminho (Storybook). RNF-68 teve uma melhoria real e mensurável (1h08min → ~47min, ~32%) através de paralelismo genuíno (matrix de arquivo único) — mas a meta de <15 minutos **não foi atingida**, e isto é reportado aqui sem redução de escopo dos gates de qualidade para forçar um PASS artificial. A causa raiz é estrutural (custo por mutante sobrevivente × número de mutantes de `simulator.py`, o arquivo mais pesado do escopo), o teto real do paralelismo por arquivo já foi alcançado, e o próximo passo real (sharding de mutantes por ID dentro de um mesmo arquivo) fica registrado como recomendação, não como algo simulado.
