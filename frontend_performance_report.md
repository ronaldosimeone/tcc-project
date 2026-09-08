# Relatório de Performance — Dashboard PredictIQ (RNF-39 / RNF-40)

Medido com dados reais (React Profiler + `requestAnimationFrame`/`PerformanceObserver`
via Playwright, e Lighthouse contra o build de produção) — não é análise estática.
Scripts reprodutíveis: `apps/frontend/scripts/measure-perf.mjs`,
`apps/frontend/lib/dev-profiler.tsx`.

## Escopo real do Dashboard (achado antes de otimizar)

A task partia da premissa de que o "histórico de alertas" teria centenas/milhares
de itens exigindo virtualização. Inspecionado o código real:

- `EventFeedCard` (Eventos Recentes) é hard-capped em **10 itens visíveis**
  (`VISIBLE_FEED_EVENTS = 10`), alimentado por `useAlertWebSocket()` que já
  mantém uma fila FIFO de **no máximo 5** alertas (`QUEUE_MAX = 5`, RNF-34) +
  5 eventos demo estáticos.
- `FleetHealthTable` (Saúde da Frota) renderiza **5 linhas fixas** (1 ativo
  real + 4 mocks) — não pagina, não cresce.
- `AssetEfficiencyChart`/`AssetRadarChart`/`SensorChart` (as visualizações
  mais pesadas do projeto, via `recharts`) **não fazem parte do Dashboard**
  — vivem só em `/sensors/[id]`, uma rota separada (code-split automático
  do Next.js App Router).

**Não existe lista grande para virtualizar no Dashboard hoje.** Instalar
`react-window`/`@tanstack/react-virtual` para 10 `<li>` seria overhead puro
(custo de virtualização > custo de renderizar 10 nós) e violaria a própria
instrução da task de não otimizar sem gargalo real observado. Documentado
aqui em vez de forçado — se o feed evoluir para um histórico real e grande,
reavaliar.

## Baseline (React Profiler, 20s de stream SSE real, dev mode)

| Componente | Renders/20s | Custo total | Custo médio |
| --- | --: | --: | --: |
| FleetKPIs | 64 | 490.0 ms | 7.656 ms |
| FleetHealthTable | 24 | 143.8 ms | 5.992 ms |
| ModelStatusCard | 48 | 188.6 ms | 3.929 ms |
| EventFeedCard | 48 | 82.9 ms | 1.727 ms |

FPS real (Playwright, sem o throttling de aba oculta que o pane do agente
sofre): **58.6 fps** médio, p95 do tempo por frame = 16.8ms, **16 frames
com jank real** (>=2 vsyncs perdidos), **16 long tasks** somando 842ms
(máx. 60ms) em 20s.

### Gargalo identificado

`FleetHealthTable`, `ModelStatusCard` e `EventFeedCard` re-renderizavam a
**cada tick SSE de 1Hz** (o dobro do esperado em alguns casos — dev mode
usa `React.StrictMode`, que duplica renders só em desenvolvimento) mesmo
quando nenhum dos seus dados relevantes mudava:

- `effectiveRiskLevel`/`effectiveProb` (props de `FleetHealthTable`) só
  mudam no poll de 5s ou em alerta WS — não no tick SSE.
- `distribution` (prop de `ModelStatusCard`) já chega memoizada
  (`useMemo` no pai, keyed em `effectiveRiskLevel`) — referência estável
  entre ticks SSE.
- `EventFeedCard` **não recebe props** — todo seu estado vem de
  `useAlertWebSocket()`, uma fonte 100% independente do SSE.

Causa raiz: nenhum dos três era `React.memo`, então toda vez que
`FleetDashboard` re-renderizava (a cada tick SSE, porque `useSensorData`
atualiza `currentLatency`/`history` uma vez por segundo), os três
re-executavam seu corpo inteiro (JSX, `recharts`, formatação) mesmo com
props idênticas — exatamente o padrão que a task pediu para evitar
("SSE event → estado global inteiro atualizado → Dashboard inteiro
re-renderizado").

`FleetKPIs` foi **deliberadamente deixado sem memo**: ele exibe a
telemetria de latência ao vivo, que muda a cada tick por design — memoizar
não traria nenhum ganho (bloquearia um render que sempre é necessário) e
esconderia um bug real caso a latência parasse de atualizar.

## Otimizações aplicadas

1. **`React.memo`** em `FleetHealthTable`, `ModelStatusCard`, `EventFeedCard`
   (`apps/frontend/components/dashboard/*.tsx`) — cada um com um comentário
   no próprio código explicando por que é seguro (quais props realmente
   mudam e quando).
2. **Code splitting** — `SimulationPanel` (formulário de simulação MLOps,
   aberto só quando o usuário clica em "Simulação" na Sidebar) passou a
   `next/dynamic(() => import(...), { ssr: false })` em
   `apps/frontend/components/sidebar.tsx`. Antes, seu JS entrava no bundle
   inicial de **toda** visita a `/` mesmo para quem nunca abre o painel.
   Não aplicado a `FleetKPIs`/`ModelStatusCard` (usam `recharts`, mas estão
   acima da dobra, visíveis no primeiro paint — lazy-load ali seria
   exatamente o anti-padrão que a task pede para evitar.
3. **Nenhuma virtualização** — ver seção acima.
4. **Nenhuma otimização de estado adicional** — `useMemo`/`useCallback` já
   presentes no código (ex.: `distribution`, `latencyTelemetry` em
   `FleetDashboard`) já cobriam o lado do produtor; faltava só o lado do
   consumidor (`React.memo`), que era o gargalo real medido.

## Depois (mesma medição, 20s, dev mode)

| Componente | Renders/20s | Custo total | Custo médio |
| --- | --: | --: | --: |
| FleetKPIs | 64 | 354.0 ms | 5.531 ms |
| FleetHealthTable | 24 | **0 ms** | 0 ms |
| ModelStatusCard | 24 | **0 ms** | 0 ms |
| EventFeedCard | 24 | **0 ms** | 0 ms |

("Renders" continua > 0 porque o `Profiler` de teste envolve o componente
por fora — ele é visitado a cada commit do pai, mas o `React.memo` faz o
corpo do componente memoizado **não re-executar**, daí custo = 0.)

FPS real: **60.0 fps** médio, p95 do tempo por frame = 16.7ms, **0 frames
com jank real**, **0 long tasks** (antes: 16 tasks / 842ms).

## Antes vs Depois

| Métrica | Antes | Depois |
| --- | --: | --: |
| FPS médio (real, 20s) | 58.6 | **60.0** |
| Frames com jank (>=2 vsyncs) | 16 | **0** |
| Long tasks (>50ms) | 16 (842ms) | **0 (0ms)** |
| Custo de render — 3 componentes memoizados | 415.3 ms/20s | **0 ms/20s** |
| Lighthouse Performance (produção) | 90 | 88* |
| Lighthouse Accessibility | 89 | 89 |
| Lighthouse Best Practices | 96 | 96 |
| Lighthouse "Reduce unused JavaScript" | ~600 ms | **~450 ms** |
| JS inicial (rota `/`) | inclui SimulationPanel | SimulationPanel sob demanda |

\* Variação de 2 pontos (90→88) está dentro do ruído normal de execução do
Lighthouse (mesma máquina, ~mesma carga); a métrica que o code-splitting
realmente afeta ("unused JavaScript") caiu de ~600ms para ~450ms nos dois
runs. Ambos os números estão folgadamente acima do limite de 85 — não há
motivo para re-rodar até "melhorar" o score, isso seria manipular a
condição do teste.

## RNF-39 — Performance de rendering

**PASS.** FPS real medido (Playwright, sem o throttling de aba oculta que
o browser pane do agente sofre — documentado abaixo) foi de **60.0 fps**
médio após a otimização (58.6 antes), com **zero frames de jank
perceptível** e **zero long tasks** durante 20s de stream SSE contínuo —
contra 16 de cada antes. Evidência: `frontend_perf_baseline.json` /
`frontend_perf_optimized.json` (raiz do repo, gerados por
`node apps/frontend/scripts/measure-perf.mjs`).

## RNF-40 — Lighthouse

**PASS.** Performance = 88 (build de produção, `next build && next start`),
acima do limite de 85. Accessibility 89, Best Practices 96, SEO 100.
Evidência: `lighthouse_prod_baseline.json` / `lighthouse_prod_optimized.json`.

## Observações e limitações do ambiente

1. **Lighthouse precisa rodar contra o build de PRODUÇÃO, não `next dev`.**
   Uma primeira tentativa contra o dev server deu Performance=60 (unminified
   JS, sem tree-shaking, overhead do HMR) — descartado como não-representativo,
   documentado aqui em vez de reportado como o número real (o arquivo
   `lighthouse_optimized.json` no repo é esse dado descartado, mantido só
   para transparência). Os números de produção citados acima vêm de um
   container temporário (`docker compose run --rm -p 3002:3000 ... pnpm start`)
   porque o container de dev principal roda `next dev` como processo PID 1.
2. **FPS via `requestAnimationFrame` no browser pane do agente sempre deu 0.**
   `document.visibilityState` reporta `"hidden"` nesse pane mesmo com a aba
   em foco, o que suspende `requestAnimationFrame` por spec do browser.
   Resolvido rodando a medição via Playwright (`measure-perf.mjs`), que não
   sofre esse throttling — confirmado com `visibilityState: "visible"` em
   todas as runs usadas neste relatório.
3. **Turbopack + bind mount do Docker no Windows não recarrega via HMR de
   forma confiável.** Depois de editar os componentes, o servidor de dev
   continuava servindo a versão antiga (sem erro, sem log) até um
   `docker compose restart frontend`. Isso é um problema do ambiente de
   desenvolvimento, não do código — mas quem for reproduzir esta medição
   precisa saber que HMR sozinho não é confiável aqui.
4. **Bug pré-existente, fora do escopo, não corrigido:** 23 testes Vitest
   falhando em 5 arquivos (`use-sensor-data`, `use-alert-websocket`,
   `sensor-monitor`, `loading-states`, `connection-resilience`) — confirmado
   que já falham em código limpo (sem nenhuma mudança desta task, verificado
   via `git stash`). Causa raiz: `RISK_THRESHOLDS.ALERT` é `0.35` em
   `lib/risk-thresholds.ts`, mas os testes assumem um limiar mais baixo
   (ex.: `getRiskLevel(0.3)` esperando `"ALERTA"`). Sinalizado como task
   separada.
5. Números de dev-mode (React Profiler) incluem overhead do
   `React.StrictMode` (Next.js liga por padrão), que duplica renders só em
   desenvolvimento — a comparação relativa (antes/depois) continua válida
   porque ambas as medições sofrem o mesmo overhead igualmente.
