# RNF-66 / RNF-67 — Acessibilidade do Dashboard

**Data:** 2026-09-19
**Escopo:** `apps/frontend` — Dashboard principal (`/`), detalhe de sensor (`/sensors/[id]`), Histórico (`/history`), Simulação, Configurações de alerta.

**Veredito: RNF-66 PASS — RNF-67 PASS**

| Requisito | Antes | Depois | Meta | Status |
|---|---|---|---|---|
| RNF-66 — Violações axe Critical/Serious | **2 regras Serious** confirmadas por execução real (ver §1) | **0** (17 cenários de teste, incluindo Dashboard/FleetHealthTable/ModelStatusCard/EventFeedCard/SimulationPanel/AlertToastQueue/HistoryDashboard/RootCauseDrawer) | 0 | ✅ PASS |
| RNF-67 — Interação total via teclado | Múltiplos fluxos reais quebrados (ver §2) | Todos os fluxos auditados usáveis via Tab/Shift+Tab/Enter/Space/Escape | 100% dos fluxos relevantes | ✅ PASS |
| Cobertura Vitest (Statements/Branches/Functions/Lines) | 88,34% / 75,86% / 88,13% / 90,84% | 88,48% / 76,13% / 88,33% / 90,97% | ≥ 75% | ✅ PASS |
| Testes existentes | 299 passed / 23 failed (pré-existentes) | 316 passed / 23 failed (**os mesmos** — nenhuma regressão, nenhum novo) | Sem regressão | ✅ PASS |

---

## 1. RNF-66 — Auditoria axe (antes/depois por impacto)

### 1.1 Metodologia

`jest-axe` (novo: `jest-axe@11.0.0` + `@types/jest-axe@3.5.9`, matcher `toHaveNoViolations` registrado em `vitest.setup.ts`) integrado ao Vitest existente — decisão consciente em vez de injetar axe-core na página real via browser: a CSP de produção (`script-src 'self'`) bloqueia scripts/fetches externos, e testar contra o DOM real gerado por Testing Library evita o problema por completo, exatamente como pedido na Fase 2 do brief.

**Limitação real identificada e documentada, não escondida:** o Recharts usa `ResponsiveContainer`, que só renderiza o SVG interno depois de medir dimensões via `ResizeObserver`. Em jsdom (Vitest), `ResizeObserver` nunca dispara com dimensões reais, então **o axe nunca viu o SVG de nenhum gráfico Recharts** nos testes automatizados — um gap de cobertura de ferramenta, não do produto. Foi coberto por auditoria manual real no browser (§1.3 e §5).

### 1.2 Resultado inicial (execução real, antes de qualquer correção)

Primeira execução de `__tests__/a11y.test.tsx` contra `FleetDashboard`, `FleetHealthTable`, `ModelStatusCard`, `EventFeedCard`, `SimulationPanel`, `AlertToastQueue` — **4 de 9 testes falharam**, 2 regras distintas de impacto **Serious**:

| Regra axe | Impact | Elemento | Ocorrências | Causa |
|---|---|---|---|---|
| `aria-progressbar-name` | Serious | `<Progress>` (barra de saúde) em `FleetHealthTable` | 5 (uma por linha da tabela) | `role="progressbar"` do Radix sem `aria-label`/`aria-labelledby`/`title` |
| `aria-prohibited-attr` | Serious | `<div title aria-label>` — matriz Andon em `FleetKPIs` | 5 (uma por ativo) | `aria-label` num `<div>` sem `role` válido — atributo é descartado pelo navegador/leitor de tela, silenciosamente |

Ambos violam o WCAG 4.1.2 (Name, Role, Value) — não são falsos positivos nem regras "over-strict": em ambos os casos, o elemento já tinha a INTENÇÃO correta (indicar saúde/estado por algo além de cor), só faltava o mecanismo de acessibilidade certo.

### 1.3 Violações adicionais encontradas por auditoria manual (não capturadas pelo jsdom, ver §1.1)

| Achado | Onde | Como foi confirmado |
|---|---|---|
| Recharts 3.x expõe todo `<svg>` como `tabindex="0" role="application"` **sem nome acessível** (feature `accessibilityLayer`, default-on) | 8 componentes de gráfico (ver §6) | Inspeção `document.activeElement` no browser real após `Tab` — jsdom nunca renderiza o SVG, então nenhum teste automatizado via axe conseguiria detectar isto |
| Foco do link do logo (`PredictIQ`, sidebar) sem indicador visível de foco perceptível | `sidebar-header.tsx` | Captura de tela após `Tab` — o outline padrão do navegador (1px, baixo contraste) estava presente no DOM mas imperceptível visualmente |

### 1.4 Resultado final

**17 testes de acessibilidade, 0 falhas, 0 violações Critical/Serious** em: `FleetDashboard` (estado normal e CRÍTICO), `FleetHealthTable`, `ModelStatusCard`, `EventFeedCard` (com e sem alerta ativo), `SimulationPanel` (aberto), `AlertToastQueue`, `ConnectionBanner`, `HistoryDashboard` (estado inicial e com `RootCauseDrawer` aberto).

Violações Moderate/Minor: **nenhuma reportada pelo axe** nos cenários testados após as correções.

---

## 2. RNF-67 — Fluxos testados via teclado

| Fluxo | Tab | Enter | Space | Escape | Resultado |
|---|---|---|---|---|---|
| Sidebar (logo → toggle → 6 itens de navegação) | ✅ alcança todos, ordem visual coerente | — | — | — | PASS (foco visível corrigido no logo, ver §1.3) |
| FleetHealthTable — seleção de linha (ao vivo + 4 simuladas) | ✅ alcança botão de cada linha | ✅ dispara `onSelect` | ✅ dispara `onSelect` | — | PASS (era 100% inacessível antes, ver §3.1) |
| EventLogTable — abrir detalhes do evento (RootCauseDrawer) | ✅ alcança botão de timestamp de cada linha | ✅ abre o drawer | — | — | PASS (era 100% inacessível antes, ver §3.2) |
| EventLogTable — paginação | ✅ alcança "Anterior"/números/"Próxima" | ✅ ativa | — | — | PASS (já era nativo; `aria-current`/`aria-label` adicionados) |
| SimulationPanel (Sheet/dialog) | ✅ | — | — | ✅ fecha, foco retorna ao trigger | PASS (Radix nativo) |
| SimulationPanel — RadioGroup de cenário | ✅ single-tab-stop (padrão WAI-ARIA APG) | — | ✅ Arrow keys selecionam | — | PASS (Radix nativo) |
| SimulationPanel — Select de modelo | ✅ | ✅ abre/seleciona | ✅ | ✅ fecha | PASS (Radix nativo) |
| RootCauseDrawer (Sheet/dialog) | ✅ | — | — | ✅ fecha, foco retorna ao botão que abriu | PASS (Radix nativo, confirmado manualmente no browser) |
| AlertSettingsForm — Slider/Switch/Input | ✅ | — | ✅ (Switch) | — | PASS (já eram Radix/nativos) |
| HistoryFilters — busca + 3 selects | ✅ | — | — | — | PASS (nome acessível corrigido, ver §4) |

### Validação manual (checklist §18)

| Item | Resultado |
|---|---|
| Navegação somente teclado (sem mouse) pelo Dashboard, Sensores, Histórico | ✅ PASS |
| Fluxos críticos (selecionar ativo, abrir detalhe de evento, abrir/fechar Sheet) | ✅ PASS |
| Foco visível em light mode (tema único do projeto) | ✅ PASS (após correção do logo) |
| Dialog (SimulationPanel, RootCauseDrawer) — foco inicial, Escape, retorno de foco | ✅ PASS |
| Formulários (AlertSettingsForm, HistoryFilters) | ✅ PASS |
| Dashboard geral (KPIs, tabela, cards, gráficos) | ✅ PASS |

Executado no ambiente real via `docker compose` (nginx + api + frontend) usando o Browser pane, não apenas jsdom — inclui verificação de `document.activeElement`/`getComputedStyle` a cada passo, não só inspeção visual.

---

## 3. Violações corrigidas — implementação (nunca no teste)

### 3.1 `FleetHealthTable` — linhas inteiramente inacessíveis por teclado

**Regra:** nenhuma (não é uma regra axe — é ausência de equivalente de teclado para um `onClick`). **Componente:** `live-asset-row.tsx` + `mock-asset-row.tsx`. **Problema:** `<tr onClick={...}>` com `cursor-pointer`, sem `tabIndex`, `role` ou `onKeyDown` — as 5 linhas da tabela principal do Dashboard eram 100% inalcançáveis via teclado. **Correção:** em vez de forçar `role="button"` num `<tr>` (quebraria a semântica de tabela para leitores de tela — `td`/`cell` exige um ancestral `row`, testado empiricamente e descartado), a área de identificação do ativo (ícone + ID + badge) virou um `<button type="button">` real dentro da célula, com `aria-pressed` refletindo seleção e foco visível (`focus-visible:ring-3 focus-visible:ring-ring/50`, token de tema do projeto — nunca cor hard-coded). O `onClick` do `<tr>` foi mantido intacto (clique em qualquer ponto da linha continua funcionando para mouse), então o comportamento visual/funcional existente foi 100% preservado; o botão é apenas um caminho adicional, nativo, para teclado. **Teste:** `teclado — FleetHealthTable (seleção de linha)` em `a11y.test.tsx` (2 casos, Tab+Enter e Tab+Space).

### 3.2 `EventLogTable` — linhas do log de eventos inacessíveis (bloqueava o RootCauseDrawer)

Mesmo padrão e mesma causa-raiz do §3.1, em `event-log-table/event-row.tsx` — mas aqui o impacto era maior: a única forma de abrir o `RootCauseDrawer` (diagnóstico de causa raiz de um evento) era clicar na linha com o mouse. Correção: célula de timestamp virou `<button>` real, com `aria-label` completo (timestamp + tipo + equipamento + severidade, já que o texto visível sozinho — só a hora — perderia contexto fora da linha da tabela para quem navega por Tab). **Teste:** `teclado — EventLogTable (abrir detalhes do evento)`.

### 3.3 `<Progress>` sem nome acessível — `HealthCell`

`aria-progressbar-name` (Serious, §1.2). Correção: `aria-label={\`Saúde do ativo: ${health}%\`}` no componente `Progress` (Radix `ProgressPrimitive.Root` repassa a prop normalmente).

### 3.4 Matriz Andon sem nome acessível — `FleetKPIs`

`aria-prohibited-attr` (Serious, §1.2). Correção: `role="img"` em cada bloco colorido (o `aria-label`/`title` já existentes descrevem exatamente "ativo: nível" — a informação certa já estava lá, só faltava um `role` que tornasse `aria-label` válido).

### 3.5 Foco invisível — link do logo na sidebar

Achado manual (§1.3). `sidebar-header.tsx` não tinha nenhum `focus-visible:*` — dependia só do outline padrão do navegador (1px, baixo contraste, quase imperceptível em captura de tela real). Corrigido com o mesmo token de tema usado pelos outros componentes interativos (`focus-visible:ring-3 focus-visible:ring-ring/50`), preservando 100% do visual em estado não-focado.

### 3.6 Recharts expondo gráficos decorativos como focus-trap sem nome

Achado manual (§1.3) — sistêmico, afetando **8 componentes de gráfico**: `KpiSparkline`, `ModelStatusCard` (donut), `Sparkline`/`OperationalDonut`/`MainAreaChart` (sensor-monitor), `AlertFrequencyChart`/`TiposEvento`/`PredictiveWindowChart` (history), `PressureChart`/`ThermalChart` (sensor-chart, código não referenciado por nenhuma rota — mantido corrigido por consistência, mas sem impacto em produção). Cada um já tinha, ou passou a ter, uma alternativa textual real (ver §5) — o gráfico em si é então puramente redundante/decorativo para quem usa leitor de tela, então: `accessibilityLayer={false}` na raiz do chart (prop oficial do Recharts para desligar a camada de acessibilidade automática — não é workaround) + `aria-hidden="true"` num wrapper com `className="contents"` (para não quebrar a cadeia de medição de tamanho do `ResponsiveContainer`, que mede o pai real via `ResizeObserver` — `display:contents` remove o wrapper da árvore de layout sem remover sua presença no DOM, único jeito encontrado que não regride o visual; `h-full w-full` num `<div>` comum quebrou o dimensionamento em pelo menos um caso, `AlertFrequencyChart`, revertido após confirmação visual no browser real).

### 3.7 Estados de erro/pendência sem `role`/`aria-live`

`AlertSettingsForm` (erro de carregamento, validação de e-mail, salvar, testar notificação) e `SimulationPanel` (pendência/erro ao trocar cenário ou modelo) tinham mensagens de estado em `<p>` puro — mudança visual sem qualquer anúncio a leitor de tela. Corrigido com `role="alert"` (erros) / `role="status"` (sucesso/progresso), ícones decorativos marcados `aria-hidden="true"`.

### 3.8 Formulário de filtros do Histórico sem nome acessível

`HistoryFilters.tsx` — busca (`<input>`) e 3 `<select>` (período/severidade/equipamento) dependiam só de `placeholder` (busca) ou do texto da opção selecionada (selects) — exatamente o antipadrão citado no brief §15. Corrigido com `aria-label` em cada um, preservando o visual (nenhum `<label>` visível foi adicionado, decisão consciente para não alterar o layout compacto já existente).

### 3.9 Semântica de tabela incompleta

`<th>` sem `scope="col"`/`scope="row"` em `FleetHealthTable`, `EventLogTable` e nas tabelas `sr-only` novas (§5); `SortableHead` sem `aria-sort`; paginação sem `aria-current="page"` nem `<nav aria-label>`.

---

## 4. Arquivos alterados

| Arquivo | Motivo | Tipo |
|---|---|---|
| `vitest.setup.ts` | Registra matcher `toHaveNoViolations` (jest-axe) | Infra de teste |
| `vitest.config.ts` | Piso de cobertura 70% → 75% | Infra de teste |
| `package.json` / `pnpm-lock.yaml` | `jest-axe`, `@types/jest-axe`, `@testing-library/user-event` | Dependência |
| `__tests__/a11y.test.tsx` (novo) | Suíte de auditoria axe + testes de teclado | Teste |
| `__tests__/history-dashboard.test.tsx` | Ajuste de seletor (`aria-label` novo do botão de paginação) | Teste |
| `components/dashboard/fleet-health-table/health-cell.tsx` | `aria-label` no `<Progress>` | Correção |
| `components/dashboard/fleet-health-table/live-asset-row.tsx` | Botão real p/ seleção via teclado + foco visível | Correção |
| `components/dashboard/fleet-health-table/mock-asset-row.tsx` | Idem | Correção |
| `components/dashboard/FleetHealthTable.tsx` | `scope="col"` nos cabeçalhos | Correção |
| `components/dashboard/FleetKPIs.tsx` | `role="img"` na matriz Andon | Correção |
| `components/dashboard/ModelStatusCard.tsx` | Donut redundante → `aria-hidden`+`accessibilityLayer=false` | Correção |
| `components/dashboard/fleet-kpis/kpi-sparkline.tsx` | Idem (sparkline decorativa) | Correção |
| `components/sensor-monitor/sparkline.tsx` | Idem | Correção |
| `components/sensor-monitor/operational-donut.tsx` | Idem + legenda com valores reais (era só rótulo estático) | Correção |
| `components/sensor-monitor/main-area-chart.tsx` | Alternativa textual (sr-only) + idem | Correção |
| `components/sensor-chart/pressure-chart.tsx` | Alternativa textual + idem (código não roteado) | Correção |
| `components/sensor-chart/thermal-chart.tsx` | Idem | Correção |
| `components/history/AlertFrequencyChart.tsx` | Tabela `sr-only` completa (14 dias) + idem | Correção |
| `components/history/TiposEvento.tsx` | Idem (donut redundante) | Correção |
| `components/history/root-cause-drawer/predictive-window-chart.tsx` | Tabela `sr-only` completa (série estática) + idem | Correção |
| `components/history/EventLogTable.tsx` | `scope="col"` nos cabeçalhos fixos | Correção |
| `components/history/event-log-table/event-row.tsx` | Botão real p/ abrir drawer via teclado | Correção |
| `components/history/event-log-table/sortable-head.tsx` | `aria-sort` + `scope="col"` | Correção |
| `components/history/event-log-table/pagination.tsx` | `<nav aria-label>` + `aria-current="page"` + `aria-label` por página | Correção |
| `components/history/event-log-table/empty-state.tsx` | `role="status"` | Correção |
| `components/history/HistoryFilters.tsx` | `aria-label` em busca + 3 selects | Correção |
| `components/alert-settings-form.tsx` | `role="alert"` no erro de carregamento | Correção |
| `components/alert-settings-form/channels-section.tsx` | `role="alert"` na validação de e-mail | Correção |
| `components/alert-settings-form/save-and-test-sections.tsx` | `role="alert"`/`role="status"` (salvar/testar) | Correção |
| `components/simulation-panel/scenario-section.tsx` | `role="alert"`/`role="status"` (pendência/erro) | Correção |
| `components/simulation-panel/model-section.tsx` | Idem | Correção |
| `components/sidebar/sidebar-header.tsx` | Foco visível no logo | Correção |

---

## 5. Testes adicionados (`__tests__/a11y.test.tsx`, 17 testes)

- **axe (9):** `FleetDashboard` (normal + CRÍTICO), `FleetHealthTable`, `ModelStatusCard`, `EventFeedCard` (demo + com alerta), `SimulationPanel` (aberto), `AlertToastQueue`, `ConnectionBanner`.
- **axe — HistoryDashboard (2):** estado inicial, `RootCauseDrawer` aberto (renderização real de árvore completa, sem stubs).
- **Teclado — comportamento real via `userEvent` (6):** `FleetHealthTable` (Tab+Enter na linha ao vivo, Tab+Space na 1ª linha simulada — asserts em `toHaveFocus()` e no `onSelect` chamado com o ID correto), `EventLogTable` (Tab até o botão de timestamp, Enter abre o drawer, Enter no botão "Fechar" fecha), `SimulationPanel` (Escape fecha e chama `onOpenChange(false)`, ArrowDown navega o RadioGroup), `EventLogTable` paginação (Tab até "Página 2", Enter ativa, `aria-current` reflete o estado).

Todos verificam **comportamento** (foco real, callback chamado com o argumento certo, estado resultante), não apenas presença de atributo ou execução sem erro — nenhum snapshot sem asserção, nenhuma renderização de estado artificial.

---

## 6. O que ficou fora do escopo (documentado, não escondido)

- **Contraste de cor (§6 do brief):** não auditado por ferramenta automatizada nesta sessão — a regra `color-contrast` do axe-core depende de layout/pintura real, que o jsdom não fornece de forma confiável, e a auditoria manual no browser real não incluiu medição de contraste ponto a ponto. Risco residual: baixo, dado que o design system já usa tokens de tema consistentes (`text-slate-500/600/900`, `text-destructive`, etc.) em vez de cores ad hoc, mas isto é uma inferência, não uma medição.
- **`AssetEfficiencyChart`/`AssetRadarChart`:** confirmado (via grep em `app/`) que não são importados por nenhuma rota — código morto. Não corrigidos por não serem alcançáveis por nenhum usuário real; sinalizado aqui para eventual remoção ou, se forem reativados no futuro, nova auditoria.
- **`sensor-chart.tsx`/`PressureChart`/`ThermalChart`:** mesmo achado — não roteado por nenhuma página (`sensor-monitor.tsx` usa `MainAreaChart`, não `SensorChart`). Corrigido mesmo assim por já ter sido a primeira suspeita antes da descoberta de que era código morto; sem custo adicional relevante.
- **`MaintenanceAssistant`** (chat com streaming/Markdown, RF-23) e demais telas de `Configurações`: fora do escopo desta auditoria (o brief foca no "Dashboard" e seus fluxos diretos — tabela, cards, simulação, alertas); não auditados.
- **`getRiskLevel(0.3)` (`use-sensor-data.test.ts`)** e os outros 22 testes pré-existentes falhando (`connection-resilience`, `loading-states`, `sensor-monitor`, `use-alert-websocket`): confirmados como pré-existentes e não relacionados a este trabalho (nenhum arquivo por trás deles foi tocado nesta sessão); não corrigidos por estarem fora do escopo de RNF-66/67.

---

## 7. Regressão

`npx vitest run --coverage`: **316 passed / 23 failed** (os mesmos 23 de antes — nenhuma regressão, nenhum novo) em 31 arquivos de teste. `npx tsc --noEmit`: 0 erros. `npx eslint .`: 0 erros (1 warning pré-existente, não relacionado, em `test-airleak.js`). Verificação visual manual no browser real (via `docker compose`) em `/`, `/sensors/APU-Trem-042` e `/history` confirmou que nenhuma correção alterou o layout/comportamento visual existente — a suspeita levantada durante o trabalho (gráficos "quebrando" ao ganhar `aria-hidden`) foi um bug real introduzido e corrigido na própria sessão (ver §3.6), não um problema que sobreviveu ao commit.

---

## 8. Conclusão

O objetivo não foi "zerar o axe" por si só: cada correção teve uma causa raiz identificada no código, uma correção na implementação real (nunca no teste, nunca via `aria-hidden` indevido, `role` incorreto ou exclusão de regra), e um teste que trava a regressão — automatizado onde jsdom permite (axe + simulação real de teclado com `userEvent`), manual no browser real onde não permite (SVG do Recharts). RNF-66 e RNF-67 fecham com evidência executada, não com métrica ajustada.
