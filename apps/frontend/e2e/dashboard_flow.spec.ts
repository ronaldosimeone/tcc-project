/**
 * E2E — Fluxo normal do dashboard (RNF-16).
 *
 * Valida:
 * 1. Estrutura inicial — KPI cards carregam corretamente.
 * 2. Primeira poll  — badge de status muda de "Aguardando" para "NORMAL"
 * e a probabilidade aparece no gauge.
 * 3. Polling contínuo — page.clock avança o relógio 5 s para acionar o
 * setInterval do useSensorData; um segundo ponto
 * aparece no painel de auditoria.
 * 4. Persistência — após uma predição bem-sucedida a entrada aparece na
 * lista de histórico do AlertPanel (RNF-14).
 *
 * Strategy
 * --------
 * MSW intercepts GET /api/v1/predictions (seed + poll a cada 5s) e retorna
 * failure_probability conforme `__E2E_SCENARIO__` — não mais POST /predict/,
 * que `hooks/use-sensor-data.ts` não chama há tempos (comentário desatualizado
 * de uma versão anterior do hook; ver `mocks/handlers.ts`).
 * page.clock controls time so the 5 s polling interval does not slow the suite.
 * All selectors use data-testid or accessible roles — never CSS classes.
 *
 * Achado de arquitetura (não relacionado a MSW): ver docstring de
 * `failure_alert.spec.ts` — o `AlertPanel` original é código morto desde o
 * commit `de44dc1`; os testids `alert-panel`/`prediction-history` usados
 * aqui foram adicionados aos elementos equivalentes JÁ EXISTENTES em
 * `sensor-monitor.tsx` (cabeçalho imersivo / card "Histórico de Eventos").
 */

import { test, expect } from "@playwright/test";

// ── Shared setup ──────────────────────────────────────────────────────────

/**
 * Navigate and wait for MSW to be fully initialised before each test.
 * Without this wait, the first useSensorData tick races against SW
 * registration and fails to reach the mock.
 */
async function gotoAndWaitMsw(
  page: import("@playwright/test").Page,
): Promise<void> {
  // Correção (bug de integração MSW/Playwright, não relacionado a MSW):
  // este spec testa `SensorMonitor`/`AlertPanel` (badge NORMAL, gauge,
  // painel de auditoria) — desde o commit 80405d7 ("feat: implement history
  // and sensors views") esse componente vive em `/sensors/[id]`, não mais em
  // `/` (raiz — hoje `FleetDashboard`, uma tela de fleet overview diferente).
  // "APU-Trem-042" é o id real usado pela navegação de produção
  // (`components/sidebar.tsx`, `FleetHealthTable.tsx`).
  await page.goto("/sensors/APU-Trem-042");
  // MswProvider renders this sentinel only after worker.start() resolves.
  await expect(page.locator('[data-testid="msw-ready"]')).toBeAttached({
    timeout: 15_000,
  });
}

// ── Tests ──────────────────────────────────────────────────────────────────

test.describe("Dashboard — fluxo normal (RNF-16)", () => {
  test("1. carrega o cabeçalho e os quatro KPI cards", async ({ page }) => {
    await gotoAndWaitMsw(page);

    // ── Heading ─────────────────────────────────────────────────────────
    // Correção (pré-existente, não relacionada a MSW): o `<h1>` real de
    // `SensorMonitor` mostra o id do equipamento ("APU-Trem-042"), não o
    // texto genérico "Monitoramento em tempo real" (que só existe hoje na
    // metadata de `app/layout.tsx`, nunca como heading renderizado).
    await expect(
      page.getByRole("heading", { name: "APU-Trem-042" }),
    ).toBeVisible();

    // ── KPI cards — verifica labels textuais reais de SensorMonitor ─────
    // Correção (pré-existente, não relacionada a MSW): os 4 títulos reais
    // dos KPI cards são estes — "Pressão TP2"/"Reservatório" nunca
    // existiram como card; eram suposições desatualizadas de uma versão
    // anterior da UI.
    await expect(page.getByText(/tp3.*pressão painel/i)).toBeVisible();
    await expect(page.getByText(/temperatura óleo/i)).toBeVisible();
    await expect(page.getByText(/corrente motor/i)).toBeVisible();
    await expect(page.getByText(/anomaly score/i)).toBeVisible();
  });

  test("2. painel de auditoria renderiza com área de histórico", async ({
    page,
  }) => {
    await gotoAndWaitMsw(page);

    // Correção (pré-existente, não relacionada a MSW): "Auditoria"/"Status
    // atual" eram labels do `AlertPanel` órfão (ver docstring do arquivo).
    // O cabeçalho imersivo real (`alert-panel`) mostra o id do equipamento
    // e o status de conexão; o histórico persistido vive no card
    // "Histórico de Eventos" (`prediction-history`).
    const alertPanel = page.getByTestId("alert-panel");
    await expect(alertPanel).toBeVisible();
    await expect(alertPanel.getByText("APU-Trem-042")).toBeVisible();

    await expect(page.getByTestId("prediction-history")).toBeVisible();
    await expect(
      page.getByTestId("prediction-history").getByText(/histórico de eventos/i),
    ).toBeVisible();
  });

  test("3. primeira poll resulta em badge NORMAL e probabilidade no gauge", async ({
    page,
  }) => {
    await gotoAndWaitMsw(page);

    // MSW retorna failure_probability=0.08 → riskLevel NORMAL
    await expect(page.getByText("NORMAL").first()).toBeVisible({
      timeout: 10_000,
    });

    // Correção (pré-existente, não relacionada a MSW): a probabilidade
    // hoje é exibida no KPI "Anomaly Score" (`sensor-monitor.tsx`), não
    // mais dentro do cabeçalho `alert-panel`.
    await expect(page.getByText("8.0%").first()).toBeVisible({
      timeout: 10_000,
    });
  });

  test("4. polling adiciona segunda entrada ao painel de auditoria", async ({
    page,
  }) => {
    // Instala o relógio falso ANTES de navegar para capturar o setInterval
    // que useSensorData registra no useEffect.
    await page.clock.install();

    await gotoAndWaitMsw(page);

    // Avança 0 ms para resolver os setTimeout(fn, 0) do usePredictionHistory
    // e aguardar o primeiro tick assíncrono (void tick() no useEffect).
    await page.clock.runFor(100);

    // Aguarda que a primeira entrada apareça no histórico. Correção
    // (pré-existente, não relacionada a MSW): o histórico vive no card
    // "Histórico de Eventos" (`prediction-history`), não no `AlertPanel`
    // órfão — ver docstring do arquivo.
    const historyEntries = page
      .getByTestId("prediction-history")
      .locator("[data-risk]");

    await expect(historyEntries.first()).toBeVisible({ timeout: 10_000 });

    // Avança o relógio além do POLL_INTERVAL_MS (5 000 ms) para disparar
    // o segundo tick do setInterval.
    await page.clock.fastForward(5_100);

    // O polling deve ter produzido uma segunda entrada no histórico
    await expect(historyEntries).toHaveCount(2, { timeout: 8_000 });
  });

  test("5. entrada do histórico aparece com probabilidade correta (RNF-14)", async ({
    page,
  }) => {
    await gotoAndWaitMsw(page);

    // Aguarda a primeira entrada surgir (persistida via usePredictionHistory)
    const firstEntry = page
      .getByTestId("prediction-history")
      .locator("[data-risk]")
      .first();

    await expect(firstEntry).toBeVisible({ timeout: 10_000 });

    // Probabilidade exibida na linha do histórico: "8.0%"
    await expect(firstEntry).toContainText("8.0%");

    // Badge de risco na linha do histórico
    await expect(firstEntry).toContainText("NORMAL");
  });

  test("6. SensorChart — seção de telemetria é visível", async ({ page }) => {
    await gotoAndWaitMsw(page);

    // Correção (pré-existente, não relacionada a MSW): o título real do
    // card do gráfico principal é "Pressão em Tempo Real — TP2 & TP3"
    // (`sensor-monitor.tsx`) — "telemetria de sensores" nunca existiu no
    // componente.
    const { container } = { container: page.locator("body") };
    await expect(container.getByText(/pressão em tempo real/i)).toBeVisible({
      timeout: 10_000,
    });
  });
});
