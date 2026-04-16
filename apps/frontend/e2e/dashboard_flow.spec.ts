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
 * MSW intercepts POST /predict/ and returns failure_probability=0.08 (NORMAL).
 * page.clock controls time so the 5 s polling interval does not slow the suite.
 * All selectors use data-testid or accessible roles — never CSS classes.
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
  await page.goto("/");
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
    await expect(
      page.getByRole("heading", { name: /monitoramento em tempo real/i }),
    ).toBeVisible();

    // ── KPI cards — verifica labels textuais ────────────────────────────
    await expect(page.getByText(/pressão tp2/i)).toBeVisible();
    await expect(page.getByText(/temperatura óleo/i)).toBeVisible();
    await expect(page.getByText(/corrente motor/i)).toBeVisible();
    await expect(page.getByText(/reservatório/i)).toBeVisible();
  });

  test("2. painel de auditoria renderiza com área de histórico", async ({
    page,
  }) => {
    await gotoAndWaitMsw(page);

    const alertPanel = page.getByTestId("alert-panel");
    await expect(alertPanel).toBeVisible();

    // Título do painel
    await expect(alertPanel.getByText("Auditoria")).toBeVisible();
    // Seção de status atual
    await expect(alertPanel.getByText(/status atual/i)).toBeVisible();
  });

  test("3. primeira poll resulta em badge NORMAL e probabilidade no gauge", async ({
    page,
  }) => {
    await gotoAndWaitMsw(page);

    // MSW retorna failure_probability=0.08 → riskLevel NORMAL
    // O StatusBadge em SensorMonitor e o RiskBadge em AlertPanel mostram "NORMAL"
    await expect(page.getByText("NORMAL").first()).toBeVisible({
      timeout: 10_000,
    });

    // A probabilidade exibida no AlertPanel deve ser "8.0%" (0.08 × 100)
    const alertPanel = page.getByTestId("alert-panel");
    await expect(alertPanel).toContainText("8.0%", { timeout: 10_000 });
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

    // Aguarda que a primeira entrada apareça no histórico
    const historyEntries = page
      .getByTestId("alert-panel")
      .locator("[data-risk]");

    await expect(historyEntries.first()).toBeVisible({ timeout: 10_000 });

    // Avança o relógio além do POLL_INTERVAL_MS (5 000 ms) para disparar
    // o segundo tick do setInterval.
    await page.clock.fastForward(5_100);

    // O polling deve ter produzido uma segunda entrada no AlertPanel
    await expect(historyEntries).toHaveCount(2, { timeout: 8_000 });
  });

  test("5. entrada do histórico aparece com probabilidade correta (RNF-14)", async ({
    page,
  }) => {
    await gotoAndWaitMsw(page);

    // Aguarda a primeira entrada surgir (persistida via usePredictionHistory)
    const firstEntry = page
      .getByTestId("alert-panel")
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

    // O SensorChart renderiza os sub-labels de pressão e corrente/temperatura
    const { container } = { container: page.locator("body") };
    await expect(container.getByText(/telemetria de sensores/i)).toBeVisible({
      timeout: 10_000,
    });
  });
});
