/**
 * E2E — Alerta de falha crítica (RF-08 / RNF-16 / RNF-17).
 *
 * Valida:
 * 1. Quando MSW retorna failure_probability=0.9, o banner vermelho de
 * "FALHA CRÍTICA DETECTADA" (RF-08) aparece na tela.
 * 2. O AlertPanel muda para data-risk="CRÍTICO".
 * 3. A borda vermelha do painel é aplicada (classe CSS verificada via
 * atributo, não seletor de classe — robustez com Tailwind JIT).
 * 4. O badge de status em SensorMonitor exibe "CRÍTICO".
 * 5. A entrada de histórico registrada no AlertPanel mostra "CRÍTICO".
 * 6. O banner NÃO aparece em estado NORMAL.
 * 7. O banner NÃO aparece em estado ALERTA (0.45).
 *
 * Strategy
 * --------
 * page.addInitScript() injeta window.__E2E_SCENARIO__ ANTES de qualquer
 * script da página. O handler de MSW lê este valor e retorna o payload
 * correspondente. Cada teste cria um contexto de navegador limpo com um
 * novo Service Worker, garantindo isolamento total.
 *
 * Achado de arquitetura (não relacionado a MSW): o componente
 * `components/alert-panel.tsx` que este spec originalmente testava (banner
 * dedicado, testid `alert-panel`/`critical-banner`, `data-risk`) foi
 * removido de `SensorMonitor` no commit `de44dc1` ("immersive critical
 * state") — muito antes desta task — e hoje é código morto (só usado pelo
 * seu próprio teste Vitest isolado, nunca no app real). `SensorMonitor`
 * ganhou UI própria equivalente (cabeçalho imersivo com cor por risco,
 * card "Histórico de Eventos"/`EventLog` usando o MESMO `usePredictionHistory`).
 * Os testids/atributos abaixo (`alert-panel`, `critical-banner`,
 * `prediction-history`, `data-risk`) foram adicionados a esses elementos
 * JÁ EXISTENTES em `sensor-monitor.tsx` — nenhuma estrutura/layout nova,
 * nenhum comportamento alterado — para que este spec volte a validar RF-08/
 * RNF-14 de forma robusta (por testid, não texto solto).
 */

import { test, expect } from "@playwright/test";

// ── Helper ────────────────────────────────────────────────────────────────

/**
 * Define o cenário MSW via addInitScript e navega para a página.
 * addInitScript é executado pelo Playwright antes de qualquer JS da página,
 * portanto window.__E2E_SCENARIO__ está disponível quando o primeiro handler
 * roda.
 */
async function gotoWithScenario(
  page: import("@playwright/test").Page,
  scenario: "normal" | "critical" | "alert",
): Promise<void> {
  // Para cenários não-críticos deixamos a janela sem o global (default = normal).
  if (scenario !== "normal") {
    await page.addInitScript((sc) => {
      window.__E2E_SCENARIO__ = sc;
    }, scenario);
  }

  // Correção (bug de integração MSW/Playwright, não relacionado a MSW):
  // `AlertPanel`/`critical-banner` são renderizados por `SensorMonitor`, que
  // desde o commit 80405d7 ("feat: implement history and sensors views")
  // vive em `/sensors/[id]`, não mais em `/` (raiz — hoje `FleetDashboard`,
  // que não renderiza `AlertPanel`). "APU-Trem-042" é o id real usado pela
  // navegação de produção (`components/sidebar.tsx`, `FleetHealthTable.tsx`).
  await page.goto("/sensors/APU-Trem-042");
  // Correção 1: Espera anexar ao DOM ao invés de estar visível
  await expect(page.locator('[data-testid="msw-ready"]')).toBeAttached({
    timeout: 15_000,
  });
}

// ── Tests ──────────────────────────────────────────────────────────────────

test.describe("Alerta de falha crítica — RF-08 / RNF-17", () => {
  test("1. banner 'FALHA CRÍTICA DETECTADA' aparece quando prob = 0.9", async ({
    page,
  }) => {
    await gotoWithScenario(page, "critical");

    const banner = page.getByTestId("critical-banner");
    await expect(banner).toBeVisible({ timeout: 10_000 });
    await expect(banner).toContainText(/falha crítica detectada/i);
  });

  test("2. banner tem role='alert' (acessibilidade)", async ({ page }) => {
    await gotoWithScenario(page, "critical");

    // Correção 2: Busca exatamente o nosso banner e verifica a role dele,
    // ignorando o <div id="__next-route-announcer__"> injetado pelo Next.js
    const banner = page.getByTestId("critical-banner");
    await expect(banner).toBeVisible({ timeout: 10_000 });
    await expect(banner).toHaveAttribute("role", "alert");
  });

  test("3. AlertPanel tem data-risk='CRÍTICO'", async ({ page }) => {
    await gotoWithScenario(page, "critical");

    await expect(page.getByTestId("critical-banner")).toBeVisible({
      timeout: 10_000,
    });

    await expect(page.getByTestId("alert-panel")).toHaveAttribute(
      "data-risk",
      "CRÍTICO",
    );
  });

  test("4. AlertPanel aplica borda vermelha no estado crítico", async ({
    page,
  }) => {
    await gotoWithScenario(page, "critical");

    await expect(page.getByTestId("critical-banner")).toBeVisible({
      timeout: 10_000,
    });

    // Verifica a classe de borda via atributo class para não depender de
    // geração de CSS pelo Tailwind JIT (que pode não rodar em dev mode).
    // Correção (pré-existente, não relacionada a MSW): a classe real hoje
    // é `border-red-500` (sem sufixo de opacidade `/30` — isso nunca
    // existiu no cabeçalho imersivo de `SensorMonitor`, era herdado do
    // `AlertPanel` órfão).
    const panel = page.getByTestId("alert-panel");
    await expect(panel).toHaveClass(/border-red-500\b/, { timeout: 5_000 });
  });

  test("5. probabilidade exibida é 90.0% em estado crítico", async ({
    page,
  }) => {
    await gotoWithScenario(page, "critical");

    await expect(page.getByTestId("critical-banner")).toBeVisible({
      timeout: 10_000,
    });

    // Correção (pré-existente, não relacionada a MSW): o cabeçalho
    // imersivo (`alert-panel`) não exibe mais a probabilidade numérica —
    // isso hoje é o KPI "Anomaly Score" (`{anomalyScoreStr}%`, mesma
    // fórmula: `(failure_probability * 100).toFixed(1)`).
    await expect(page.getByText("90.0%").first()).toBeVisible({
      timeout: 5_000,
    });
  });

  test("6. entrada CRÍTICO aparece no histórico do painel de auditoria", async ({
    page,
  }) => {
    await gotoWithScenario(page, "critical");

    // Correção (pré-existente, não relacionada a MSW): o histórico
    // persistido por `usePredictionHistory` (RNF-14) hoje é renderizado
    // pelo card "Histórico de Eventos" (`EventLog`, testid
    // `prediction-history`), não mais pelo `AlertPanel` órfão — mesmo
    // hook, mesmos dados, UI diferente.
    const criticalEntry = page
      .getByTestId("prediction-history")
      .locator("[data-risk='CRÍTICO']")
      .first();

    await expect(criticalEntry).toBeVisible({ timeout: 10_000 });
    // A linha de histórico deve exibir a probabilidade
    await expect(criticalEntry).toContainText("90.0%");
  });

  test("7. badge 'CRÍTICO' aparece em SensorMonitor (StatusBadge)", async ({
    page,
  }) => {
    await gotoWithScenario(page, "critical");

    await expect(page.getByTestId("critical-banner")).toBeVisible({
      timeout: 10_000,
    });

    // getAllByText porque "CRÍTICO" aparece em StatusBadge E em RiskBadge do AlertPanel
    await expect(page.getByText("CRÍTICO").first()).toBeVisible({
      timeout: 5_000,
    });
  });

  // ── Testes de ausência — garantem que o banner não aparece indevidamente ──

  test("8. banner NÃO aparece no estado NORMAL (prob=0.08)", async ({
    page,
  }) => {
    await gotoWithScenario(page, "normal");

    // Aguarda pelo menos uma poll completar (badge NORMAL visível)
    await expect(page.getByText("NORMAL").first()).toBeVisible({
      timeout: 10_000,
    });

    // Correção 3: Removemos a verificação genérica de getByRole('alert')
    // para não conflitar com as injeções invisíveis do Next.js
    await expect(page.getByTestId("critical-banner")).not.toBeVisible();
  });

  test("9. AlertPanel tem data-risk='NORMAL' por padrão", async ({ page }) => {
    await gotoWithScenario(page, "normal");

    // Aguarda a primeira poll estabilizar o estado
    await expect(page.getByText("NORMAL").first()).toBeVisible({
      timeout: 10_000,
    });

    await expect(page.getByTestId("alert-panel")).toHaveAttribute(
      "data-risk",
      "NORMAL",
    );
  });
});
