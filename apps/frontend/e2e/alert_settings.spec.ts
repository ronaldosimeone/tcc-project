/**
 * E2E — Painel de Configurações de Alertas (RF-25 / RNF-49 / RNF-53-54).
 *
 * Convenção de arquivo: segue `apps/frontend/e2e/*.spec.ts` (padrão real do
 * projeto — ver nota equivalente em `maintenance_assistant.spec.ts`), não o
 * caminho `apps/frontend/tests/...` sugerido no enunciado.
 *
 * Autenticação (item 19 do enunciado): auditado — `lib/api-client.ts` NUNCA
 * envia `X-Admin-Token` em nenhuma chamada (nem em produção). A proteção real
 * do backend depende do modo dev do RF-11 (token default = auth desativada).
 * Não há nada para mockar do lado do browser — nenhuma autenticação paralela
 * foi criada.
 */

import { test, expect, type Page } from "@playwright/test";

type SettingsScenario = "configured" | "get-error" | "put-error" | "test-error";

async function setSettingsScenario(
  page: Page,
  scenario: SettingsScenario,
): Promise<void> {
  await page.addInitScript((sc) => {
    window.__E2E_SETTINGS_SCENARIO__ = sc;
  }, scenario);
}

async function gotoSettings(page: Page): Promise<void> {
  await page.goto("/settings/alerts");
  await expect(page.locator('[data-testid="msw-ready"]')).toBeAttached({
    timeout: 15_000,
  });
}

function telegramSwitch(page: Page) {
  return page.getByRole("switch", {
    name: /ativar notificações por telegram/i,
  });
}

function emailSwitch(page: Page) {
  return page.getByRole("switch", { name: /ativar notificações por e-mail/i });
}

// ---------------------------------------------------------------------------
// TESTE 1 — Carregamento
// ---------------------------------------------------------------------------

test.describe("Configurações de Alertas — carregamento (RF-25)", () => {
  test("página carrega com a configuração default (0.85, Telegram ON, e-mail OFF)", async ({
    page,
  }) => {
    await gotoSettings(page);

    await expect(
      page.getByRole("heading", { name: "Configurações de Alertas" }),
    ).toBeVisible();

    await expect(page.getByTestId("alert-threshold-value")).toHaveText("85%");
    await expect(page.getByRole("slider")).toHaveAttribute(
      "aria-valuenow",
      "0.85",
    );
    await expect(telegramSwitch(page)).toHaveAttribute("aria-checked", "true");
    await expect(emailSwitch(page)).toHaveAttribute("aria-checked", "false");
    // Campo de e-mail só aparece quando o canal está habilitado.
    await expect(
      page.getByLabel("E-mail para receber alertas"),
    ).not.toBeVisible();

    await page.screenshot({
      path: "e2e-results/screenshots/alert-settings-default.png",
    });
  });

  test("carrega uma configuração já salva (0.75, Telegram+e-mail ON, endereço preenchido)", async ({
    page,
  }) => {
    await setSettingsScenario(page, "configured");
    await gotoSettings(page);

    await expect(page.getByTestId("alert-threshold-value")).toHaveText("75%");
    await expect(telegramSwitch(page)).toHaveAttribute("aria-checked", "true");
    await expect(emailSwitch(page)).toHaveAttribute("aria-checked", "true");
    await expect(page.getByLabel("E-mail para receber alertas")).toHaveValue(
      "alertas@empresa.com.br",
    );
  });

  test("erro do backend no GET mostra estado de erro com opção de retry", async ({
    page,
  }) => {
    await setSettingsScenario(page, "get-error");
    await gotoSettings(page);

    // `AlertSettingsForm` mostra `err.message` (real, do api-client — sempre
    // "[api-client] ... falhou — HTTP <status> ...") — não a string de
    // fallback genérica, que só aparece para exceções que NÃO são `Error`.
    await expect(page.getByText(/HTTP 500/i)).toBeVisible({ timeout: 10_000 });
    await expect(
      page.getByRole("button", { name: /tentar novamente/i }),
    ).toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// TESTE 2 — Alterar o threshold (slider real, via DOM)
// ---------------------------------------------------------------------------

test.describe("Configurações de Alertas — slider (RF-25)", () => {
  test("mover o slider via teclado atualiza o valor exibido (85% → 75%)", async ({
    page,
  }) => {
    await gotoSettings(page);

    const slider = page.getByRole("slider");
    await expect(slider).toHaveAttribute("aria-valuenow", "0.85");

    await slider.focus();
    for (let i = 0; i < 10; i += 1) {
      await page.keyboard.press("ArrowLeft");
    }

    await expect(slider).toHaveAttribute("aria-valuenow", "0.75");
    await expect(page.getByTestId("alert-threshold-value")).toHaveText("75%");
  });

  test("slider respeita os limites 50%–95%", async ({ page }) => {
    await gotoSettings(page);

    const slider = page.getByRole("slider");
    await expect(slider).toHaveAttribute("aria-valuemin", "0.5");
    await expect(slider).toHaveAttribute("aria-valuemax", "0.95");

    await slider.focus();
    await page.keyboard.press("End"); // Radix: salta para o máximo
    await expect(slider).toHaveAttribute("aria-valuenow", "0.95");

    await page.keyboard.press("Home"); // Radix: salta para o mínimo
    await expect(slider).toHaveAttribute("aria-valuenow", "0.5");
  });
});

// ---------------------------------------------------------------------------
// TESTE 3 — Ativar e-mail e salvar (PUT interceptado)
// ---------------------------------------------------------------------------

test.describe("Configurações de Alertas — salvar com e-mail habilitado (RF-25)", () => {
  test("ativa e-mail, preenche endereço, salva — PUT com payload correto e feedback de sucesso", async ({
    page,
  }) => {
    await gotoSettings(page);

    await emailSwitch(page).click();
    await expect(emailSwitch(page)).toHaveAttribute("aria-checked", "true");

    const emailInput = page.getByLabel("E-mail para receber alertas");
    await emailInput.fill("operador@predictiq-dev.com");

    const saveButton = page.getByRole("button", {
      name: /salvar configurações/i,
    });
    await expect(saveButton).toBeEnabled();
    await saveButton.click();

    await expect(page.getByText("Configuração salva.")).toBeVisible({
      timeout: 10_000,
    });
    // Loading desapareceu — botão de volta ao estado normal.
    await expect(saveButton).toBeEnabled();

    // MSW — prova de que a UI realmente chamou a API com o payload certo,
    // não só que "parece" ter salvo.
    const lastPut = await page.evaluate(() => window.__E2E_LAST_SETTINGS_PUT__);
    expect(lastPut).toEqual({
      alert_threshold: 0.85,
      telegram_enabled: true,
      email_enabled: true,
      alert_email: "operador@predictiq-dev.com",
    });

    await page.screenshot({
      path: "e2e-results/screenshots/alert-settings-saved.png",
    });
  });
});

// ---------------------------------------------------------------------------
// TESTE 4 — Validação (bloqueio client-side, nenhuma request enviada)
// ---------------------------------------------------------------------------

test.describe("Configurações de Alertas — validação (RF-25)", () => {
  test("e-mail habilitado sem destinatário bloqueia o Salvar e não envia PUT", async ({
    page,
  }) => {
    await gotoSettings(page);

    await emailSwitch(page).click();

    const saveButton = page.getByRole("button", {
      name: /salvar configurações/i,
    });
    await expect(saveButton).toBeDisabled();
    await expect(page.getByText(/informe um e-mail válido/i)).toBeVisible();

    const lastPut = await page.evaluate(() => window.__E2E_LAST_SETTINGS_PUT__);
    expect(lastPut).toBeUndefined();
  });

  test("e-mail com formato inválido mantém o Salvar bloqueado", async ({
    page,
  }) => {
    await gotoSettings(page);

    await emailSwitch(page).click();
    await page.getByLabel("E-mail para receber alertas").fill("nao-e-email");

    await expect(
      page.getByRole("button", { name: /salvar configurações/i }),
    ).toBeDisabled();
    await expect(page.getByText(/informe um e-mail válido/i)).toBeVisible();
  });

  test("nenhum canal habilitado bloqueia o Salvar", async ({ page }) => {
    await gotoSettings(page);

    await telegramSwitch(page).click(); // desliga o único canal ligado por default

    await expect(
      page.getByRole("button", { name: /salvar configurações/i }),
    ).toBeDisabled();
    await expect(page.getByText(/habilite ao menos um canal/i)).toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// TESTE 5 — Erro do backend ao salvar
// ---------------------------------------------------------------------------

test.describe("Configurações de Alertas — erro do backend ao salvar (RF-25)", () => {
  test("PUT retorna 500 — mostra erro, mantém os dados no formulário, loading encerra, retry funciona", async ({
    page,
  }) => {
    await setSettingsScenario(page, "put-error");
    await gotoSettings(page);

    const saveButton = page.getByRole("button", {
      name: /salvar configurações/i,
    });
    await saveButton.click();

    await expect(page.getByText(/HTTP 500/i)).toBeVisible({ timeout: 10_000 });
    // Loading encerrou — botão voltou a ficar clicável.
    await expect(saveButton).toBeEnabled();
    // Estado local não foi descartado pelo erro.
    await expect(telegramSwitch(page)).toHaveAttribute("aria-checked", "true");

    // Retry — a UI não trava; o mesmo clique pode ser repetido.
    await saveButton.click();
    await expect(page.getByText(/HTTP 500/i)).toBeVisible({ timeout: 10_000 });

    await page.screenshot({
      path: "e2e-results/screenshots/alert-settings-error.png",
    });
  });
});

// ---------------------------------------------------------------------------
// Botão "Testar Notificação"
// ---------------------------------------------------------------------------

test.describe('Configurações de Alertas — "Testar Notificação" (RF-25)', () => {
  test("sucesso — feedback exibe os canais informados pelo backend", async ({
    page,
  }) => {
    await gotoSettings(page);

    const testButton = page.getByRole("button", {
      name: /testar notificação/i,
    });
    await testButton.click();

    await expect(
      page.getByRole("button", { name: /notificação enviada/i }),
    ).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText("Telegram: enviado")).toBeVisible();

    const callCount = await page.evaluate(
      () => window.__E2E_TEST_NOTIFICATION_CALL_COUNT__,
    );
    expect(callCount).toBe(1);

    await page.screenshot({
      path: "e2e-results/screenshots/alert-settings-test-notification-success.png",
    });
  });

  test("falha — mensagem de erro visível, UI não quebra", async ({ page }) => {
    await setSettingsScenario(page, "test-error");
    await gotoSettings(page);

    await page.getByRole("button", { name: /testar notificação/i }).click();

    await expect(
      page.getByRole("button", { name: /tentar novamente/i }),
    ).toBeVisible({ timeout: 10_000 });
    // `handleTest` mostra `err.message` (real, do api-client — "[api-client]
    // testAlertNotification() falhou — HTTP 502 ...") — o backend real
    // devolveria "Telegram: falhou · E-mail: falhou" como `detail`, mas o
    // api-client atual não repassa o corpo JSON do erro, só o status HTTP
    // (mesmo comportamento em toda a página, não específico deste teste).
    await expect(page.getByText(/HTTP 502/i)).toBeVisible();

    // Nenhum crash — o resto da página continua interativa.
    await expect(
      page.getByRole("button", { name: /salvar configurações/i }),
    ).toBeEnabled();

    await page.screenshot({
      path: "e2e-results/screenshots/alert-settings-test-notification-error.png",
    });
  });
});
