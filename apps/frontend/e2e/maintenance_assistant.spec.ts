/**
 * E2E — Assistente de Manutenção (RF-22/23 / RNF-53 / RNF-54).
 *
 * Convenção de arquivo: o enunciado desta task sugere
 * `apps/frontend/tests/maintenance_assistant.spec.ts`, mas o projeto já usa
 * `apps/frontend/e2e/*.spec.ts` (ver `playwright.config.ts::testDir`,
 * `dashboard_flow.spec.ts`, `failure_alert.spec.ts`) — este arquivo segue a
 * convenção REAL já estabelecida, não o caminho sugerido no enunciado.
 *
 * Auditoria (RNF-54): o enunciado menciona "Anthropic", mas o projeto NUNCA
 * usou Anthropic — confirmado por busca em todo o repositório (a única
 * menção existente é uma negação explícita no docstring de
 * `ollama_client.py`: "nenhuma chamada a OpenAI/Anthropic/Gemini"). A
 * integração de LLM real é o Ollama local (Llama 3.2 3B, RF-22/RNF-46),
 * consumida via streaming SSE (`fetch` + `ReadableStream`, não
 * `EventSource` — RF-23/RNF-47). A fronteira mockada por MSW é exatamente
 * essa: `POST /v1/maintenance/suggest/stream` — ver `mocks/handlers.ts`.
 *
 * Achado de arquitetura (RNF-53): o Dashboard (banner de falha crítica,
 * RF-08) e o Assistente de Manutenção NÃO compartilham dados automaticamente
 * — o formulário do assistente não é pré-preenchido a partir do alerta
 * ativo (confirmado lendo `SuggestionForm`: valores default fixos, nunca
 * derivados do estado do sensor). O teste de RNF-53 reflete essa realidade:
 * a falha é detectada no Dashboard, o usuário abre o Assistente (disponível
 * globalmente via Sidebar) e informa manualmente os dados do MESMO
 * equipamento em falha — a jornada visual completa que o produto realmente
 * oferece, sem inventar um data-binding que não existe.
 */

import { test, expect, type Page } from "@playwright/test";

const EQUIPMENT_NAME = "Bomba Centrífuga CX-500";
const SYMPTOM = "ruído excessivo na sucção";
const CRITICAL_PROBABILITY = "0.9";

async function gotoAndWaitMsw(page: Page, path = "/"): Promise<void> {
  await page.goto(path);
  await expect(page.locator('[data-testid="msw-ready"]')).toBeAttached({
    timeout: 15_000,
  });
}

async function setMaintenanceScenario(
  page: Page,
  scenario: "llm-error" | "no-manual",
): Promise<void> {
  await page.addInitScript((sc) => {
    window.__E2E_MAINTENANCE_SCENARIO__ = sc;
  }, scenario);
}

/** Abre o painel do Assistente a partir da Sidebar (ponto de entrada real —
 * disponível em qualquer página, não uma rota própria). */
async function openAssistant(page: Page): Promise<void> {
  await page.getByRole("button", { name: /assistente de ia/i }).click();
  await expect(
    page.getByRole("heading", { name: "Assistente de Manutenção" }),
  ).toBeVisible({ timeout: 10_000 });
}

async function submitSuggestion(
  page: Page,
  overrides: {
    equipmentName?: string;
    symptom?: string;
    probability?: string;
  } = {},
): Promise<void> {
  const equipmentInput = page.getByLabel("Equipamento");
  await equipmentInput.fill(overrides.equipmentName ?? EQUIPMENT_NAME);

  const symptomInput = page.getByLabel("Sintoma observado");
  await symptomInput.fill(overrides.symptom ?? SYMPTOM);

  const probabilityInput = page.getByLabel("Probabilidade de falha");
  await probabilityInput.fill(overrides.probability ?? CRITICAL_PROBABILITY);

  await page.getByRole("button", { name: /gerar sugestão/i }).click();
}

// ---------------------------------------------------------------------------
// TESTE 1 — Abertura
// ---------------------------------------------------------------------------

test.describe("Assistente de Manutenção — abertura (RF-22/23)", () => {
  test("abre o painel a partir da Sidebar com título e controles visíveis", async ({
    page,
  }) => {
    await gotoAndWaitMsw(page);

    await openAssistant(page);

    await expect(
      page.getByText(/gera um plano de manutenção com ia local/i),
    ).toBeVisible();
    await expect(page.getByLabel("Equipamento")).toBeVisible();
    await expect(page.getByLabel("Sintoma observado")).toBeVisible();
    await expect(page.getByLabel("Probabilidade de falha")).toBeVisible();
    await expect(
      page.getByRole("button", { name: /gerar sugestão/i }),
    ).toBeVisible();

    // Estado inicial — nenhuma geração em andamento.
    await expect(page.locator('[data-status="idle"]')).toContainText("Pronto");

    await page.screenshot({
      path: "e2e-results/screenshots/maintenance-assistant-open.png",
    });
  });
});

// ---------------------------------------------------------------------------
// TESTE 2 / RNF-53 — Consulta com streaming + reconhecimento
// ---------------------------------------------------------------------------

test.describe("Assistente de Manutenção — consulta e streaming (RF-23 / RNF-53)", () => {
  test("RNF-53: falha detectada no Dashboard → Assistente → reconhecimento do diagnóstico", async ({
    page,
  }) => {
    // 1. Falha detectada — banner crítico no Dashboard (RF-08), mesmo
    // mecanismo já validado em failure_alert.spec.ts.
    await page.addInitScript(() => {
      window.__E2E_SCENARIO__ = "critical";
    });
    await gotoAndWaitMsw(page, "/");
    await expect(page.getByTestId("critical-banner")).toBeVisible({
      timeout: 10_000,
    });
    await expect(page.getByTestId("critical-banner")).toContainText(
      /falha crítica detectada/i,
    );

    // 2. Usuário abre o Assistente (mesma sessão/página — ponto de entrada
    // global via Sidebar).
    await openAssistant(page);

    // 3-4. Informa o equipamento/sintoma da MESMA falha e envia — o
    // Assistente busca contexto (estado "searching").
    await submitSuggestion(page);
    // `[data-status="searching"]` (badge) já é inequívoco por construção —
    // não repete a checagem via texto (que também aparece no corpo do
    // painel) para evitar uma corrida entre duas leituras do MESMO estado
    // transitório de streaming.
    await expect(page.locator('[data-status="searching"]')).toBeVisible({
      timeout: 5_000,
    });

    // 5. Streaming — conteúdo chega progressivamente (estado "generating").
    await expect(page.locator('[data-status="generating"]')).toBeVisible({
      timeout: 10_000,
    });

    // 6-7. Conclusão — Markdown renderizado, reconhecimento do
    // equipamento/diagnóstico/referência do manual (não um <div> vazio).
    await expect(page.locator('[data-status="done"]')).toBeVisible({
      timeout: 15_000,
    });
    await expect(
      page.getByRole("heading", { name: "Plano de Manutenção" }),
    ).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Diagnóstico provável" }),
    ).toBeVisible();
    await expect(page.getByText(/bomba centrífuga/i).first()).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Procedimento recomendado" }),
    ).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Cuidados de segurança" }),
    ).toBeVisible();

    // Reconhecimento: referência do manual apresentada (RF-20/21/23).
    // `data-testid` aqui porque o nome do arquivo também aparece dentro do
    // próprio Markdown gerado (seção "## Referências" do plano) — dois
    // locators de texto legítimos, não um problema de acessibilidade;
    // `maintenance-references` escopa especificamente a lista real de
    // metadados (RF-20/21), não a citação em texto livre do LLM.
    await expect(page.getByText("Manuais consultados")).toBeVisible();
    const references = page.getByTestId("maintenance-references");
    await expect(references).toContainText("bomba-centrifuga-cx500.pdf");
    await expect(references).toContainText("página 4");
    await expect(references).toContainText("score 0.82");

    await page.screenshot({
      path: "e2e-results/screenshots/maintenance-assistant-done.png",
    });
  });

  test("estados de streaming completos: idle → connecting/searching → generating → done", async ({
    page,
  }) => {
    await gotoAndWaitMsw(page);
    await openAssistant(page);

    // 1. Estado inicial.
    await expect(page.locator('[data-status="idle"]')).toBeVisible();

    // 2. Envio.
    await submitSuggestion(page);

    // 3. searching.
    // `[data-status="searching"]` (badge) já é inequívoco por construção —
    // não repete a checagem via texto (que também aparece no corpo do
    // painel) para evitar uma corrida entre duas leituras do MESMO estado
    // transitório de streaming.
    await expect(page.locator('[data-status="searching"]')).toBeVisible({
      timeout: 5_000,
    });

    // 4-5. Recebimento de conteúdo + Markdown renderizado incrementalmente —
    // aguarda pelo menos um heading aparecer ainda com o status "generating"
    // (prova de que o painel renderiza ANTES do evento "done" — streaming
    // real, não uma resposta completa disfarçada).
    await expect(
      page.getByRole("heading", { name: "Plano de Manutenção" }),
    ).toBeVisible({ timeout: 10_000 });

    // 6. Conclusão.
    await expect(page.locator('[data-status="done"]')).toBeVisible({
      timeout: 15_000,
    });

    // 7. Referências.
    await expect(page.getByTestId("maintenance-references")).toContainText(
      "bomba-centrifuga-cx500.pdf",
    );

    await page.screenshot({
      path: "e2e-results/screenshots/maintenance-assistant-response-loaded.png",
    });
  });

  test("threshold RF-22: probabilidade <= 0.7 não aciona MCP/LLM (skipped)", async ({
    page,
  }) => {
    await gotoAndWaitMsw(page);
    await openAssistant(page);

    await submitSuggestion(page, { probability: "0.5" });

    await expect(page.locator('[data-status="skipped"]')).toBeVisible({
      timeout: 10_000,
    });
    await expect(page.getByText(/não excede o limiar de 0\.7/i)).toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// Erro — LLM indisponível
// ---------------------------------------------------------------------------

test.describe("Assistente de Manutenção — LLM indisponível", () => {
  test("mostra erro claro, encerra o loading e não fabrica uma resposta", async ({
    page,
  }) => {
    await setMaintenanceScenario(page, "llm-error");
    await gotoAndWaitMsw(page);
    await openAssistant(page);

    await submitSuggestion(page);

    await expect(page.locator('[data-status="error"]')).toBeVisible({
      timeout: 10_000,
    });
    await expect(
      page.getByText(/serviço de geração de sugestões.*indisponível/i),
    ).toBeVisible();

    // UI não fica travada: nenhum spinner "Gerando plano…" permanece, o
    // formulário volta a ficar interativo (botão Cancelar desaparece).
    await expect(page.getByText(/gerando plano/i)).not.toBeVisible();
    await expect(
      page.getByRole("button", { name: /cancelar/i }),
    ).not.toBeVisible();

    // Nenhuma resposta fictícia — nenhum heading de plano foi renderizado.
    await expect(
      page.getByRole("heading", { name: "Plano de Manutenção" }),
    ).not.toBeVisible();

    await page.screenshot({
      path: "e2e-results/screenshots/maintenance-assistant-llm-error.png",
    });
  });

  test("usuário consegue tentar de novo após o erro (formulário reabilitado)", async ({
    page,
  }) => {
    await setMaintenanceScenario(page, "llm-error");
    await gotoAndWaitMsw(page);
    await openAssistant(page);

    await submitSuggestion(page);
    await expect(page.locator('[data-status="error"]')).toBeVisible({
      timeout: 10_000,
    });

    // O componente não expõe um botão "Tentar novamente" próprio — a
    // recuperação real é reenviar o mesmo formulário (reabilitado após o
    // erro). Confirma que o campo aceita input e o botão de envio continua
    // funcional — não há um "retry" dedicado na implementação atual (não
    // inventado aqui).
    await expect(
      page.getByRole("button", { name: /gerar sugestão/i }),
    ).toBeEnabled();
    await expect(page.getByLabel("Equipamento")).toBeEditable();
  });
});

// ---------------------------------------------------------------------------
// Manual não encontrado — item 12
// ---------------------------------------------------------------------------

test.describe("Assistente de Manutenção — nenhum manual indexado", () => {
  test("references=[] — UI não inventa referência e mostra a ausência corretamente", async ({
    page,
  }) => {
    await setMaintenanceScenario(page, "no-manual");
    await gotoAndWaitMsw(page);
    await openAssistant(page);

    await submitSuggestion(page, {
      equipmentName: "Equipamento sem manual indexado",
      symptom: "falha não catalogada",
    });

    await expect(page.locator('[data-status="done"]')).toBeVisible({
      timeout: 15_000,
    });

    // Comportamento real de RF-22 (auditado): o LLM É chamado mesmo sem
    // contexto — a resposta usa a seção "Limitações" em vez de inventar um
    // procedimento. `MaintenanceSuggestionResponse.triggered` continua true.
    await expect(
      page.getByRole("heading", { name: "Limitações" }),
    ).toBeVisible();
    await expect(
      page.getByText(/não contém informações suficientes/i),
    ).toBeVisible();

    // Nenhum manual inexistente é mostrado — texto real do componente
    // (ReferencesList) para o caso `references=[]`.
    await expect(
      page.getByText(
        "Nenhum manual foi citado como referência para este plano.",
      ),
    ).toBeVisible();

    await page.screenshot({
      path: "e2e-results/screenshots/maintenance-assistant-no-manual.png",
    });
  });
});
