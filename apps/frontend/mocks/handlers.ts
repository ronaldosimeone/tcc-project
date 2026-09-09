/**
 * MSW request handlers — RNF-17.
 *
 * These handlers intercept every fetch issued by the app in the browser and
 * return deterministic, pre-crafted payloads so E2E tests never need a live
 * backend.
 *
 * Scenario switching
 * ------------------
 * Tests can set `window.__E2E_SCENARIO__ = 'critical'` via
 * `page.addInitScript()` *before* navigating to the page.  When the
 * predict handler runs it reads this value from the window context
 * (MSW browser handlers execute in the client JS context, not the SW thread)
 * and returns the appropriate payload.
 *
 * Available scenarios
 *   (undefined)  → NORMAL  failure_probability: 0.08  predicted_class: 0
 *   'critical'   → CRÍTICO failure_probability: 0.90  predicted_class: 1
 */

import { http, HttpResponse } from "msw";

// next.js replaces NEXT_PUBLIC_* at bundle time, so this resolves to the
// literal string 'http://localhost:8000' for the E2E build.
const API_BASE = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

// ── Type augmentation for the scenario global ─────────────────────────────

declare global {
  interface Window {
    __E2E_SCENARIO__?: string;
    /** RNF-53/54 — cenário do Assistente de Manutenção (RF-23), independente
     * de `__E2E_SCENARIO__` (que é só do Dashboard/RF-08). */
    __E2E_MAINTENANCE_SCENARIO__?: string;
    /** RNF-53/54 — cenário da página /settings/alerts (RF-25). */
    __E2E_SETTINGS_SCENARIO__?: string;
    /** Última request recebida por cada handler mockado — usado pelos testes
     * para provar que a UI realmente chamou a API esperada com o payload
     * correto (não só "parece funcionar"), sem precisar de um backend real. */
    __E2E_LAST_SETTINGS_PUT__?: unknown;
    __E2E_TEST_NOTIFICATION_CALL_COUNT__?: number;
  }
}

// ── Handlers ──────────────────────────────────────────────────────────────

export const handlers = [
  /**
   * GET /api/stream/sensors — SSE em tempo real (RF-12).
   *
   * Registrado com PATH RELATIVO (sem `API_BASE`) — correção de bug real de
   * integração MSW/Playwright: `hooks/use-sensor-data.ts::SSE_URL` conecta
   * via `new EventSource("/api/stream/sensors")`, um caminho relativo que o
   * browser resolve contra a origem da PRÓPRIA página (`http://localhost:3000`,
   * o `next dev` iniciado pelo `webServer` do `playwright.config.ts` — sem
   * Nginx na frente, ao contrário da produção). Um handler registrado em
   * `${API_BASE}` (`http://localhost:8000`) nunca casa com essa origem — a
   * requisição passava para a rede real (`onUnhandledRequest: "bypass"`) e
   * recebia 404 silenciosamente, sem nenhum aviso do MSW. Path relativo no
   * handler casa com QUALQUER origem (convenção oficial do MSW v2 para
   * chamadas same-origin), exatamente como o consumidor real chama.
   *
   * Retorna um único evento sensor_reading e mantém o stream aberto.
   * MSW v2 intercepta EventSource via service worker da mesma forma que fetch.
   */
  http.get("/api/stream/sensors", () => {
    const scenario =
      typeof window !== "undefined" ? window.__E2E_SCENARIO__ : undefined;
    const isCritical = scenario === "critical";
    const isAlert = scenario === "alert";

    const reading = JSON.stringify({
      timestamp: new Date().toISOString(),
      TP2: isCritical ? 2.1 : isAlert ? 5.0 : 8.4,
      TP3: isCritical ? 1.2 : isAlert ? 5.5 : 9.1,
      H1: 8.5,
      DV_pressure: 2.1,
      Reservoirs: 8.7,
      Motor_current: isCritical ? 9.8 : isAlert ? 6.5 : 4.2,
      Oil_temperature: isCritical ? 88.5 : isAlert ? 78.0 : 68.5,
      COMP: 1.0,
      DV_eletric: 0.0,
      Towers: 1.0,
      MPG: 1.0,
      Oil_level: 1.0,
    });

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(
          encoder.encode(
            `event: sensor_reading\ndata: ${reading}\nid: 1\nretry: 3000\n\n`,
          ),
        );
        // Mantém o stream aberto (não fecha) para simular SSE contínuo
      },
    });

    return new HttpResponse(stream, {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  }),

  /**
   * POST /predict/
   * Returns a fault prediction based on the active E2E scenario.
   */
  http.post(`${API_BASE}/predict/`, () => {
    const isCritical =
      typeof window !== "undefined" && window.__E2E_SCENARIO__ === "critical";

    return HttpResponse.json(
      {
        predicted_class: isCritical ? 1 : 0,
        failure_probability: isCritical ? 0.9 : 0.08,
        // Always use the real clock so timestamps are unique across polls.
        timestamp: new Date().toISOString(),
      },
      { status: 200 },
    );
  }),

  /**
   * GET /api/v1/predictions
   *
   * Correção de bug real de integração MSW/Playwright (mesma causa do
   * handler de SSE acima): registrado com PATH RELATIVO, porque
   * `hooks/use-sensor-data.ts` chama `fetch("/api/v1/predictions?...")` —
   * caminho relativo à origem da página, não a `API_BASE`.
   *
   * Também corrigido para responder ao cenário (`__E2E_SCENARIO__`) em vez
   * de devolver sempre uma página vazia. `latest`/`riskLevel` em
   * `useSensorData` — e, por consequência, o banner crítico do `AlertPanel`
   * (RF-08) e o histórico persistido por `usePredictionHistory` (RNF-14) —
   * são alimentados EXCLUSIVAMENTE pelos itens desta resposta (seed no mount
   * + poll a cada `POLL_INTERVAL_MS`); o único outro caminho que atualiza
   * `latest` é o WebSocket `/ws/alerts`, que este MSW não simula. Uma
   * resposta sempre vazia — como antes — significa que nenhum cenário jamais
   * chega a NORMAL/ALERTA/CRÍTICO via polling, independentemente da origem
   * estar correta ou não. `timestamp` usa o relógio real (respeita
   * `page.clock` quando instalado pelo teste, pois este handler roda no
   * contexto JS da página, não na thread do Service Worker) para que cada
   * chamada gere uma entrada nova e distinta no histórico.
   */
  http.get("/api/v1/predictions", ({ request }) => {
    const url = new URL(request.url);
    const page = Number(url.searchParams.get("page") ?? "1");
    const size = Number(url.searchParams.get("size") ?? "20");

    const scenario =
      typeof window !== "undefined" ? window.__E2E_SCENARIO__ : undefined;
    const failure_probability =
      scenario === "critical" ? 0.9 : scenario === "alert" ? 0.45 : 0.08;
    const predicted_class = scenario === "critical" ? 1 : 0;

    const item = {
      timestamp: new Date().toISOString(),
      TP2: 8.4,
      TP3: 9.1,
      H1: 8.5,
      DV_pressure: 2.1,
      Reservoirs: 8.7,
      Motor_current: 4.2,
      Oil_temperature: 68.5,
      COMP: 1.0,
      DV_eletric: 0.0,
      Towers: 1.0,
      MPG: 1.0,
      Oil_level: 1.0,
      failure_probability,
      predicted_class,
    };

    return HttpResponse.json(
      { items: [item], total: 1, page, size, pages: 1 },
      { status: 200 },
    );
  }),

  // ── RF-25 / RNF-53-54 — GET/PUT /v1/settings/alerts, POST .../test ──────────
  //
  // O frontend NUNCA envia `X-Admin-Token` (auditado em lib/api-client.ts) —
  // a proteção real depende do modo dev do backend (RF-11); não há nada de
  // autenticação para simular do lado do browser.
  //
  // Cenário via `window.__E2E_SETTINGS_SCENARIO__` (addInitScript, mesmo
  // padrão de `__E2E_SCENARIO__` acima):
  //   (undefined)     → configuração default (0.85, Telegram ON, e-mail OFF)
  //   'configured'    → configuração já salva (0.75, Telegram+e-mail ON)
  //   'get-error'     → GET retorna 500
  //   'put-error'     → PUT retorna 500
  //   'test-error'    → POST .../test retorna 502

  http.get(`${API_BASE}/v1/settings/alerts`, () => {
    const scenario =
      typeof window !== "undefined"
        ? window.__E2E_SETTINGS_SCENARIO__
        : undefined;

    if (scenario === "get-error") {
      return HttpResponse.json(
        { error: "InternalServerError", detail: "Erro interno no servidor." },
        { status: 500 },
      );
    }
    if (scenario === "configured") {
      return HttpResponse.json({
        alert_threshold: 0.75,
        telegram_enabled: true,
        email_enabled: true,
        alert_email: "alertas@empresa.com.br",
      });
    }
    return HttpResponse.json({
      alert_threshold: 0.85,
      telegram_enabled: true,
      email_enabled: false,
      alert_email: null,
    });
  }),

  http.put(`${API_BASE}/v1/settings/alerts`, async ({ request }) => {
    const scenario =
      typeof window !== "undefined"
        ? window.__E2E_SETTINGS_SCENARIO__
        : undefined;
    const payload = await request.json();

    // Registrado para os testes inspecionarem via
    // page.evaluate(() => window.__E2E_LAST_SETTINGS_PUT__) — prova de que
    // a UI enviou o payload esperado, não só que "parece" ter salvo.
    if (typeof window !== "undefined") {
      window.__E2E_LAST_SETTINGS_PUT__ = payload;
    }

    if (scenario === "put-error") {
      return HttpResponse.json(
        {
          error: "InternalServerError",
          detail: "Erro ao salvar a configuração.",
        },
        { status: 500 },
      );
    }
    // Backend real devolve a configuração efetivamente salva — eco do
    // payload recebido é o equivalente determinístico aqui.
    return HttpResponse.json(payload);
  }),

  http.post(`${API_BASE}/v1/settings/alerts/test`, () => {
    const scenario =
      typeof window !== "undefined"
        ? window.__E2E_SETTINGS_SCENARIO__
        : undefined;

    if (typeof window !== "undefined") {
      window.__E2E_TEST_NOTIFICATION_CALL_COUNT__ =
        (window.__E2E_TEST_NOTIFICATION_CALL_COUNT__ ?? 0) + 1;
    }

    if (scenario === "test-error") {
      return HttpResponse.json(
        {
          error: "NotificationTestFailedError",
          detail: "Telegram: falhou · E-mail: falhou",
        },
        { status: 502 },
      );
    }
    return HttpResponse.json({ message: "Telegram: enviado" });
  }),

  // ── RF-22/23 / RNF-53-54 — POST /v1/maintenance/suggest/stream ──────────────
  //
  // Auditoria (RNF-54): RNF-54 menciona "Anthropic", mas o projeto NUNCA usou
  // Anthropic — a integração de LLM real é o Ollama local (Llama 3.2 3B,
  // RNF-46), consumido via streaming SSE por
  // `lib/maintenance-stream.ts::streamMaintenanceSuggestion` (fetch +
  // ReadableStream, NÃO EventSource — o payload precisa ir no corpo POST).
  // A fronteira mockada aqui é exatamente essa: o endpoint HTTP que o
  // frontend realmente chama, não uma integração Anthropic inexistente.
  //
  // O corpo da resposta é um ReadableStream real (não uma string única
  // disfarçada) — os eventos SSE (`searching`/`token`*/`done`|`skipped`|
  // `error`) são enfileirados com pequenos delays entre si para que os
  // estados intermediários da UI (buscando/gerando) tenham uma janela real
  // de tempo para renderizar, exercitando o parser incremental de
  // `maintenance-stream.ts` de verdade — nunca uma resposta REST completa
  // disfarçada de stream.
  //
  // Cenário via `window.__E2E_MAINTENANCE_SCENARIO__`:
  //   (undefined)   → plano completo (bomba centrífuga), com referência
  //   'llm-error'   → busca ok, LLM indisponível
  //   'no-manual'   → nenhuma referência encontrada, LLM ainda é chamado

  http.post(
    `${API_BASE}/v1/maintenance/suggest/stream`,
    async ({ request }) => {
      const scenario =
        typeof window !== "undefined"
          ? window.__E2E_MAINTENANCE_SCENARIO__
          : undefined;
      const payload = (await request.json()) as {
        failure_probability: number;
      };

      if (payload.failure_probability <= 0.7) {
        return sseStream([
          {
            event: "skipped",
            data: {
              message: `Probabilidade de falha (${payload.failure_probability.toFixed(
                2,
              )}) não excede o limiar de 0.7 — sugestão automática não acionada.`,
            },
          },
        ]);
      }

      if (scenario === "llm-error") {
        return sseStream([
          { event: "searching", data: {} },
          {
            event: "error",
            data: {
              message:
                "Serviço de geração de sugestões (Ollama) indisponível no momento.",
            },
          },
        ]);
      }

      if (scenario === "no-manual") {
        return sseStream([
          { event: "searching", data: {} },
          ...tokensFor(NO_MANUAL_MARKDOWN).map((token) => ({
            event: "token",
            data: { token },
          })),
          {
            event: "done",
            data: { markdown: NO_MANUAL_MARKDOWN, references: [] },
          },
        ]);
      }

      return sseStream([
        { event: "searching", data: {} },
        ...tokensFor(PUMP_PLAN_MARKDOWN).map((token) => ({
          event: "token",
          data: { token },
        })),
        {
          event: "done",
          data: { markdown: PUMP_PLAN_MARKDOWN, references: PUMP_REFERENCES },
        },
      ]);
    },
  ),
];

// ── Fixtures determinísticas do Assistente de Manutenção (RNF-53/54) ─────────

export const PUMP_PLAN_MARKDOWN = `# Plano de Manutenção

## Diagnóstico provável
Ruído excessivo na sucção da bomba centrífuga é compatível com desgaste do rolamento ou folga na vedação mecânica, conforme o manual técnico do equipamento.

## Procedimento recomendado
1. Desligar o equipamento e isolar a alimentação elétrica.
2. Inspecionar o rolamento quanto a folga radial.
3. Verificar a vedação mecânica e substituir se houver vazamento visível.

## Ferramentas / peças
Kit de vedação mecânica compatível com o modelo CX-500.

## Cuidados de segurança
Aguardar o resfriamento completo do equipamento antes de qualquer intervenção.

## Referências
- \`bomba-centrifuga-cx500.pdf\`, página 4
`;

export const NO_MANUAL_MARKDOWN = `# Plano de Manutenção

## Limitações
O manual recuperado não contém informações suficientes para determinar com segurança o procedimento necessário.
`;

export const PUMP_REFERENCES = [
  {
    file_name: "bomba-centrifuga-cx500.pdf",
    page: 4,
    chunk_index: 2,
    source: "bomba-centrifuga-cx500.pdf",
    score: 0.82,
  },
];

/** Divide um Markdown em blocos pequenos (linha a linha) — streaming real,
 * nunca um `.split(" ")` puramente cosmético nem uma resposta única. */
function tokensFor(markdown: string): string[] {
  const lines = markdown.split("\n");
  return lines.map((line, i) => (i < lines.length - 1 ? `${line}\n` : line));
}

/** Constrói uma resposta SSE real (ReadableStream), com um pequeno delay
 * entre eventos — dá ao React uma janela de tempo real para renderizar cada
 * estado intermediário (searching/generating), em vez de resolver tudo num
 * único microtask (o que tornaria os estados intermediários invisíveis para
 * o Playwright, mesmo existindo de verdade no código de produção). */
function sseStream(events: Array<{ event: string; data: unknown }>) {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      for (const { event, data } of events) {
        controller.enqueue(
          encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`),
        );
        await new Promise((resolve) => setTimeout(resolve, 15));
      }
      controller.close();
    },
  });
  return new HttpResponse(stream, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
