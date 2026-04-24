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
  }
}

// ── Handlers ──────────────────────────────────────────────────────────────

export const handlers = [
  /**
   * GET /api/stream/sensors — SSE em tempo real (RF-12).
   *
   * Retorna um único evento sensor_reading e mantém o stream aberto.
   * MSW v2 intercepta EventSource via service worker da mesma forma que fetch.
   */
  http.get(`${API_BASE}/api/stream/sensors`, () => {
    const isCritical =
      typeof window !== "undefined" && window.__E2E_SCENARIO__ === "critical";

    const reading = JSON.stringify({
      timestamp: new Date().toISOString(),
      TP2: isCritical ? 2.1 : 8.4,
      TP3: isCritical ? 1.2 : 9.1,
      H1: 8.5,
      DV_pressure: 2.1,
      Reservoirs: 8.7,
      Motor_current: isCritical ? 9.8 : 4.2,
      Oil_temperature: isCritical ? 88.5 : 68.5,
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
   * Returns an empty paginated response so the history endpoint never errors.
   * The AlertPanel history is driven by usePredictionHistory (localStorage),
   * not this endpoint, so an empty response is correct for E2E.
   */
  http.get(`${API_BASE}/api/v1/predictions`, ({ request }) => {
    const url = new URL(request.url);
    const page = Number(url.searchParams.get("page") ?? "1");
    const size = Number(url.searchParams.get("size") ?? "20");

    return HttpResponse.json(
      { items: [], total: 0, page, size, pages: 0 },
      { status: 200 },
    );
  }),
];
