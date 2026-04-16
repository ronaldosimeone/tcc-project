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
