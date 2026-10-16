/**
 * Simulação da API de histórico para testes — respostas no formato exato do
 * backend (`GET /v1/events`, `/v1/kpis`, `/v1/events/{id}`, work orders).
 */
import { vi } from "vitest";

import type {
  FailureEventDetailDto,
  FailureEventDto,
  KpiReportDto,
} from "@/lib/history-api";

const now = Date.now();
const iso = (minutesAgo: number): string =>
  new Date(now - minutesAgo * 60_000).toISOString();

const makeEvent = (
  id: number,
  overrides: Partial<FailureEventDto> = {},
): FailureEventDto => ({
  id,
  asset_id: id % 3 === 0 ? "APU-Trem-007" : "APU-Trem-042",
  started_at: iso(id * 300),
  ended_at: iso(id * 300 - 12),
  status: "closed",
  severity: id % 2 === 0 ? "critico" : "alerta",
  peak_probability: id % 2 === 0 ? 0.96 : 0.78,
  mean_probability: 0.8,
  n_samples: 72,
  model_name: "random_forest_v2",
  threshold: 0.7,
  duration_minutes: 12,
  work_order_status: null,
  confirmed_cause: null,
  is_false_alarm: null,
  top_sensors: [{ sensor: "Towers", contribution: 0.41 }],
  ...overrides,
});

export const EVENTS: FailureEventDto[] = Array.from({ length: 14 }, (_, i) =>
  makeEvent(i + 1),
);
EVENTS[0] = makeEvent(1, {
  work_order_status: "closed",
  confirmed_cause: "air_leak",
  is_false_alarm: false,
});

export const KPIS: KpiReportDto = {
  asset_id: null,
  window_start: iso(43_200),
  window_end: iso(0),
  observed_hours: 720,
  events_total: 14,
  confirmed_failures: 1,
  false_alarms: 0,
  unlabeled_events: 13,
  alarm_precision: 1,
  false_alarms_per_week: 0,
  mtbf_hours: 720,
  mttr_hours: 2.5,
  availability: 0.9965,
  oee: null,
  oee_note: "OEE needs production counts",
};

let workOrderStatus: "open" | "in_progress" | null = null;

const detailFor = (id: number): FailureEventDetailDto => ({
  event: EVENTS.find((e) => e.id === id) ?? EVENTS[1],
  root_cause: {
    method: "sensor_occlusion",
    created_at: iso(10),
    sensor_contributions: [
      { sensor: "Towers", contribution: 0.41 },
      { sensor: "DV_pressure", contribution: 0.12 },
    ],
    suggestion_markdown:
      "# Plano de Manutenção\n\n## Procedimento recomendado\n1. Inspecionar o dessecante.",
    references: [],
    grounding: {
      status: "grounded",
      invalid_citations: [],
      injection_chunks_dropped: 0,
      safety_violations: [],
    },
    llm_model: "llama3.2:3b",
  },
  work_order:
    workOrderStatus === null
      ? null
      : {
          id: 9,
          event_id: id,
          asset_id: "APU-Trem-042",
          status: workOrderStatus,
          assignee: null,
          opened_at: iso(5),
          started_at: workOrderStatus === "in_progress" ? iso(4) : null,
          closed_at: null,
          parts: [],
          confirmed_cause: null,
          is_false_alarm: null,
          notes: null,
        },
  sensor_window: [
    { timestamp: iso(id * 300 + 60), failure_probability: 0.2, values: {} },
    { timestamp: iso(id * 300), failure_probability: 0.9, values: {} },
  ],
});

export const json = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

export const fetchMock = vi.fn(
  async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = new URL(String(input), "http://test");
    const method = init?.method ?? "GET";
    if (url.pathname === "/api/v1/events")
      return json({ items: EVENTS, next_cursor: null });
    if (url.pathname === "/api/v1/kpis") return json(KPIS);
    if (url.pathname === "/api/v1/assets") {
      return json(
        ["APU-Trem-042", "APU-Trem-007", "APU-Trem-001"].map((id) => ({
          id,
          name: id,
          installation: "Metro do Porto",
          system: "APU",
          equipment_unit: "Compressor",
          subunit: "Compressor + secador",
          manufacturer_model: null,
          criticality: 2,
        })),
      );
    }
    const detail = url.pathname.match(/^\/api\/v1\/events\/(\d+)$/);
    if (detail) return json(detailFor(Number(detail[1])));
    if (url.pathname === "/api/v1/work-orders" && method === "POST") {
      workOrderStatus = "open";
      return json({}, 201);
    }
    if (url.pathname === "/api/v1/work-orders/9" && method === "PATCH") {
      workOrderStatus = "in_progress";
      return json({});
    }
    return json({ detail: "not found" }, 404);
  },
);

/** Instala o fetch simulado + NEXT_PUBLIC_API_URL; chame em beforeEach. */
export const installHistoryApiMock = (): void => {
  workOrderStatus = null;
  vi.stubEnv("NEXT_PUBLIC_API_URL", "/api");
  vi.stubGlobal("fetch", fetchMock);
};

export const uninstallHistoryApiMock = (): void => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  fetchMock.mockClear();
};
