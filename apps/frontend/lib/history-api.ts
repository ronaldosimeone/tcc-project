/**
 * Cliente da API de histórico (backend `src/routers/history.py`).
 *
 * Eventos = episódios de alarme (RN-03) gravados pelo pipeline; KPIs só
 * contam falhas confirmadas pelo técnico na OS (ground truth).
 */

import dayjs from "dayjs";

import { resolveBaseUrl } from "@/lib/api-client";
import { apiFetch } from "@/lib/auth";
import {
  formatMinutes,
  type FailureCause,
  type HistoryEvent,
  type SensorContribution,
  type WorkOrderStatus,
} from "@/lib/history-types";
import { riskIndex } from "@/lib/risk-thresholds";

// ── DTOs (espelham src/schemas/history.py) ───────────────────────────────

export interface FailureEventDto {
  id: number;
  asset_id: string;
  started_at: string;
  ended_at: string | null;
  status: "open" | "closed";
  severity: "alerta" | "critico";
  peak_probability: number;
  mean_probability: number;
  n_samples: number;
  model_name: string | null;
  threshold: number;
  duration_minutes: number | null;
  work_order_status: WorkOrderStatus | null;
  confirmed_cause: FailureCause | null;
  is_false_alarm: boolean | null;
  top_sensors: SensorContribution[];
}

export interface EventPage {
  items: FailureEventDto[];
  next_cursor: string | null;
}

export interface SensorPoint {
  timestamp: string;
  failure_probability: number;
  values: Record<string, number>;
}

export interface GroundingDto {
  status: string;
  invalid_citations: { file_name: string; page: number }[];
  injection_chunks_dropped: number;
  safety_violations: string[];
}

export interface RootCauseDto {
  method: string;
  created_at: string;
  sensor_contributions: SensorContribution[];
  suggestion_markdown: string | null;
  references: { file_name: string; page: number; score: number }[];
  grounding: GroundingDto | null;
  llm_model: string | null;
}

export interface WorkOrderDto {
  id: number;
  event_id: number;
  asset_id: string;
  status: WorkOrderStatus;
  assignee: string | null;
  opened_at: string;
  started_at: string | null;
  closed_at: string | null;
  parts: { part: string; quantity: number }[];
  confirmed_cause: FailureCause | null;
  is_false_alarm: boolean | null;
  notes: string | null;
}

export interface FailureEventDetailDto {
  event: FailureEventDto;
  root_cause: RootCauseDto | null;
  work_order: WorkOrderDto | null;
  sensor_window: SensorPoint[];
}

export interface KpiReportDto {
  asset_id: string | null;
  window_start: string;
  window_end: string;
  observed_hours: number;
  events_total: number;
  confirmed_failures: number;
  false_alarms: number;
  unlabeled_events: number;
  alarm_precision: number | null;
  false_alarms_per_week: number;
  mtbf_hours: number | null;
  mttr_hours: number | null;
  availability: number | null;
  oee: number | null;
  oee_note: string;
}

export interface WorkOrderUpdatePayload {
  status?: WorkOrderStatus;
  assignee?: string;
  confirmed_cause?: FailureCause;
  is_false_alarm?: boolean;
  notes?: string;
}

// ── HTTP ─────────────────────────────────────────────────────────────────

const getJson = async <T>(path: string): Promise<T> => {
  const response = await apiFetch(`${resolveBaseUrl()}${path}`, {
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(
      `[history-api] GET ${path} falhou — HTTP ${response.status}`,
    );
  }
  return response.json() as Promise<T>;
};

const sendJson = async <T>(
  method: "POST" | "PATCH",
  path: string,
  body: unknown,
): Promise<T> => {
  const response = await apiFetch(`${resolveBaseUrl()}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    const message =
      detail && typeof detail === "object" && "detail" in detail
        ? String(detail.detail)
        : `HTTP ${response.status}`;
    throw new Error(message);
  }
  return response.json() as Promise<T>;
};

/** Todas as páginas de eventos com início em [start, end) — até `max`. */
export const fetchEventsInWindow = async ({
  start,
  end,
  max = 2000,
}: {
  start: string;
  end: string;
  max?: number;
}): Promise<FailureEventDto[]> => {
  const out: FailureEventDto[] = [];
  let cursor: string | null = null;
  do {
    const params = new URLSearchParams({ start, end, limit: "500" });
    if (cursor) params.set("cursor", cursor);
    const page: EventPage = await getJson<EventPage>(
      `/v1/events?${params.toString()}`,
    );
    out.push(...page.items);
    cursor = page.next_cursor;
  } while (cursor && out.length < max);
  return out;
};

export interface AssetDto {
  id: string;
  name: string;
  installation: string;
  system: string;
  equipment_unit: string;
  subunit: string;
  manufacturer_model: string | null;
  criticality: number;
}

export const fetchAssets = (): Promise<AssetDto[]> =>
  getJson<AssetDto[]>("/v1/assets");

export const fetchKpis = ({
  start,
  end,
}: {
  start: string;
  end: string;
}): Promise<KpiReportDto> =>
  getJson<KpiReportDto>(
    `/v1/kpis?${new URLSearchParams({ start, end }).toString()}`,
  );

export const fetchEventDetail = (
  eventId: number,
): Promise<FailureEventDetailDto> =>
  getJson<FailureEventDetailDto>(`/v1/events/${eventId}`);

export const createWorkOrder = (eventId: number): Promise<WorkOrderDto> =>
  sendJson<WorkOrderDto>("POST", "/v1/work-orders", { event_id: eventId });

export const updateWorkOrder = (
  id: number,
  payload: WorkOrderUpdatePayload,
): Promise<WorkOrderDto> =>
  sendJson<WorkOrderDto>("PATCH", `/v1/work-orders/${id}`, payload);

// ── Conversão para o formato de UI ───────────────────────────────────────

const describe = (dto: FailureEventDto): string => {
  const peak = `pico de risco ${Math.round(
    riskIndex(dto.peak_probability, dto.threshold) * 100,
  )}%`;
  const sensors = dto.top_sensors
    .filter((s) => s.contribution > 0)
    .map((s) => s.sensor);
  const signals = sensors.length > 0 ? ` — sinais: ${sensors.join(", ")}` : "";
  if (dto.is_false_alarm === true) return `Falso alarme confirmado (${peak})`;
  return `Episódio de alarme, ${peak}${signals}`;
};

export const toHistoryEvent = (dto: FailureEventDto): HistoryEvent => ({
  id: `EVT-${dto.id}`,
  eventId: dto.id,
  timestamp: dto.started_at,
  equipment: dto.asset_id,
  type: dto.is_false_alarm === false ? "Falha" : "Alerta",
  severity: dto.severity === "critico" ? "CRÍTICO" : "ALERTA",
  duration:
    dto.duration_minutes === null
      ? "em andamento"
      : formatMinutes(dto.duration_minutes),
  durationMinutes: dto.duration_minutes,
  description: describe(dto),
  peakProbability: dto.peak_probability,
  workOrderStatus: dto.work_order_status,
  confirmedCause: dto.confirmed_cause,
  isFalseAlarm: dto.is_false_alarm,
  topSensors: dto.top_sensors,
});

/** Janela [now − days, now) em ISO, para períodos relativos ("7", "30"…). */
export const periodWindow = (
  period: string,
): { start: string; end: string } => {
  const end = dayjs();
  const days = period === "all" ? 3650 : Number.parseInt(period, 10);
  return {
    start: end.subtract(days, "day").toISOString(),
    end: end.add(1, "minute").toISOString(),
  };
};
