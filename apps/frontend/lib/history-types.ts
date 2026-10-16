/**
 * Tipos de UI do Histórico + utilitários de formatação.
 *
 * Substitui `history-mock.ts`: os eventos agora vêm de `GET /v1/events`
 * (episódios RN-03 persistidos pelo backend) e são convertidos para este
 * formato por `toHistoryEvent` (lib/history-api.ts).
 */

export type Severity = "NORMAL" | "ALERTA" | "CRÍTICO";
/** "Falha" = falha confirmada pelo técnico na OS; "Alerta" = ainda não confirmada ou falso alarme. */
export type EventType = "Diagnóstico" | "Alerta" | "Falha";

export interface SensorContribution {
  sensor: string;
  contribution: number;
}

export interface HistoryEvent {
  /** Rótulo de exibição (EVT-123). */
  id: string;
  /** Id numérico do evento no backend. */
  eventId: number;
  timestamp: string;
  equipment: string;
  type: EventType;
  severity: Severity;
  /** Duração formatada ("1h 12min", "em andamento"). */
  duration: string;
  /** Duração em minutos; null enquanto o evento está aberto. */
  durationMinutes: number | null;
  description: string;
  peakProbability: number;
  workOrderStatus: WorkOrderStatus | null;
  confirmedCause: FailureCause | null;
  isFalseAlarm: boolean | null;
  topSensors: SensorContribution[];
}

export type WorkOrderStatus = "open" | "in_progress" | "closed" | "cancelled";

export type FailureCause =
  | "air_leak"
  | "oil_system"
  | "air_drier"
  | "valve"
  | "electrical_motor"
  | "sensor_fault"
  | "other";

export const FAILURE_CAUSE_LABEL: Record<FailureCause, string> = {
  air_leak: "Vazamento de ar",
  oil_system: "Sistema de óleo",
  air_drier: "Secador de ar",
  valve: "Válvula",
  electrical_motor: "Motor elétrico",
  sensor_fault: "Falha de sensor",
  other: "Outra",
};

export const WORK_ORDER_STATUS_LABEL: Record<WorkOrderStatus, string> = {
  open: "Aberta",
  in_progress: "Em execução",
  closed: "Fechada",
  cancelled: "Cancelada",
};

/** Formata minutos numéricos como "Xh Ymin" ou "Ymin" para exibição. */
export const formatMinutes = (totalMin: number): string => {
  if (totalMin <= 0) return "—";
  if (totalMin < 1) return "< 1min";
  const h = Math.floor(totalMin / 60);
  const m = Math.round(totalMin - h * 60);
  if (h === 0) return `${m}min`;
  if (m === 0) return `${h}h`;
  return `${h}h ${m}min`;
};

/** Formata horas (KPIs) em "Xh" / "X,Y dias". */
export const formatHours = (hours: number | null): string => {
  if (hours === null) return "—";
  if (hours < 48) return `${hours.toFixed(1)}h`;
  return `${(hours / 24).toFixed(1)} dias`;
};
