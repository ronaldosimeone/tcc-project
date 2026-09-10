// ── Constantes de configuração — RNF-58: extraído de EventLogTable ──────────

import {
  AlertOctagon,
  AlertTriangle,
  CheckCircle2,
  ClipboardCheck,
  Zap,
} from "lucide-react";
import type { EventType, Severity } from "@/lib/history-mock";

export const PAGE_SIZE = 10;

export const SEVERITY_ORDER: Record<Severity, number> = {
  CRÍTICO: 2,
  ALERTA: 1,
  NORMAL: 0,
};

export const SEVERITY_CONFIG: Record<
  Severity,
  { className: string; icon: typeof CheckCircle2 }
> = {
  CRÍTICO: {
    className: "border-red-500/40 bg-red-500/10 text-red-400",
    icon: AlertOctagon,
  },
  ALERTA: {
    className: "border-amber-500/40 bg-amber-500/10 text-amber-400",
    icon: AlertTriangle,
  },
  NORMAL: {
    className: "border-green-500/40 bg-green-500/10 text-green-400",
    icon: CheckCircle2,
  },
};

export const EVENT_TYPE_CONFIG: Record<
  EventType,
  { className: string; icon: typeof CheckCircle2 }
> = {
  Falha: { className: "text-red-400", icon: AlertOctagon },
  Alerta: { className: "text-amber-400", icon: Zap },
  Diagnóstico: { className: "text-green-400", icon: ClipboardCheck },
};

export type SortColumn = "timestamp" | "equipment" | "type" | "severity";
export type SortDir = "asc" | "desc";

export interface SortState {
  column: SortColumn;
  direction: SortDir;
}
