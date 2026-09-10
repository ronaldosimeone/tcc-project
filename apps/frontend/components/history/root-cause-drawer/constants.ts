// ── Constantes/mocks — RNF-58: extraído de RootCauseDrawer ──────────────────

import {
  AlertOctagon,
  AlertTriangle,
  CheckCircle2,
  type LucideIcon,
} from "lucide-react";
import type { Severity } from "@/lib/history-mock";

// Severity tokens (literais — evita tokens hsl quebrados).
export const SEVERITY_BADGE: Record<Severity, string> = {
  CRÍTICO: "border-red-200 bg-red-50 text-red-700",
  ALERTA: "border-amber-200 bg-amber-50 text-amber-700",
  NORMAL: "border-emerald-200 bg-emerald-50 text-emerald-700",
};

export const SEVERITY_ICON: Record<Severity, LucideIcon> = {
  CRÍTICO: AlertOctagon,
  ALERTA: AlertTriangle,
  NORMAL: CheckCircle2,
};

// Mock: 2h antes da falha (1 ponto / 10 min = 13 pontos). Linha que começa
// estável, oscila no meio e dispara perto do fim, cruzando o threshold
// (0.65 → CRÍTICO). Determinístico — independente do `event`.
export const PREDICTIVE_WINDOW = [
  { t: "-2h", p: 0.12 },
  { t: "-1h50", p: 0.14 },
  { t: "-1h40", p: 0.18 },
  { t: "-1h30", p: 0.22 },
  { t: "-1h20", p: 0.21 },
  { t: "-1h10", p: 0.27 },
  { t: "-1h", p: 0.31 },
  { t: "-50min", p: 0.38 },
  { t: "-40min", p: 0.42 },
  { t: "-30min", p: 0.51 },
  { t: "-20min", p: 0.62 },
  { t: "-10min", p: 0.78 },
  { t: "0", p: 0.91 },
] as const;

export const CRITICAL_THRESHOLD = 0.65;

export interface TimelineStep {
  label: string;
  detail: string;
  tone: "warn" | "danger" | "neutral";
}

export const TIMELINE: ReadonlyArray<TimelineStep> = [
  {
    label: "Alerta Inicial",
    detail: "Vibração anômala detectada no rolamento principal",
    tone: "warn",
  },
  {
    label: "Degradação Crítica",
    detail: "Pressão DV cruza o threshold de 65% de risco",
    tone: "danger",
  },
  {
    label: "Falha do Equipamento",
    detail: "Compressor desliga; intervenção mecânica acionada",
    tone: "neutral",
  },
];

export const TIMELINE_DOT: Record<TimelineStep["tone"], string> = {
  warn: "bg-amber-500 ring-amber-200",
  danger: "bg-red-500 ring-red-200 animate-pulse",
  neutral: "bg-slate-400 ring-slate-200",
};
