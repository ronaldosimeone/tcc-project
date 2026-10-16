// ── Constantes visuais do RootCauseDrawer ───────────────────────────────────
// Os dados (janela de probabilidade, linha do tempo, diagnóstico) vêm de
// `GET /v1/events/{id}`; aqui ficam só os mapeamentos de severidade → estilo.

import {
  AlertOctagon,
  AlertTriangle,
  CheckCircle2,
  type LucideIcon,
} from "lucide-react";
import type { Severity } from "@/lib/history-types";

export const SEVERITY_BADGE: Record<Severity, string> = {
  CRÍTICO: "border-destructive/30 bg-destructive/10 text-destructive",
  ALERTA: "border-chart-4/30 bg-chart-4/10 text-foreground",
  NORMAL: "border-chart-2/30 bg-chart-2/10 text-foreground",
};

export const SEVERITY_ICON: Record<Severity, LucideIcon> = {
  CRÍTICO: AlertOctagon,
  ALERTA: AlertTriangle,
  NORMAL: CheckCircle2,
};
