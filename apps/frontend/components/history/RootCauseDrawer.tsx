"use client";

/**
 * Root Cause Drawer — análise preditiva de um evento histórico.
 *
 * Abre da direita ao clicar numa linha do EventLogTable. Conteúdo:
 *  1. Cabeçalho — descrição do alerta + badge de severidade.
 *  2. Janela de 2h antes da falha — mini LineChart (Recharts) com
 *     ReferenceLine tracejada marcando o threshold de probabilidade.
 *  3. Causa raiz da IA — bloco indigo com o diagnóstico mockado.
 *  4. Timeline — 3 pontos: Alerta inicial → Degradação crítica → Falha.
 *
 * O drawer é totalmente client-side (puro Tailwind, sem libs extra além
 * do Sheet já existente). Recebe `event` (ou null) e `onClose`.
 *
 * RNF-58: decomposto em `components/history/root-cause-drawer/*` —
 * constants (severidade/mocks/timeline), PredictiveTooltip,
 * PredictiveWindowChart, AiDiagnosis, IncidentTimeline. Nenhuma mudança de
 * comportamento/DOM.
 */

import { AlertOctagon } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { cn } from "@/lib/utils";
import type { HistoryEvent } from "@/lib/history-mock";
import { AiDiagnosis } from "./root-cause-drawer/ai-diagnosis";
import { SEVERITY_BADGE, SEVERITY_ICON } from "./root-cause-drawer/constants";
import { IncidentTimeline } from "./root-cause-drawer/incident-timeline";
import { PredictiveWindowChart } from "./root-cause-drawer/predictive-window-chart";

// ── Componente principal ────────────────────────────────────────────────────

interface RootCauseDrawerProps {
  event: HistoryEvent | null;
  onClose: () => void;
}

export default function RootCauseDrawer({
  event,
  onClose,
}: RootCauseDrawerProps) {
  const open = event !== null;
  const SevIcon = event ? SEVERITY_ICON[event.severity] : AlertOctagon;
  const badgeCls = event
    ? SEVERITY_BADGE[event.severity]
    : SEVERITY_BADGE.ALERTA;

  return (
    <Sheet
      open={open}
      onOpenChange={(o) => {
        if (!o) onClose();
      }}
    >
      <SheetContent
        side="right"
        className="w-full overflow-y-auto sm:max-w-[500px]"
      >
        <SheetHeader>
          <div className="flex items-start justify-between gap-3 pr-8">
            <div className="min-w-0 flex-1">
              <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-500">
                Análise de Causa Raiz
              </p>
              <SheetTitle className="mt-0.5 text-base font-semibold text-slate-900">
                {event?.description ?? "—"}
              </SheetTitle>
              <SheetDescription className="mt-1 font-mono text-[11px] text-slate-500">
                {event?.equipment ?? ""} · {event?.timestamp ?? ""}
              </SheetDescription>
            </div>
            {event && (
              <Badge
                variant="outline"
                className={cn("gap-1 text-[11px] font-semibold", badgeCls)}
              >
                <SevIcon className="h-3 w-3" />
                {event.severity}
              </Badge>
            )}
          </div>
        </SheetHeader>

        {event && (
          <div className="flex flex-col gap-5 px-6 pb-6">
            <PredictiveWindowChart />
            <AiDiagnosis />
            <IncidentTimeline />
          </div>
        )}
      </SheetContent>
    </Sheet>
  );
}
