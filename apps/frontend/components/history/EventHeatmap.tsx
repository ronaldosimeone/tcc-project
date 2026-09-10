"use client";

/**
 * Event Heatmap — matriz de calor operacional (Turnos × Dias da semana).
 *
 * Recharts não oferece heatmap nativo; implementação pura com CSS Grid +
 * Tailwind, mantendo o estilo SCADA do resto do dashboard.
 *
 * Eixos
 * -----
 *   Linhas  : Turnos (00h, 06h, 12h, 18h) — 4 slots de 6 horas
 *   Colunas : Dias da semana (Seg → Dom) — 7 colunas
 *
 * Cores (severidade por densidade de eventos no turno)
 * ----------------------------------------------------
 *   0          → slate-100  (sem ocorrências)
 *   1–2        → emerald-500 (normal)
 *   3–5        → amber-500   (alerta)
 *   ≥ 6        → red-500 + animate-pulse (crítico)
 *
 * RNF-58: decomposto em `components/history/event-heatmap/*` — helpers
 * (eixos/matriz/cores), Cell, RowGroup, Swatch. Nenhuma mudança de
 * comportamento/DOM.
 */

import { useMemo } from "react";
import { Activity } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { HistoryEvent } from "@/lib/history-mock";
import { buildMatrix, DAYS, SHIFTS } from "./event-heatmap/helpers";
import { RowGroup } from "./event-heatmap/row-group";
import { Swatch } from "./event-heatmap/swatch";

// ── Componente ──────────────────────────────────────────────────────────────

interface EventHeatmapProps {
  events: HistoryEvent[];
}

export default function EventHeatmap({ events }: EventHeatmapProps) {
  const matrix = useMemo(() => buildMatrix(events), [events]);
  const totalEvents = useMemo(
    () => matrix.flat().reduce((sum, c) => sum + c.total, 0),
    [matrix],
  );

  return (
    <Card className="flex h-full flex-col overflow-visible border border-slate-200 bg-white shadow-sm ring-0">
      <CardHeader className="px-5 pb-2 pt-4">
        <div className="flex items-baseline justify-between">
          <CardTitle className="flex items-center gap-2 text-sm font-semibold text-slate-900">
            <Activity className="h-4 w-4 text-slate-500" />
            Mapa de Calor de Ocorrências
          </CardTitle>
          <p className="text-[10px] font-medium uppercase tracking-wider text-slate-400">
            Turno × Dia da semana · {totalEvents} eventos
          </p>
        </div>
      </CardHeader>

      <CardContent className="flex flex-1 flex-col justify-center px-5 pb-5">
        {/* Eixos discretos: cabeçalho de dias no topo, turnos à esquerda.
            Grid 8 cols = 1 (turno label) + 7 (dias). */}
        <div className="grid grid-cols-[max-content_repeat(7,minmax(0,1fr))] gap-1.5">
          {/* Linha 0: cabeçalho de dias */}
          <span aria-hidden="true" />
          {DAYS.map((day) => (
            <span
              key={day}
              className="text-center font-mono text-[10px] uppercase tracking-wider text-slate-400"
            >
              {day}
            </span>
          ))}

          {/* Linhas dos turnos */}
          {SHIFTS.map((shift, rowIdx) => (
            <RowGroup key={shift} shift={shift} cells={matrix[rowIdx] ?? []} />
          ))}
        </div>

        {/* Legenda — escala de intensidade. */}
        <div className="mt-4 flex items-center gap-3 border-t border-slate-100 pt-3 text-[10px] text-slate-500">
          <span className="font-medium uppercase tracking-wider text-slate-400">
            Intensidade
          </span>
          <Swatch className="bg-slate-100" label="0" />
          <Swatch className="bg-emerald-500" label="1–2" />
          <Swatch className="bg-amber-500" label="3–5" />
          <Swatch className="bg-red-500" label="≥ 6" />
        </div>
      </CardContent>
    </Card>
  );
}
