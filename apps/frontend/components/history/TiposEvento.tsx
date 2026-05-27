"use client";

/**
 * Distribuição de tipos de evento (Falha / Alerta / Diagnóstico)
 * derivada do array `events` filtrado. Donut Recharts compacto + legenda.
 */

import { useMemo } from "react";
import { Cell, Pie, PieChart, ResponsiveContainer } from "recharts";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { EventType, HistoryEvent } from "@/lib/history-mock";

interface Slice {
  name: string;
  value: number;
  pct: number;
  color: string;
}

const TYPE_META: Record<EventType, { label: string; color: string }> = {
  Falha: { label: "Falha", color: "#ef4444" },
  Alerta: { label: "Alerta", color: "#f59e0b" },
  Diagnóstico: { label: "Diag", color: "#10b981" },
};

const TYPE_ORDER: ReadonlyArray<EventType> = ["Alerta", "Falha", "Diagnóstico"];

function aggregateTypes(events: HistoryEvent[]): {
  slices: Slice[];
  total: number;
} {
  const counts: Record<EventType, number> = {
    Falha: 0,
    Alerta: 0,
    Diagnóstico: 0,
  };
  for (const e of events) counts[e.type]++;

  const total = counts.Falha + counts.Alerta + counts.Diagnóstico;
  const slices: Slice[] = TYPE_ORDER.map((type) => ({
    name: TYPE_META[type].label,
    value: counts[type],
    pct: total > 0 ? Math.round((counts[type] / total) * 100) : 0,
    color: TYPE_META[type].color,
  }));

  return { slices, total };
}

interface TiposEventoProps {
  events: HistoryEvent[];
}

export default function TiposEvento({ events }: TiposEventoProps) {
  const { slices, total } = useMemo(() => aggregateTypes(events), [events]);

  // Filtra slices com valor 0 só para o donut (legenda mostra todos).
  const donutSlices = slices.filter((s) => s.value > 0);

  return (
    <Card className="flex h-full flex-col border border-slate-200 bg-white shadow-sm ring-0">
      <CardHeader className="px-4 pb-2 pt-3">
        <CardTitle className="text-xs font-semibold uppercase tracking-wider text-slate-500">
          Tipos de evento
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col items-center justify-center gap-2 px-4 pb-4">
        <div className="relative h-20 w-20 shrink-0">
          {donutSlices.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={donutSlices}
                  dataKey="value"
                  innerRadius={24}
                  outerRadius={38}
                  paddingAngle={2}
                  stroke="none"
                  isAnimationActive={false}
                >
                  {donutSlices.map((s) => (
                    <Cell key={s.name} fill={s.color} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="absolute inset-0 rounded-full bg-slate-100" />
          )}
          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center leading-none">
            <span className="font-mono text-sm font-bold tabular-nums text-slate-900">
              {total}
            </span>
            <span className="text-[8px] uppercase tracking-wider text-slate-400">
              eventos
            </span>
          </div>
        </div>

        <ul className="flex w-full flex-col gap-0.5">
          {slices.map((s) => (
            <li
              key={s.name}
              className="flex items-center gap-1.5 text-[10px] text-slate-600"
            >
              <span
                className="h-2 w-2 shrink-0 rounded-sm"
                style={{ backgroundColor: s.color }}
                aria-hidden="true"
              />
              <span className="flex-1 truncate">{s.name}</span>
              <span className="font-mono font-semibold tabular-nums text-slate-900">
                {s.pct}%
              </span>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}
