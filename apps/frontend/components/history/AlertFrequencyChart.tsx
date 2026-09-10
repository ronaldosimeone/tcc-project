"use client";

/**
 * RNF-58: decomposto em `components/history/alert-frequency-chart/*` —
 * helpers (bucketização diária), CustomTooltip. Nenhuma mudança de
 * comportamento/DOM.
 */

import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { TrendingUp } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { HistoryEvent } from "@/lib/history-mock";
import {
  AXIS_TICK,
  bucketEventsByDay,
  GRID_STROKE,
} from "./alert-frequency-chart/helpers";
import { TOOLTIP_CONTENT } from "./alert-frequency-chart/chart-tooltip";

interface AlertFrequencyChartProps {
  events: HistoryEvent[];
}

export default function AlertFrequencyChart({
  events,
}: AlertFrequencyChartProps) {
  const data = useMemo(() => bucketEventsByDay(events), [events]);
  const totalAlerts = data.reduce((acc, d) => acc + d.alerta + d.critico, 0);
  const peakDay = data.reduce(
    (max, d) => Math.max(max, d.alerta + d.critico),
    0,
  );

  return (
    <Card className="flex h-full flex-col border border-slate-200 bg-white shadow-sm ring-0">
      <CardHeader className="px-5 pb-2 pt-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <CardTitle className="flex items-center gap-2 text-sm font-semibold text-foreground/90">
              <TrendingUp className="h-4 w-4 text-primary" />
              Frequência de Ocorrências
            </CardTitle>
            <p className="mt-0.5 text-[11px] text-muted-foreground">
              Alertas e falhas por dia · Últimos 14 dias
            </p>
          </div>

          <div className="flex items-center gap-6 text-right">
            <div>
              <p className="font-mono text-2xl font-bold tabular-nums text-foreground">
                {totalAlerts}
              </p>
              <p className="text-[10px] text-muted-foreground">
                total no período
              </p>
            </div>
            <div>
              <p className="font-mono text-2xl font-bold tabular-nums tracking-tight text-amber-600 dark:text-amber-400">
                {peakDay}
              </p>
              <p className="text-[10px] text-muted-foreground">pico diário</p>
            </div>
          </div>
        </div>
      </CardHeader>

      {/* `flex-1 min-h-[200px]` faz o gráfico esticar verticalmente até o
          espaço residual do grid (100dvh layout) sem cair abaixo do mínimo
          legível em viewports baixos. */}
      <CardContent className="flex flex-1 min-h-[200px] flex-col px-5 pb-4">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            margin={{ top: 8, right: 4, bottom: 0, left: -10 }}
            barSize={14}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              vertical={false}
              stroke={GRID_STROKE}
            />
            <XAxis
              dataKey="date"
              tick={AXIS_TICK}
              tickLine={false}
              axisLine={false}
              interval={1}
            />
            <YAxis
              stroke="#94a3b8"
              fontSize={11}
              tickLine={false}
              axisLine={false}
              allowDecimals={false}
              width={28}
            />
            <Tooltip
              content={TOOLTIP_CONTENT}
              cursor={{ fill: "#f1f5f9", fillOpacity: 0.6 }}
            />
            <Bar dataKey="critico" name="Crítico" stackId="a" fill="#ef4444" />
            <Bar
              dataKey="alerta"
              name="Alerta"
              stackId="a"
              fill="#f59e0b"
              radius={[3, 3, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>

        <div className="mt-3 flex items-center justify-end gap-5 border-t border-border pt-3">
          <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
            <span className="inline-block h-2 w-2 rounded-full bg-red-400" />
            Falha Crítica
          </div>
          <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
            <span className="inline-block h-2 w-2 rounded-full bg-amber-400" />
            Alerta
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
