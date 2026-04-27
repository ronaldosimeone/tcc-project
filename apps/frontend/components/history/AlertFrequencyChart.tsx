"use client";

import { memo } from "react";
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
import type { DayAlertCount } from "@/lib/history-mock";

const AXIS_TICK = { fontSize: 10, fill: "hsl(215 15% 45%)" } as const;
const GRID_STROKE = "hsl(217 18% 16%)";

interface TooltipEntry {
  dataKey: string;
  value: number;
}

interface CustomTooltipProps {
  active?: boolean;
  label?: string;
  payload?: TooltipEntry[];
}

const CustomTooltip = memo(function CustomTooltip({
  active,
  label,
  payload,
}: CustomTooltipProps) {
  if (!active || !payload?.length) return null;

  const critico = payload.find((p) => p.dataKey === "critico")?.value ?? 0;
  const alerta = payload.find((p) => p.dataKey === "alerta")?.value ?? 0;
  const total = critico + alerta;

  return (
    <div className="rounded-lg border border-white/10 bg-zinc-900/95 px-3 py-2 shadow-xl backdrop-blur-sm">
      <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
        {label}
      </p>
      {total === 0 ? (
        <p className="text-xs text-zinc-500">Sem ocorrências</p>
      ) : (
        <>
          {critico > 0 && (
            <div className="flex items-center gap-2 text-xs">
              <span className="inline-block h-2 w-2 rounded-full bg-red-500" />
              <span className="text-zinc-400">Crítico</span>
              <span className="ml-auto font-bold tabular-nums text-red-400">
                {critico}
              </span>
            </div>
          )}
          {alerta > 0 && (
            <div className="flex items-center gap-2 text-xs">
              <span className="inline-block h-2 w-2 rounded-full bg-amber-400" />
              <span className="text-zinc-400">Alerta</span>
              <span className="ml-auto font-bold tabular-nums text-amber-400">
                {alerta}
              </span>
            </div>
          )}
          <div className="mt-1.5 border-t border-white/10 pt-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-zinc-500">Total</span>
              <span className="font-bold tabular-nums text-zinc-200">
                {total}
              </span>
            </div>
          </div>
        </>
      )}
    </div>
  );
});

const TOOLTIP_CONTENT = <CustomTooltip />;

interface AlertFrequencyChartProps {
  data: DayAlertCount[];
}

export default function AlertFrequencyChart({
  data,
}: AlertFrequencyChartProps) {
  const totalAlerts = data.reduce((acc, d) => acc + d.alerta + d.critico, 0);
  const peakDay = data.reduce(
    (max, d) => Math.max(max, d.alerta + d.critico),
    0,
  );

  return (
    <Card className="border-border bg-card">
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
              <p className="font-mono text-2xl font-bold tabular-nums text-amber-400">
                {peakDay}
              </p>
              <p className="text-[10px] text-muted-foreground">pico diário</p>
            </div>
          </div>
        </div>
      </CardHeader>

      <CardContent className="px-5 pb-4">
        <ResponsiveContainer width="100%" height={130}>
          <BarChart
            data={data}
            margin={{ top: 4, right: 4, bottom: 0, left: -20 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke={GRID_STROKE}
              vertical={false}
            />
            <XAxis
              dataKey="date"
              tick={AXIS_TICK}
              tickLine={false}
              axisLine={false}
              interval={1}
            />
            <YAxis
              tick={AXIS_TICK}
              tickLine={false}
              axisLine={false}
              allowDecimals={false}
              width={20}
            />
            <Tooltip
              content={TOOLTIP_CONTENT}
              cursor={{ fill: "rgba(255,255,255,0.03)" }}
            />
            <Bar
              dataKey="critico"
              name="Crítico"
              stackId="a"
              fill="#f87171"
              maxBarSize={28}
            />
            <Bar
              dataKey="alerta"
              name="Alerta"
              stackId="a"
              fill="#fbbf24"
              radius={[3, 3, 0, 0]}
              maxBarSize={28}
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
