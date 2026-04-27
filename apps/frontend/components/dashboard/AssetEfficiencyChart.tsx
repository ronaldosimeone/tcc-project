"use client";

import { memo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { BarChart2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

// ── Mock data ─────────────────────────────────────────────────────────────

const EFFICIENCY_DATA = [
  { day: "Seg", carga: 22, ocioso: 2 },
  { day: "Ter", carga: 21, ocioso: 3 },
  { day: "Qua", carga: 23, ocioso: 1 },
  { day: "Qui", carga: 20, ocioso: 4 },
  { day: "Sex", carga: 22, ocioso: 2 },
  { day: "Sáb", carga: 18, ocioso: 6 },
  { day: "Dom", carga: 16, ocioso: 8 },
];

// ── Estilos ───────────────────────────────────────────────────────────────

const AXIS_TICK = { fontSize: 10, fill: "hsl(215 15% 45%)" } as const;
const GRID_STROKE = "hsl(217 18% 16%)";

// ── Tooltip ───────────────────────────────────────────────────────────────

interface TooltipEntry {
  name: string;
  value: number;
  color: string;
  dataKey: string;
}

interface TooltipProps {
  active?: boolean;
  label?: string;
  payload?: TooltipEntry[];
}

const ChartTooltip = memo(function ChartTooltip({
  active,
  label,
  payload,
}: TooltipProps) {
  if (!active || !payload?.length) return null;

  const carga = payload.find((p) => p.dataKey === "carga")?.value ?? 0;
  const ocioso = payload.find((p) => p.dataKey === "ocioso")?.value ?? 0;
  const total = carga + ocioso;
  const efficiency = total > 0 ? ((carga / total) * 100).toFixed(0) : "0";

  return (
    <div className="rounded-lg border border-zinc-200/60 bg-white/60 px-3 py-2 shadow-md backdrop-blur-md">
      <div className="mb-2 flex items-center justify-between gap-4">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
          {label}
        </p>
        <span className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-[10px] font-bold text-zinc-700">
          {efficiency}% efic.
        </span>
      </div>
      {payload.map((entry) => (
        <div key={entry.name} className="flex items-center gap-2 text-xs">
          <span
            className="inline-block h-2 w-2 rounded-full"
            style={{ background: entry.color }}
          />
          <span className="text-zinc-600">{entry.name}</span>
          <span className="ml-auto font-bold tabular-nums text-zinc-900">
            {entry.value}h
          </span>
        </div>
      ))}
      <div className="mt-1.5 border-t border-zinc-200/60 pt-1.5">
        <div className="flex items-center justify-between text-xs">
          <span className="text-zinc-500">Total</span>
          <span className="font-bold tabular-nums text-zinc-900">{total}h</span>
        </div>
      </div>
    </div>
  );
});

const TOOLTIP_CONTENT = <ChartTooltip />;

// ── Component ─────────────────────────────────────────────────────────────

const AssetEfficiencyChart = memo(function AssetEfficiencyChart() {
  const avgEfficiency = (
    (EFFICIENCY_DATA.reduce(
      (acc, d) => acc + d.carga / (d.carga + d.ocioso),
      0,
    ) /
      EFFICIENCY_DATA.length) *
    100
  ).toFixed(0);

  return (
    <Card className="border-border bg-card">
      <CardHeader className="px-5 pb-2 pt-4">
        <div className="flex items-start justify-between">
          <CardTitle className="flex items-center gap-2 text-sm font-semibold text-foreground/90">
            <BarChart2 className="h-4 w-4 text-primary" />
            Eficiência Semanal
          </CardTitle>
          <span className="font-mono text-[11px] font-semibold text-green-400">
            {avgEfficiency}% avg
          </span>
        </div>
        <p className="text-[11px] text-muted-foreground">
          APU-Trem-042 · Horas em carga vs ocioso (24h/dia)
        </p>
      </CardHeader>

      <CardContent className="px-5 pb-5">
        <ResponsiveContainer width="100%" height={160}>
          <BarChart
            data={EFFICIENCY_DATA}
            margin={{ top: 4, right: 8, bottom: 0, left: -10 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke={GRID_STROKE}
              vertical={false}
            />
            <XAxis
              dataKey="day"
              tick={AXIS_TICK}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              tick={AXIS_TICK}
              tickLine={false}
              axisLine={false}
              width={30}
              tickFormatter={(v: number) => `${v}h`}
              domain={[0, 24]}
              ticks={[0, 6, 12, 18, 24]}
            />
            {/* Meta de eficiência mínima: 20h em carga */}
            <ReferenceLine
              y={20}
              stroke="hsl(38 92% 50%)"
              strokeDasharray="4 2"
              strokeWidth={1}
              label={{
                value: "Meta",
                position: "insideTopRight",
                fontSize: 9,
                fill: "hsl(38 92% 50%)",
              }}
            />
            <Tooltip
              content={TOOLTIP_CONTENT}
              cursor={{ fill: "hsl(217 18% 14%)" }}
            />
            <Legend
              iconSize={8}
              iconType="circle"
              wrapperStyle={{
                fontSize: 11,
                color: "hsl(215 15% 50%)",
                paddingTop: 6,
              }}
            />
            <Bar
              dataKey="carga"
              name="Em Carga"
              stackId="a"
              fill="#60a5fa"
              radius={[0, 0, 0, 0]}
              maxBarSize={36}
            />
            <Bar
              dataKey="ocioso"
              name="Ocioso"
              stackId="a"
              fill="#c084fc"
              radius={[3, 3, 0, 0]}
              maxBarSize={36}
            />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
});

export default AssetEfficiencyChart;
