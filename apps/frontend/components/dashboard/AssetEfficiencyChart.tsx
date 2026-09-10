"use client";

/**
 * RNF-58: decomposto em `components/dashboard/asset-efficiency-chart/*` —
 * mock-data (EFFICIENCY_BY_ASSET/estilos), ChartTooltip. Nenhuma mudança
 * de comportamento/DOM.
 */

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
import {
  AXIS_TICK,
  DEFAULT_EFFICIENCY,
  EFFICIENCY_BY_ASSET,
  GRID_STROKE,
} from "./asset-efficiency-chart/mock-data";
import { TOOLTIP_CONTENT } from "./asset-efficiency-chart/chart-tooltip";

interface AssetEfficiencyChartProps {
  assetId: string;
}

const AssetEfficiencyChart = memo(function AssetEfficiencyChart({
  assetId,
}: AssetEfficiencyChartProps) {
  const data = EFFICIENCY_BY_ASSET[assetId] ?? DEFAULT_EFFICIENCY;

  const avgEfficiency = (
    (data.reduce((acc, d) => acc + d.carga / (d.carga + d.ocioso), 0) /
      data.length) *
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
          <span className="font-mono text-[11px] font-semibold tabular-nums text-emerald-600 dark:text-emerald-400">
            {avgEfficiency}% avg
          </span>
        </div>
        <p className="text-[11px] text-muted-foreground">
          {assetId} · Horas em carga vs ocioso (24h/dia)
        </p>
      </CardHeader>

      <CardContent className="px-5 pb-5">
        <ResponsiveContainer width="100%" height={160}>
          <BarChart
            data={data}
            margin={{ top: 4, right: 8, bottom: 0, left: 0 }}
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
              width={38}
              tickFormatter={(v: number) => `${v}h`}
              domain={[0, 24]}
              ticks={[0, 6, 12, 18, 24]}
            />
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
              cursor={{ fill: "var(--muted)", fillOpacity: 0.4 }}
            />
            <Legend
              iconSize={8}
              iconType="circle"
              wrapperStyle={{
                fontSize: 11,
                color: "var(--muted-foreground)",
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
