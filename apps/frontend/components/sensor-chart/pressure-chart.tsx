// ── Sub-chart de Pressão (TP2 + TP3) — RNF-58: extraído de sensor-chart.tsx ─

import { memo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SensorDataPoint } from "@/hooks/use-sensor-data";
import { AXIS_TICK_STYLE, GRID_STROKE, LINE_COLORS } from "./constants";
import { TOOLTIP_CONTENT } from "./chart-tooltip";

export interface PressureChartProps {
  data: SensorDataPoint[];
  strokeWidth: number;
}

export const PressureChart = memo(function PressureChart({
  data,
  strokeWidth,
}: PressureChartProps) {
  return (
    <div>
      <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
        Pressão &mdash; TP2 &amp; TP3 (bar)
      </p>
      <ResponsiveContainer width="100%" height={160}>
        <LineChart
          data={data}
          margin={{ top: 4, right: 8, bottom: 0, left: -10 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke={GRID_STROKE}
            vertical={false}
          />
          <XAxis
            dataKey="time"
            tick={AXIS_TICK_STYLE}
            tickLine={false}
            axisLine={false}
            interval="preserveStartEnd"
          />
          <YAxis
            domain={[0, 12]}
            tick={AXIS_TICK_STYLE}
            tickLine={false}
            axisLine={false}
            width={28}
          />
          <Tooltip content={TOOLTIP_CONTENT} />
          <Line
            type="monotone"
            dataKey="TP2"
            name="TP2"
            stroke={LINE_COLORS.TP2}
            strokeWidth={strokeWidth}
            dot={false}
            activeDot={{ r: 4, fill: LINE_COLORS.TP2 }}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="TP3"
            name="TP3"
            stroke={LINE_COLORS.TP3}
            strokeWidth={strokeWidth}
            dot={false}
            activeDot={{ r: 4, fill: LINE_COLORS.TP3 }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
});
