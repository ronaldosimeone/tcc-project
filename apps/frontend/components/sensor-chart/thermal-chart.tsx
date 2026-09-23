// ── Sub-chart Elétrico/Térmico — RNF-58: extraído de sensor-chart.tsx ───────

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
import { AXIS_TICK_STYLE, GRID_STROKE, LINE_COLORS } from "./constants";
import { TOOLTIP_CONTENT } from "./chart-tooltip";
import type { PressureChartProps } from "./pressure-chart";

export const ThermalChart = memo(function ThermalChart({
  data,
  strokeWidth,
}: PressureChartProps) {
  const latest = data[data.length - 1];

  return (
    <div>
      <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
        Corrente (A) &amp; Temperatura (°C)
      </p>
      {/* Alternativa textual ao gráfico (RNF-66 §12) — ver PressureChart. */}
      <p className="sr-only">
        Leitura mais recente: corrente {latest.Motor_current.toFixed(1)} A,
        temperatura {latest.Oil_temperature.toFixed(1)}°C.
      </p>
      <div aria-hidden="true" className="contents">
        <ResponsiveContainer width="100%" height={160}>
          <LineChart
            accessibilityLayer={false}
            data={data}
            margin={{ top: 4, right: 40, bottom: 0, left: -10 }}
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
              yAxisId="current"
              domain={[0, 10]}
              tick={AXIS_TICK_STYLE}
              tickLine={false}
              axisLine={false}
              width={28}
            />
            <YAxis
              yAxisId="temp"
              orientation="right"
              domain={[50, 90]}
              tick={AXIS_TICK_STYLE}
              tickLine={false}
              axisLine={false}
              width={36}
            />
            <Tooltip content={TOOLTIP_CONTENT} />
            <Line
              yAxisId="current"
              type="monotone"
              dataKey="Motor_current"
              name="Corrente"
              stroke={LINE_COLORS.Motor_current}
              strokeWidth={strokeWidth}
              dot={false}
              activeDot={{ r: 4, fill: LINE_COLORS.Motor_current }}
              isAnimationActive={false}
            />
            <Line
              yAxisId="temp"
              type="monotone"
              dataKey="Oil_temperature"
              name="Temperatura"
              stroke={LINE_COLORS.Oil_temperature}
              strokeWidth={strokeWidth}
              dot={false}
              activeDot={{ r: 4, fill: LINE_COLORS.Oil_temperature }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
});
