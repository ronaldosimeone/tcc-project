"use client";

// ── Sparkline — RNF-58: extraído de sensor-monitor.tsx ──────────────────────

import { memo } from "react";
import { Line, LineChart, ResponsiveContainer } from "recharts";
import type { SensorDataPoint } from "@/hooks/use-sensor-data";

export interface SparklineProps {
  data: SensorDataPoint[];
  dataKey: keyof SensorDataPoint;
  color: string;
}

export const Sparkline = memo(function Sparkline({
  data,
  dataKey,
  color,
}: SparklineProps) {
  return (
    // aria-hidden + accessibilityLayer={false} (RNF-66 §12/§13): puramente
    // decorativo — o valor real já está em texto no SparkKpiCard. Ver
    // kpi-sparkline.tsx para o motivo (Recharts 3.x accessibilityLayer).
    <div aria-hidden="true" className="contents">
      <ResponsiveContainer width="100%" height={36}>
        <LineChart
          accessibilityLayer={false}
          data={data}
          margin={{ top: 2, right: 2, bottom: 2, left: 2 }}
        >
          <Line
            type="monotone"
            dataKey={dataKey as string}
            stroke={color}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
});
