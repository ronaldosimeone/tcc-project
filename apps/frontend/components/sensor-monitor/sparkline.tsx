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
    <ResponsiveContainer width="100%" height={36}>
      <LineChart data={data} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
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
  );
});
