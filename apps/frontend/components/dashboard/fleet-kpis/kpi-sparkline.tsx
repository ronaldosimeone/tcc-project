// ── Sparkline genérica — RNF-58: extraído de FleetKPIs ───────────────────────

import { useMemo } from "react";
import { Area, AreaChart, ResponsiveContainer } from "recharts";
import { SPARK_COLOR, type SparkTone } from "./constants";

interface KpiSparklineProps {
  data: ReadonlyArray<number>;
  tone: SparkTone;
}

/** Mini AreaChart — 44px, sem eixos, sem grid. Decorativo. */
export function KpiSparkline({ data, tone }: KpiSparklineProps) {
  const color = SPARK_COLOR[tone];
  const gradId = `kpi-spark-${tone}`;
  const chartData = useMemo(() => data.map((v, i) => ({ i, v })), [data]);

  return (
    <ResponsiveContainer width="100%" height={44}>
      <AreaChart
        data={chartData}
        margin={{ top: 4, right: 0, left: 0, bottom: 0 }}
      >
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.32} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <Area
          type="monotone"
          dataKey="v"
          stroke={color}
          strokeWidth={1.75}
          fill={`url(#${gradId})`}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
