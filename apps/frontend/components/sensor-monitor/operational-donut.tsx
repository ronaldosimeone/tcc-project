"use client";

// ── Donut — Estado Operacional — RNF-58: extraído de sensor-monitor.tsx ─────

import { memo, useMemo } from "react";
import {
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import type { SensorDataPoint } from "@/hooks/use-sensor-data";
import { OP_STATES } from "./constants";
import { computeOpState, makeDonutFormatter } from "./helpers";

export interface OperationalDonutProps {
  history: SensorDataPoint[];
  isLoading: boolean;
}

export const OperationalDonut = memo(function OperationalDonut({
  history,
  isLoading,
}: OperationalDonutProps) {
  const data = useMemo(() => computeOpState(history), [history]);
  const total = useMemo(() => data.reduce((s, d) => s + d.value, 0), [data]);
  const donutFormatter = useMemo(() => makeDonutFormatter(total), [total]);

  return (
    <Card className="border-slate-200">
      <CardHeader className="px-4 pb-2 pt-4">
        <CardTitle className="text-sm font-semibold text-foreground/90">
          Estado Operacional
        </CardTitle>
        <p className="text-[10px] text-muted-foreground">
          Derivado da corrente do motor
        </p>
      </CardHeader>
      <CardContent className="flex flex-col items-center gap-3 px-4 pb-4">
        {isLoading || data.length === 0 ? (
          <div className="flex h-[140px] items-center justify-center">
            {isLoading ? (
              <Skeleton className="h-[120px] w-[120px] rounded-full" />
            ) : (
              <p className="text-xs text-muted-foreground">Sem dados</p>
            )}
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={140}>
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={58}
                paddingAngle={3}
                dataKey="value"
                isAnimationActive={false}
              >
                {data.map((entry, i) => (
                  <Cell key={`cell-${i}`} fill={entry.color} />
                ))}
              </Pie>
              <RechartsTooltip
                formatter={donutFormatter}
                contentStyle={{
                  backgroundColor: "#ffffff",
                  color: "#0f172a",
                  border: "1px solid #e2e8f0",
                  borderRadius: "8px",
                  fontSize: "12px",
                  fontWeight: "500",
                }}
                labelStyle={{ color: "#0f172a", fontWeight: "600" }}
                itemStyle={{ color: "#334155" }}
              />
            </PieChart>
          </ResponsiveContainer>
        )}
        <div className="grid w-full grid-cols-2 gap-x-3 gap-y-1">
          {OP_STATES.map((s) => (
            <div key={s.name} className="flex items-center gap-1.5">
              <span
                className="h-2 w-2 shrink-0 rounded-full"
                style={{ background: s.color }}
              />
              <span className="truncate text-[10px] text-muted-foreground">
                {s.name}
              </span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
});
