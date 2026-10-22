"use client";

// ── Gráfico principal — AreaChart TP2 + TP3 — RNF-58: extraído de
// sensor-monitor.tsx ─────────────────────────────────────────────────────────

import { memo } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Activity } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { RiskLevel, SensorDataPoint } from "@/hooks/use-sensor-data";
import { formatFixed } from "@/lib/format";
import { cn } from "@/lib/utils";
import { AREA_TOOLTIP } from "./area-chart-tooltip";
import { AXIS_TICK, C, GRID_STROKE } from "./constants";

export interface MainAreaChartProps {
  data: SensorDataPoint[];
  isLive: boolean;
  riskLevel: RiskLevel;
}

export const MainAreaChart = memo(function MainAreaChart({
  data,
  isLive,
  riskLevel,
}: MainAreaChartProps) {
  const isEmpty = data.length < 2;
  const latest = data[data.length - 1];

  return (
    <Card
      className={cn(
        "border-border transition-colors duration-700",
        riskLevel === "ALERTA" && "border-amber-500/30",
        riskLevel === "CRÍTICO" && "border-red-500/30",
      )}
    >
      <CardHeader className="px-5 pb-2 pt-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-sm font-semibold text-foreground/90">
            <Activity className="h-4 w-4" style={{ color: C.tp2 }} />
            Pressão em Tempo Real — TP2 &amp; TP3
            {isLive && (
              <span className="flex items-center gap-1">
                <span className="relative flex h-2 w-2">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-400 opacity-75" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-green-500" />
                </span>
                <span className="text-[10px] font-semibold uppercase tracking-wider text-green-600 dark:text-green-400">
                  Ao vivo
                </span>
              </span>
            )}
          </CardTitle>
          <div className="flex items-center gap-3">
            {(["TP2", "TP3"] as const).map((k) => (
              <div key={k} className="flex items-center gap-1.5">
                <span
                  className="inline-block h-2 w-4 rounded-full"
                  style={{ background: k === "TP2" ? C.tp2 : C.tp3 }}
                />
                <span className="text-[10px] text-muted-foreground">
                  {k} (bar)
                </span>
              </div>
            ))}
          </div>
        </div>
      </CardHeader>
      <CardContent className="px-4 pb-4">
        {isEmpty ? (
          <div className="flex h-[180px] items-center justify-center text-sm text-muted-foreground">
            Coletando dados…
          </div>
        ) : (
          <>
            {/* Alternativa textual ao gráfico (RNF-66 §12) — sem aria-live:
                1Hz seria ruidoso demais para anúncio contínuo (RNF-66 §14). */}
            <p className="sr-only">
              Leitura mais recente de pressão: TP2 {formatFixed(latest.TP2, 1)}{" "}
              bar, TP3 {formatFixed(latest.TP3, 1)} bar.
            </p>
            <div aria-hidden="true" className="contents">
              <ResponsiveContainer width="100%" height={180}>
                <AreaChart
                  accessibilityLayer={false}
                  data={data}
                  margin={{ top: 4, right: 8, bottom: 0, left: -10 }}
                >
                  <defs>
                    <linearGradient id="gradTP2" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={C.tp2} stopOpacity={0.25} />
                      <stop offset="100%" stopColor={C.tp2} stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="gradTP3" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={C.tp3} stopOpacity={0.25} />
                      <stop offset="100%" stopColor={C.tp3} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke={GRID_STROKE}
                    vertical={false}
                  />
                  <XAxis
                    dataKey="time"
                    tick={AXIS_TICK}
                    tickLine={false}
                    axisLine={false}
                    interval="preserveStartEnd"
                  />
                  {/* Ticks fixos: o ruído negativo de TP2 (−0,014 bar)
                      gerava ticks −0,014 / 3,986 / 7,986, cortados. */}
                  <YAxis
                    domain={[0, 12]}
                    ticks={[0, 3, 6, 9, 12]}
                    allowDataOverflow
                    tick={AXIS_TICK}
                    tickLine={false}
                    axisLine={false}
                    width={28}
                  />
                  <RechartsTooltip content={AREA_TOOLTIP} />
                  <Area
                    type="monotone"
                    dataKey="TP2"
                    name="TP2"
                    stroke={C.tp2}
                    strokeWidth={2}
                    fill="url(#gradTP2)"
                    dot={false}
                    activeDot={{ r: 4, fill: C.tp2 }}
                    isAnimationActive={false}
                  />
                  <Area
                    type="monotone"
                    dataKey="TP3"
                    name="TP3"
                    stroke={C.tp3}
                    strokeWidth={2}
                    fill="url(#gradTP3)"
                    dot={false}
                    activeDot={{ r: 4, fill: C.tp3 }}
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
});
