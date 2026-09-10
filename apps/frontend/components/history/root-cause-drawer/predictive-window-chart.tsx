// ── Janela preditiva (2h antes da falha) — RNF-58: extraído de
// RootCauseDrawer ─────────────────────────────────────────────────────────

import { useMemo } from "react";
import {
  Area,
  AreaChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { CRITICAL_THRESHOLD, PREDICTIVE_WINDOW } from "./constants";
import { PredictiveTooltip } from "./predictive-tooltip";

export function PredictiveWindowChart() {
  // Memoizado para evitar re-mount do Recharts a cada hover.
  const chartData = useMemo(() => [...PREDICTIVE_WINDOW], []);

  return (
    <section>
      <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-slate-500">
        Janela preditiva · 2 horas antes da falha
      </p>
      <div className="rounded-lg border border-slate-200 bg-white p-3">
        <ResponsiveContainer width="100%" height={160}>
          <AreaChart
            data={chartData}
            margin={{ top: 8, right: 8, bottom: 0, left: -16 }}
          >
            <defs>
              <linearGradient id="drawer-prob-grad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#f43f5e" stopOpacity={0.35} />
                <stop offset="100%" stopColor="#f43f5e" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="t"
              stroke="#94a3b8"
              fontSize={10}
              tickLine={false}
              axisLine={false}
              interval={1}
            />
            <YAxis
              stroke="#94a3b8"
              fontSize={10}
              tickLine={false}
              axisLine={false}
              domain={[0, 1]}
              tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
              width={32}
            />
            <Tooltip content={<PredictiveTooltip />} />
            <ReferenceLine
              y={CRITICAL_THRESHOLD}
              stroke="#ef4444"
              strokeDasharray="4 3"
              strokeWidth={1.5}
              label={{
                value: "CRÍTICO 65%",
                position: "insideTopRight",
                fontSize: 9,
                fill: "#ef4444",
              }}
            />
            <Area
              type="monotone"
              dataKey="p"
              stroke="#f43f5e"
              strokeWidth={2}
              fill="url(#drawer-prob-grad)"
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
