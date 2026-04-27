"use client";

import { memo } from "react";
import {
  Legend,
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { Activity } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

// ── Props ─────────────────────────────────────────────────────────────────

interface AssetRadarChartProps {
  tp2: number;
  tp3: number;
  motorCurrent: number;
  oilTemp: number;
  isLoading?: boolean;
}

// ── Normalização (0–100) ──────────────────────────────────────────────────

function normalize(value: number, min: number, max: number): number {
  return parseFloat(
    Math.min(100, Math.max(0, ((value - min) / (max - min)) * 100)).toFixed(1),
  );
}

// ── Valores de referência "ótimo" ─────────────────────────────────────────
// Baseline calibrada pelo manual MetroPT-3 em operação normal.
const OPTIMAL = {
  TP2: normalize(8.0, 0, 12), // 66.7
  TP3: normalize(8.0, 0, 12), // 66.7
  Corrente: normalize(4.5, 0, 10), // 45.0
  "Temp. Óleo": normalize(68, 50, 90), // 45.0
};

// ── Estilos compartilhados ────────────────────────────────────────────────

const GRID_STROKE = "hsl(217 18% 16%)";
const ANGLE_TICK = { fontSize: 11, fill: "hsl(215 15% 45%)" };

// ── Tooltip ───────────────────────────────────────────────────────────────

interface TooltipEntry {
  name: string;
  value: number;
  color: string;
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
  return (
    <div className="rounded-lg border border-zinc-200/60 bg-white/60 px-3 py-2 shadow-md backdrop-blur-md">
      <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
        {label}
      </p>
      {payload.map((entry) => (
        <div key={entry.name} className="flex items-center gap-2 text-xs">
          <span
            className="inline-block h-2 w-2 rounded-full"
            style={{ background: entry.color }}
          />
          <span className="text-zinc-600">{entry.name}</span>
          <span className="ml-auto font-bold tabular-nums text-zinc-900">
            {entry.value.toFixed(0)}
            <span className="font-normal text-zinc-500">/100</span>
          </span>
        </div>
      ))}
    </div>
  );
});

const TOOLTIP_CONTENT = <ChartTooltip />;

// ── Component ─────────────────────────────────────────────────────────────

const AssetRadarChart = memo(function AssetRadarChart({
  tp2,
  tp3,
  motorCurrent,
  oilTemp,
  isLoading,
}: AssetRadarChartProps) {
  const data = [
    {
      subject: "TP2",
      ótimo: OPTIMAL.TP2,
      atual: normalize(tp2, 0, 12),
    },
    {
      subject: "TP3",
      ótimo: OPTIMAL.TP3,
      atual: normalize(tp3, 0, 12),
    },
    {
      subject: "Corrente",
      ótimo: OPTIMAL.Corrente,
      atual: normalize(motorCurrent, 0, 10),
    },
    {
      subject: "Temp. Óleo",
      ótimo: OPTIMAL["Temp. Óleo"],
      atual: normalize(oilTemp, 50, 90),
    },
  ];

  return (
    <Card className="border-border bg-card">
      <CardHeader className="px-5 pb-2 pt-4">
        <CardTitle className="flex items-center gap-2 text-sm font-semibold text-foreground/90">
          <Activity className="h-4 w-4 text-primary" />
          Perfil Operacional
        </CardTitle>
        <p className="text-[11px] text-muted-foreground">
          APU-Trem-042 · Ótimo vs Atual (normalizado 0–100)
        </p>
      </CardHeader>

      <CardContent className="px-5 pb-5">
        {isLoading ? (
          <div className="flex items-center justify-center">
            <Skeleton className="h-[210px] w-[210px] rounded-full" />
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={210}>
            <RadarChart
              data={data}
              margin={{ top: 8, right: 20, bottom: 8, left: 20 }}
            >
              <PolarGrid stroke={GRID_STROKE} radialLines={true} />
              <PolarAngleAxis dataKey="subject" tick={ANGLE_TICK} />
              <Tooltip content={TOOLTIP_CONTENT} />
              <Radar
                name="Ótimo"
                dataKey="ótimo"
                stroke="#4ade80"
                fill="#4ade80"
                fillOpacity={0.1}
                strokeWidth={1.5}
                strokeDasharray="4 2"
              />
              <Radar
                name="Atual"
                dataKey="atual"
                stroke="#60a5fa"
                fill="#60a5fa"
                fillOpacity={0.25}
                strokeWidth={2}
              />
              <Legend
                iconSize={8}
                iconType="circle"
                wrapperStyle={{
                  fontSize: 11,
                  color: "hsl(215 15% 50%)",
                  paddingTop: 8,
                }}
              />
            </RadarChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
});

export default AssetRadarChart;
