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
  assetId: string;
  isLoading?: boolean;
}

// ── Normalização (0–100) ──────────────────────────────────────────────────

function normalize(value: number, min: number, max: number): number {
  return parseFloat(
    Math.min(100, Math.max(0, ((value - min) / (max - min)) * 100)).toFixed(1),
  );
}

// ── Valores de referência "ótimo" ─────────────────────────────────────────
const OPTIMAL = {
  TP2: normalize(10.1, 0, 12),
  TP3: normalize(10.1, 0, 12),
  Corrente: normalize(3.8, 0, 10),
  "Temp. Óleo": normalize(64.0, 50, 90),
};

// ── Estilos compartilhados ────────────────────────────────────────────────

const GRID_STROKE = "hsl(var(--border))";
const ANGLE_TICK = { fontSize: 11, fill: "hsl(var(--muted-foreground))" };

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

const ChartTooltip = memo(function ChartTooltip({ active, label, payload }: TooltipProps) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-border/80 bg-popover/90 px-3 py-2 shadow-md backdrop-blur-md">
      <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {label}
      </p>
      {payload.map((entry) => (
        <div key={entry.name} className="flex items-center gap-2 text-xs">
          <span className="inline-block h-2 w-2 rounded-full" style={{ background: entry.color }} />
          <span className="text-muted-foreground">{entry.name}</span>
          <span className="ml-auto font-bold tabular-nums text-foreground">
            {entry.value.toFixed(0)}
            <span className="font-normal text-muted-foreground">/100</span>
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
  assetId,
  isLoading,
}: AssetRadarChartProps) {
  const data = [
    { subject: "TP2",       ótimo: OPTIMAL.TP2,           atual: normalize(tp2,         0,  12) },
    { subject: "TP3",       ótimo: OPTIMAL.TP3,           atual: normalize(tp3,         0,  12) },
    { subject: "Corrente",  ótimo: OPTIMAL.Corrente,      atual: normalize(motorCurrent, 0, 10) },
    { subject: "Temp. Óleo",ótimo: OPTIMAL["Temp. Óleo"], atual: normalize(oilTemp,    50,  90) },
  ];

  return (
    <Card className="border-border bg-card">
      <CardHeader className="px-5 pb-2 pt-4">
        <CardTitle className="flex items-center gap-2 text-sm font-semibold text-foreground/90">
          <Activity className="h-4 w-4 text-primary" />
          Perfil Operacional
        </CardTitle>
        <p className="text-[11px] text-muted-foreground">
          {assetId} · Ótimo vs Atual (normalizado 0–100)
        </p>
      </CardHeader>

      <CardContent className="px-5 pb-4">
        {isLoading ? (
          <div className="flex items-center justify-center py-4">
            <Skeleton className="h-[200px] w-[200px] rounded-full" />
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <RadarChart
              data={data}
              outerRadius={70}
              margin={{ top: 8, right: 28, bottom: 4, left: 28 }}
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
                verticalAlign="bottom"
                wrapperStyle={{
                  fontSize: 11,
                  color: "hsl(var(--muted-foreground))",
                  paddingTop: 20,
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
