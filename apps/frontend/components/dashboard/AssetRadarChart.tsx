"use client";

import { memo, useMemo } from "react";
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
import type { PredictPayload } from "@/lib/api-client";

// ── Physical scale ceilings ───────────────────────────────────────────────
// Each value is the absolute upper operational limit of the sensor.
// Normalization: (raw / max) × 100  →  percentage of capacity [0–100].
// This ensures TP2 (max 12 bar) and Oil_temperature (max 80 °C) share the
// same 0–100 axis without one dominating the polygon.
const MAX_SCALE = {
  TP2: 12, // bar
  TP3: 12, // bar
  H1: 12, // bar
  Motor_current: 10, // A
  Oil_temperature: 80, // °C
  Reservoirs: 12, // bar
} as const;

// ── Helpers ───────────────────────────────────────────────────────────────

/** Map a raw sensor reading to a 0–100 percentage of its physical ceiling. */
function pct(value: number, max: number): number {
  return parseFloat(Math.min(100, Math.max(0, (value / max) * 100)).toFixed(1));
}

// ── Optimal / healthy reference ───────────────────────────────────────────
// Approximate median values from healthy (label=0) windows in the
// MetroPT-3 training dataset — pressurised, steady-state operation.
// Used as the green "Ótimo" area; divergence from this shape signals trouble.
const OPTIMAL_PCT = {
  TP2: pct(10.1, MAX_SCALE.TP2), // 84.2 %
  TP3: pct(10.1, MAX_SCALE.TP3), // 84.2 %
  H1: pct(8.5, MAX_SCALE.H1), // 70.8 %
  Motor_current: pct(3.8, MAX_SCALE.Motor_current), // 38.0 %
  Oil_temperature: pct(64, MAX_SCALE.Oil_temperature), // 80.0 %
  Reservoirs: pct(7.0, MAX_SCALE.Reservoirs), // 58.3 %
} as const;

// ── Anomaly score helpers ─────────────────────────────────────────────────

type AnomalyLevel = "NORMAL" | "ALERTA" | "CRÍTICO";

function toAnomalyLevel(score: number): AnomalyLevel {
  if (score < 0.3) return "NORMAL";
  if (score < 0.65) return "ALERTA";
  return "CRÍTICO";
}

const ANOMALY_STYLE: Record<AnomalyLevel, string> = {
  NORMAL: "border-green-500/30 bg-green-500/10 text-green-400",
  ALERTA: "border-amber-500/30 bg-amber-500/10 text-amber-400",
  CRÍTICO: "border-red-500/30   bg-red-500/10   text-red-400",
};

// ── Chart style constants ─────────────────────────────────────────────────

const GRID_STROKE = "hsl(var(--border))";
const ANGLE_TICK = { fontSize: 11, fill: "hsl(var(--muted-foreground))" };

// ── Tooltip ───────────────────────────────────────────────────────────────

interface TooltipEntry {
  name: string;
  value: number;
  color: string;
}

interface RadarTooltipProps {
  active?: boolean;
  label?: string;
  payload?: TooltipEntry[];
}

const ChartTooltip = memo(function ChartTooltip({
  active,
  label,
  payload,
}: RadarTooltipProps) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-border/80 bg-popover/90 px-3 py-2 shadow-md backdrop-blur-md">
      <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {label}
      </p>
      {payload.map((entry) => (
        <div key={entry.name} className="flex items-center gap-2 text-xs">
          <span
            className="inline-block h-2 w-2 rounded-full"
            style={{ background: entry.color }}
          />
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

// Static JSX element — Recharts clones it with runtime tooltip props.
const TOOLTIP_CONTENT = <ChartTooltip />;

// ── Sub-component: raw instantaneous sensor value chip ────────────────────

interface RawValueProps {
  label: string;
  value: number;
  decimals?: number;
  unit: string;
}

function RawValue({ label, value, decimals = 2, unit }: RawValueProps) {
  return (
    <span className="text-[11px] text-muted-foreground">
      {label}{" "}
      <span className="font-mono font-semibold text-foreground">
        {value.toFixed(decimals)} <span className="font-normal">{unit}</span>
      </span>
    </span>
  );
}

// ── Radar datum type ──────────────────────────────────────────────────────

interface RadarDatum {
  subject: string;
  ótimo: number;
  atual: number;
}

// ── Props ─────────────────────────────────────────────────────────────────

interface AssetRadarChartProps {
  /** Pre-resolved sensor snapshot: live currentPayload or static mock data. */
  sensorData: PredictPayload;
  /** True when APU-Trem-042 is selected — shows LIVE badge + raw readouts. */
  isLive: boolean;
  assetId: string;
  /** failure_probability from the latest model prediction (0–1). */
  anomalyScore?: number;
  isLoading?: boolean;
}

// ── Component ─────────────────────────────────────────────────────────────

const AssetRadarChart = memo(function AssetRadarChart({
  sensorData,
  isLive,
  assetId,
  anomalyScore = 0,
  isLoading,
}: AssetRadarChartProps) {
  // Recalculate radar polygon whenever sensorData changes (1 Hz in live mode).
  const data: RadarDatum[] = useMemo(
    () => [
      {
        subject: "TP2",
        ótimo: OPTIMAL_PCT.TP2,
        atual: pct(sensorData.TP2, MAX_SCALE.TP2),
      },
      {
        subject: "TP3",
        ótimo: OPTIMAL_PCT.TP3,
        atual: pct(sensorData.TP3, MAX_SCALE.TP3),
      },
      {
        subject: "H1",
        ótimo: OPTIMAL_PCT.H1,
        atual: pct(sensorData.H1, MAX_SCALE.H1),
      },
      {
        subject: "Corrente",
        ótimo: OPTIMAL_PCT.Motor_current,
        atual: pct(sensorData.Motor_current, MAX_SCALE.Motor_current),
      },
      {
        subject: "Temp.",
        ótimo: OPTIMAL_PCT.Oil_temperature,
        atual: pct(sensorData.Oil_temperature, MAX_SCALE.Oil_temperature),
      },
      {
        subject: "Reserv.",
        ótimo: OPTIMAL_PCT.Reservoirs,
        atual: pct(sensorData.Reservoirs, MAX_SCALE.Reservoirs),
      },
    ],
    [sensorData],
  );

  const level = toAnomalyLevel(anomalyScore);

  return (
    <Card className="border-border bg-card">
      <CardHeader className="px-5 pb-2 pt-4">
        {/* ── Title row with status badges ─────────────────────────── */}
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="flex items-center gap-2 text-sm font-semibold text-foreground/90">
            <Activity className="h-4 w-4 text-primary" />
            Perfil Operacional
          </CardTitle>

          <div className="flex shrink-0 items-center gap-1.5">
            {isLive ? (
              <span className="flex items-center gap-1.5 rounded-full border border-green-500/30 bg-green-500/10 px-2 py-0.5 text-[10px] font-semibold text-green-400">
                <span
                  className="h-1.5 w-1.5 animate-pulse rounded-full bg-green-400"
                  aria-hidden="true"
                />
                LIVE
              </span>
            ) : (
              <span className="rounded-full border border-border bg-muted/30 px-2 py-0.5 text-[10px] font-medium text-muted-foreground">
                IDLE
              </span>
            )}

            {/* Anomaly score badge — only shown in live mode */}
            {isLive && (
              <span
                className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${ANOMALY_STYLE[level]}`}
                aria-label={`Score de anomalia: ${level} ${(
                  anomalyScore * 100
                ).toFixed(1)}%`}
              >
                {level} {(anomalyScore * 100).toFixed(1)}%
              </span>
            )}
          </div>
        </div>

        <p className="text-[11px] text-muted-foreground">
          {assetId} · {isLive ? "Ótimo vs Atual" : "Ótimo vs Estático"}{" "}
          (normalizado 0–100)
        </p>

        {/* ── Raw instantaneous values strip — live mode only ──────── */}
        {isLive && !isLoading && (
          <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-border/50 pt-1.5">
            <RawValue label="TP2" value={sensorData.TP2} unit="bar" />
            <span className="text-border/60" aria-hidden="true">
              ·
            </span>
            <RawValue
              label="Temp"
              value={sensorData.Oil_temperature}
              decimals={1}
              unit="°C"
            />
            <span className="text-border/60" aria-hidden="true">
              ·
            </span>
            <RawValue label="Motor" value={sensorData.Motor_current} unit="A" />
          </div>
        )}
      </CardHeader>

      <CardContent className="px-5 pb-4">
        {isLoading ? (
          <div className="flex items-center justify-center py-4">
            <Skeleton className="h-[200px] w-[200px] rounded-full" />
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={200}>
            <RadarChart
              data={data}
              outerRadius={62}
              margin={{ top: 8, right: 28, bottom: 0, left: 28 }}
            >
              <PolarGrid stroke={GRID_STROKE} radialLines />
              <PolarAngleAxis dataKey="subject" tick={ANGLE_TICK} />
              <Tooltip content={TOOLTIP_CONTENT} />
              <Radar
                name="Ótimo"
                dataKey="ótimo"
                stroke="#4ade80"
                fill="#4ade80"
                fillOpacity={0.08}
                strokeWidth={1.5}
                strokeDasharray="4 2"
              />
              <Radar
                name="Atual"
                dataKey="atual"
                stroke="#60a5fa"
                fill="#60a5fa"
                fillOpacity={0.22}
                strokeWidth={2}
              />
              <Legend
                iconSize={8}
                iconType="circle"
                verticalAlign="bottom"
                wrapperStyle={{
                  fontSize: 11,
                  color: "hsl(var(--muted-foreground))",
                  paddingTop: 12,
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
