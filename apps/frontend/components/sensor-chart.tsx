"use client";

/**
 * SensorChart — RF-06 / RF-07.
 *
 * RF-06: Plota 4 séries de sensores com dados em tempo real via SSE (1Hz).
 *   • Chart A — Pressão:   TP2 (bar) + TP3 (bar)
 *   • Chart B — Elétrico:  Corrente (A, eixo esquerdo) + Temperatura (°C, eixo direito)
 *
 * RF-07: Destaque visual de anomalias.
 *   • Borda e fundo do card mudam suavemente para âmbar (ALERTA) ou vermelho (CRÍTICO).
 *   • Ícone de alerta pulsante aparece no header.
 *   • Largura das linhas aumenta levemente no estado de anomalia.
 *
 * RNF-33: React.memo em todos os sub-componentes + isAnimationActive={false}
 * eliminam re-renders desnecessários durante atualizações a 1 Hz.
 */

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
import { AlertTriangle, TrendingUp } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { SensorDataPoint, RiskLevel } from "@/hooks/use-sensor-data";

// ── Props ─────────────────────────────────────────────────────────────────

interface SensorChartProps {
  history: SensorDataPoint[];
  isAnomaly: boolean;
  riskLevel: RiskLevel;
  isLive?: boolean;
  className?: string;
}

// ── Paleta de cores das séries ────────────────────────────────────────────

const LINE_COLORS = {
  TP2: "#60a5fa", // blue-400
  TP3: "#4ade80", // green-400
  Motor_current: "#c084fc", // purple-400
  Oil_temperature: "#fb923c", // orange-400
} as const;

// ── Tooltip customizado (memoizado para estabilidade de referência) ────────

interface TooltipPayloadEntry {
  name: string;
  value: number;
  color: string;
}

interface CustomTooltipProps {
  active?: boolean;
  label?: string;
  payload?: TooltipPayloadEntry[];
}

const ChartTooltip = memo(function ChartTooltip({
  active,
  label,
  payload,
}: CustomTooltipProps) {
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
            {entry.value}
          </span>
        </div>
      ))}
    </div>
  );
});

// ── Estilos compartilhados dos eixos (constantes de módulo — refs estáveis) ─

const AXIS_TICK_STYLE = {
  fontSize: 10,
  fill: "hsl(215 15% 45%)",
} as const;

const GRID_STROKE = "hsl(217 18% 16%)";

// Instâncias únicas do tooltip para evitar recriação a cada render
const TOOLTIP_CONTENT = <ChartTooltip />;

// ── Indicador "Live" ──────────────────────────────────────────────────────

function LiveIndicator() {
  return (
    <span className="flex items-center gap-1.5">
      <span className="relative flex h-2 w-2">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-400 opacity-75" />
        <span className="relative inline-flex h-2 w-2 rounded-full bg-green-500" />
      </span>
      <span className="text-[10px] font-semibold uppercase tracking-wider text-green-400">
        Live
      </span>
    </span>
  );
}

// ── Sub-chart de Pressão (TP2 + TP3) ─────────────────────────────────────

interface PressureChartProps {
  data: SensorDataPoint[];
  strokeWidth: number;
}

const PressureChart = memo(function PressureChart({
  data,
  strokeWidth,
}: PressureChartProps) {
  return (
    <div>
      <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
        Pressão &mdash; TP2 &amp; TP3 (bar)
      </p>
      <ResponsiveContainer width="100%" height={160}>
        <LineChart
          data={data}
          margin={{ top: 4, right: 8, bottom: 0, left: -10 }}
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
            domain={[0, 12]}
            tick={AXIS_TICK_STYLE}
            tickLine={false}
            axisLine={false}
            width={28}
          />
          <Tooltip content={TOOLTIP_CONTENT} />
          <Line
            type="monotone"
            dataKey="TP2"
            name="TP2"
            stroke={LINE_COLORS.TP2}
            strokeWidth={strokeWidth}
            dot={false}
            activeDot={{ r: 4, fill: LINE_COLORS.TP2 }}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="TP3"
            name="TP3"
            stroke={LINE_COLORS.TP3}
            strokeWidth={strokeWidth}
            dot={false}
            activeDot={{ r: 4, fill: LINE_COLORS.TP3 }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
});

// ── Sub-chart Elétrico / Térmico (Motor_current + Oil_temperature) ────────

const ThermalChart = memo(function ThermalChart({
  data,
  strokeWidth,
}: PressureChartProps) {
  return (
    <div>
      <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
        Corrente (A) &amp; Temperatura (°C)
      </p>
      <ResponsiveContainer width="100%" height={160}>
        <LineChart
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
  );
});

// ── Legenda das séries ────────────────────────────────────────────────────

const LEGEND_ITEMS: Array<{ key: keyof typeof LINE_COLORS; label: string }> = [
  { key: "TP2", label: "TP2 (bar)" },
  { key: "TP3", label: "TP3 (bar)" },
  { key: "Motor_current", label: "Corrente (A)" },
  { key: "Oil_temperature", label: "Temperatura (°C)" },
];

const ChartLegend = memo(function ChartLegend() {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1">
      {LEGEND_ITEMS.map(({ key, label }) => (
        <div key={key} className="flex items-center gap-1.5">
          <span
            className="inline-block h-2 w-4 rounded-full"
            style={{ background: LINE_COLORS[key] }}
          />
          <span className="text-[10px] text-muted-foreground">{label}</span>
        </div>
      ))}
    </div>
  );
});

// ── Componente principal (RF-06 + RF-07 + RNF-33) ────────────────────────

export const SensorChart = memo(function SensorChart({
  history,
  isAnomaly,
  riskLevel,
  isLive = false,
  className,
}: SensorChartProps) {
  const cardStyle = cn(
    "border transition-all duration-700",
    riskLevel === "NORMAL" && "border-border bg-card",
    riskLevel === "ALERTA" &&
      "border-amber-500/40 bg-amber-500/5 shadow-lg shadow-amber-500/5",
    riskLevel === "CRÍTICO" &&
      "border-red-500/50 bg-red-500/5 shadow-xl shadow-red-500/10",
    className,
  );

  const strokeWidth = isAnomaly ? 2.5 : 1.8;
  const isEmpty = history.length < 2;

  return (
    <Card className={cardStyle} data-anomaly={isAnomaly} data-risk={riskLevel}>
      <CardHeader className="pb-3 pt-4 px-5">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-sm font-semibold text-foreground/90">
            <TrendingUp className="h-4 w-4 text-primary" />
            Telemetria de Sensores
            {isLive && <LiveIndicator />}
          </CardTitle>

          {/* RF-07 — ícone de alerta pulsante */}
          {isAnomaly && (
            <div
              className={cn(
                "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold",
                riskLevel === "ALERTA"
                  ? "bg-amber-500/15 text-amber-400"
                  : "bg-red-500/15 text-red-400",
              )}
              role="alert"
              aria-label={`Anomalia detectada: ${riskLevel}`}
            >
              <AlertTriangle className="h-3.5 w-3.5 animate-pulse" />
              {riskLevel}
            </div>
          )}
        </div>

        <ChartLegend />
      </CardHeader>

      <CardContent className="flex flex-col gap-6 px-5 pb-5">
        {isEmpty ? (
          <div
            className="flex h-[160px] items-center justify-center text-sm text-muted-foreground"
            role="status"
            data-testid="chart-empty-state"
          >
            Coletando dados…
          </div>
        ) : (
          <>
            <PressureChart data={history} strokeWidth={strokeWidth} />
            <div className="h-px bg-border" />
            <ThermalChart data={history} strokeWidth={strokeWidth} />
          </>
        )}
      </CardContent>
    </Card>
  );
});
