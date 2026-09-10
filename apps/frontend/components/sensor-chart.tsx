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
 *
 * RNF-58: decomposto em `components/sensor-chart/*` — constants, ChartTooltip,
 * PressureChart, ThermalChart, ChartLegend, LiveIndicator. Nenhuma mudança de
 * comportamento/DOM (incl. `data-testid="chart-empty-state"`, preservado).
 */

import { memo } from "react";
import { AlertTriangle, TrendingUp } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { SensorDataPoint, RiskLevel } from "@/hooks/use-sensor-data";
import { ChartLegend } from "./sensor-chart/chart-legend";
import { LiveIndicator } from "./sensor-chart/live-indicator";
import { PressureChart } from "./sensor-chart/pressure-chart";
import { ThermalChart } from "./sensor-chart/thermal-chart";

// ── Props ─────────────────────────────────────────────────────────────────

interface SensorChartProps {
  history: SensorDataPoint[];
  isAnomaly: boolean;
  riskLevel: RiskLevel;
  isLive?: boolean;
  className?: string;
}

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
