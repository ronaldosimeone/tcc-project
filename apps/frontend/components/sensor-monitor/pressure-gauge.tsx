"use client";

// ── Manômetros de Pressão (gauge SVG semi-circular) — RNF-58: extraído de
// sensor-monitor.tsx ─────────────────────────────────────────────────────────

import { memo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { C } from "./constants";
import { buildArcPaths } from "./helpers";

interface PressureGaugeProps {
  label: string;
  value: number;
  max: number;
  unit: string;
  color: string;
}

function PressureGauge({ label, value, max, unit, color }: PressureGaugeProps) {
  const pct = Math.min(1, Math.max(0, value / max));
  const cx = 50;
  const cy = 46;
  const r = 34;
  const sw = 7;
  const { bg, fg } = buildArcPaths(cx, cy, r, pct);

  return (
    <div className="flex flex-col items-center gap-1">
      {/* viewBox 100×72 dá folga absoluta para descida das letras
          ("bar" tem descender visual mesmo sem 'g'/'p'/'q'); somado a
          overflow-visible no <svg>, garante que nada é cortado mesmo
          quando o navegador aplica antialiasing agressivo. */}
      <svg
        viewBox="0 0 100 72"
        className="w-full max-w-[96px] overflow-visible"
      >
        <path
          d={bg}
          fill="none"
          stroke="rgb(226 232 240)"
          strokeWidth={sw}
          strokeLinecap="round"
        />
        {fg && (
          <path
            d={fg}
            fill="none"
            stroke={color}
            strokeWidth={sw}
            strokeLinecap="round"
          />
        )}
        <text
          x={cx}
          y={cy - 4}
          textAnchor="middle"
          fill="rgb(15 23 42)"
          fontSize="11"
          fontWeight="700"
          fontFamily="var(--font-geist-sans, sans-serif)"
        >
          {value.toFixed(1)}
        </text>
        <text
          x={cx}
          y={cy + 14}
          textAnchor="middle"
          dominantBaseline="middle"
          fill="rgb(100 116 139)"
          fontSize="9"
          fontFamily="var(--font-geist-sans, sans-serif)"
        >
          {unit}
        </text>
      </svg>
      <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
      <div className="w-full max-w-[80px]">
        <Progress
          value={pct * 100}
          className="h-1"
          style={
            {
              "--progress-bg": color,
            } as React.CSSProperties
          }
        />
      </div>
    </div>
  );
}

export interface PressureRadialsProps {
  H1: number;
  DV_pressure: number;
  Reservoirs: number;
  isLoading: boolean;
}

export const PressureRadials = memo(function PressureRadials({
  H1,
  DV_pressure,
  Reservoirs,
  isLoading,
}: PressureRadialsProps) {
  return (
    <Card className="border-slate-200">
      <CardHeader className="px-4 pb-2 pt-4">
        <CardTitle className="text-sm font-semibold text-foreground/90">
          Pressões Secundárias
        </CardTitle>
        <p className="text-[10px] text-muted-foreground">
          H1 · DV Pressure · Reservatório
        </p>
      </CardHeader>
      <CardContent className="px-4 pb-6">
        {isLoading ? (
          <div className="flex items-center justify-around gap-2">
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-[100px] w-[80px] rounded-lg" />
            ))}
          </div>
        ) : (
          <div className="flex items-center justify-around gap-2">
            <PressureGauge
              label="H1"
              value={H1}
              max={11}
              unit="bar"
              color={C.h1}
            />
            <PressureGauge
              label="DV Press"
              value={DV_pressure}
              max={4}
              unit="bar"
              color={C.dvp}
            />
            <PressureGauge
              label="Reserv."
              value={Reservoirs}
              max={12}
              unit="bar"
              color={C.res}
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
});
