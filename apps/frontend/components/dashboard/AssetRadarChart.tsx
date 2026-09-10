"use client";

/**
 * RNF-58: decomposto em `components/dashboard/asset-radar-chart/*` —
 * helpers (escalas/anomalia), ChartTooltip, RawValue, RadarHeader,
 * RadarPolygonChart. Este arquivo mantém só o cálculo do polígono
 * (`data`) e a orquestração do Card. Nenhuma mudança de comportamento/DOM.
 */

import { memo, useMemo } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import type { PredictPayload } from "@/lib/api-client";
import { MAX_SCALE, OPTIMAL_PCT, pct } from "./asset-radar-chart/helpers";
import { RadarHeader } from "./asset-radar-chart/radar-header";
import {
  RadarPolygonChart,
  type RadarDatum,
} from "./asset-radar-chart/radar-polygon-chart";

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

  return (
    <Card className="border-border bg-card">
      <RadarHeader
        sensorData={sensorData}
        isLive={isLive}
        assetId={assetId}
        anomalyScore={anomalyScore}
        isLoading={isLoading}
      />

      <CardContent className="px-5 pb-4">
        {isLoading ? (
          <div className="flex items-center justify-center py-4">
            <Skeleton className="h-[200px] w-[200px] rounded-full" />
          </div>
        ) : (
          <RadarPolygonChart data={data} />
        )}
      </CardContent>
    </Card>
  );
});

export default AssetRadarChart;
