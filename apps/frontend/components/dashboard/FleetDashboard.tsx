"use client";

import { useMemo, useState } from "react";
import { useSensorData, getRiskLevel } from "@/hooks/use-sensor-data";
import { useAlertWebSocket } from "@/hooks/use-alert-websocket";
import FleetKPIs from "@/components/dashboard/FleetKPIs";
import AssetTable, { MOCK_ASSETS } from "@/components/dashboard/AssetTable";
import AssetRadarChart from "@/components/dashboard/AssetRadarChart";
import AssetEfficiencyChart from "@/components/dashboard/AssetEfficiencyChart";
import type { PredictPayload } from "@/lib/api-client";

// The only asset with a live SSE + prediction feed.
const LIVE_ASSET_ID = "APU-Trem-042";

// Sensible static defaults for fields that MockAsset does not carry.
// Used when a non-live asset is selected — the radar shows a static profile.
const STATIC_DEFAULTS: Omit<
  PredictPayload,
  "TP2" | "TP3" | "Motor_current" | "Oil_temperature"
> = {
  H1: 8.5,
  DV_pressure: 0.05,
  Reservoirs: 7.0,
  COMP: 1,
  DV_eletric: 0,
  Towers: 0,
  MPG: 1,
  Oil_level: 1,
};

export default function FleetDashboard() {
  const { latest, currentPayload, isLoading, sseStatus } = useSensorData();
  const { alerts } = useAlertWebSocket();

  const [selectedAssetId, setSelectedAssetId] = useState<string>(LIVE_ASSET_ID);

  const probability = latest?.failure_probability ?? 0;
  const alertProb = alerts.reduce((max, a) => Math.max(max, a.probability), 0);
  const effectiveProb = Math.max(probability, alertProb);
  const effectiveRiskLevel = getRiskLevel(effectiveProb);

  const selectedMock = MOCK_ASSETS.find((a) => a.id === selectedAssetId);
  const isLiveSelected = selectedAssetId === LIVE_ASSET_ID;

  // Resolve the sensor snapshot shown in the radar chart.
  // Live asset → real-time currentPayload (updates at 1 Hz via SSE).
  // Other assets → static mock payload derived from MOCK_ASSETS fields.
  const radarSensorData: PredictPayload = useMemo(
    () =>
      isLiveSelected
        ? currentPayload
        : {
            ...STATIC_DEFAULTS,
            TP2: selectedMock?.tp2 ?? 8.0,
            TP3: selectedMock?.tp3 ?? 8.0,
            Motor_current: selectedMock?.motorCurrent ?? 5.0,
            Oil_temperature: selectedMock?.oilTemp ?? 70.0,
          },
    [isLiveSelected, currentPayload, selectedMock],
  );

  return (
    <div className="flex flex-col gap-6 p-6">
      {/* ── Cabeçalho ── */}
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
            PredictIQ · MetroPT-3
          </p>
          <h1 className="mt-0.5 text-xl font-bold tracking-tight text-foreground">
            Gestão de Frota
          </h1>
          <p className="mt-0.5 text-sm text-muted-foreground">
            Visão macro de todos os compressores monitorados
          </p>
        </div>
        <div className="flex items-center gap-1.5 rounded-lg border border-border bg-card px-3 py-1.5">
          <span
            className={
              sseStatus === "connected"
                ? "h-1.5 w-1.5 animate-pulse rounded-full bg-green-400"
                : "h-1.5 w-1.5 rounded-full bg-amber-400"
            }
            aria-hidden="true"
          />
          <span className="text-[11px] font-medium text-muted-foreground">
            {sseStatus === "connected" ? "Telemetria ao vivo" : "Reconectando…"}
          </span>
        </div>
      </div>

      {/* ── KPI Cards ── */}
      <FleetKPIs
        effectiveRiskLevel={effectiveRiskLevel}
        sseStatus={sseStatus}
        isLoading={isLoading}
      />

      {/* ── Grid principal ── */}
      <div className="grid gap-4 lg:grid-cols-5">
        {/* Tabela de ativos — coluna principal */}
        <div className="lg:col-span-3">
          <AssetTable
            effectiveRiskLevel={effectiveRiskLevel}
            effectiveProb={effectiveProb}
            tp2={currentPayload.TP2}
            oilTemp={currentPayload.Oil_temperature}
            isLoading={isLoading}
            selectedId={selectedAssetId}
            onSelect={setSelectedAssetId}
          />
        </div>

        {/* Coluna lateral — radar e eficiência reativos ao ativo selecionado */}
        <div className="flex flex-col gap-4 lg:col-span-2">
          <AssetRadarChart
            sensorData={radarSensorData}
            isLive={isLiveSelected}
            assetId={selectedAssetId}
            anomalyScore={
              isLiveSelected
                ? latest?.failure_probability ?? 0
                : selectedMock?.prob ?? 0
            }
            isLoading={isLoading && isLiveSelected}
          />
          <AssetEfficiencyChart assetId={selectedAssetId} />
        </div>
      </div>
    </div>
  );
}
