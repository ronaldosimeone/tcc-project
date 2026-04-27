"use client";

import { useSensorData, getRiskLevel } from "@/hooks/use-sensor-data";
import { useAlertWebSocket } from "@/hooks/use-alert-websocket";
import FleetKPIs from "@/components/dashboard/FleetKPIs";
import AssetTable from "@/components/dashboard/AssetTable";
import AssetRadarChart from "@/components/dashboard/AssetRadarChart";
import AssetEfficiencyChart from "@/components/dashboard/AssetEfficiencyChart";

export default function FleetDashboard() {
  const { latest, currentPayload, isLoading, sseStatus } = useSensorData();
  const { alerts } = useAlertWebSocket();

  const probability = latest?.failure_probability ?? 0;
  const alertProb = alerts.reduce((max, a) => Math.max(max, a.probability), 0);
  const effectiveProb = Math.max(probability, alertProb);
  const effectiveRiskLevel = getRiskLevel(effectiveProb);

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
          />
        </div>

        {/* Coluna lateral de gráficos */}
        <div className="flex flex-col gap-4 lg:col-span-2">
          <AssetRadarChart
            tp2={currentPayload.TP2}
            tp3={currentPayload.TP3}
            motorCurrent={currentPayload.Motor_current}
            oilTemp={currentPayload.Oil_temperature}
            isLoading={isLoading}
          />
          <AssetEfficiencyChart />
        </div>
      </div>
    </div>
  );
}
