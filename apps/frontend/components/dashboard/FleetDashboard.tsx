"use client";

/**
 * Dashboard de frota — "cockpit operacional" do PredictIQ.
 *
 * Layout vertical full-height (flex col + h-full vindo do layout.tsx):
 *   1. Cabeçalho        — altura natural
 *   2. KPIs             — altura natural
 *   3. Grid intermediário (flex-1) — Tabela (col-span-2) + Modelo (col-span-1)
 *      lado a lado; ambos com h-full para emparelhar a altura.
 *   4. EventFeedCard    — full-width abaixo; lista interna em grid responsivo.
 *
 * O `flex-1` no item 3 faz a tabela/modelo absorverem todo o espaço vertical
 * que sobra do viewport, eliminando o "espaço em branco" no rodapé.
 */

import { useMemo, useState } from "react";

import EventFeedCard from "@/components/dashboard/EventFeedCard";
import FleetHealthTable from "@/components/dashboard/FleetHealthTable";
import FleetKPIs from "@/components/dashboard/FleetKPIs";
import ModelStatusCard from "@/components/dashboard/ModelStatusCard";
import { DevProfiler } from "@/lib/dev-profiler";
import {
  getRiskLevel,
  useSensorData,
  type RiskLevel,
} from "@/hooks/use-sensor-data";

const LIVE_ASSET_ID = "APU-Trem-042";

export default function FleetDashboard() {
  // currentLatency vem da stream SSE contínua (1 Hz) — atualiza-se em modo
  // NORMAL também, ao contrário do canal WS que só dispara em alertas.
  const { latest, currentLatency, isLoading, sseStatus } = useSensorData();

  const [selectedAssetId, setSelectedAssetId] = useState<string>(LIVE_ASSET_ID);

  const effectiveProb = latest?.failure_probability ?? 0;
  const effectiveRiskLevel = getRiskLevel(effectiveProb);

  // Adapta o shape de `currentLatency` ({ key, latencyMs }) para a prop do
  // FleetKPIs ({ messageId, latencyMs }) — o nome do campo identificador é
  // diferente mas a semântica (chave única por frame) é idêntica.
  const latencyTelemetry = useMemo(() => {
    if (!currentLatency) return null;
    return {
      messageId: currentLatency.key,
      latencyMs: currentLatency.latencyMs,
    };
  }, [currentLatency]);

  // Distribuição da frota para o donut do ModelStatusCard.
  // Deriva os contadores directamente do RiskLevel (oficial — lib/risk-thresholds)
  // em vez de aproximar via faixas de health: assim o donut nunca contradiz
  // o badge mostrado na tabela para o mesmo ativo.
  //
  // 4 mocks fixos: APU-015 NORMAL, APU-023 ALERTA, APU-031 NORMAL, APU-055 NORMAL
  // → 3 NORMAL + 1 ALERTA + 0 CRÍTICO de base; o LIVE soma 1 ao seu nível.
  const distribution = useMemo(() => {
    const liveContribution = (level: RiskLevel) =>
      effectiveRiskLevel === level ? 1 : 0;

    return {
      healthy: liveContribution("NORMAL") + 3,
      warning: liveContribution("ALERTA") + 1,
      critical: liveContribution("CRÍTICO") + 0,
    };
  }, [effectiveRiskLevel]);

  return (
    // Padding responsivo: respiro horizontal mínimo (px-4) no mobile evita
    // que cards e linhas do feed encostem na borda; px-8 no desktop dá ar
    // suficiente sem desperdiçar viewport.
    <div className="flex h-full flex-col gap-6 px-4 py-6 lg:px-8">
      {/* ── Cabeçalho ── */}
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-500">
            PredictIQ · MetroPT-3
          </p>
          <h1 className="mt-0.5 text-xl font-bold tracking-tight text-slate-900">
            Cockpit Operacional
          </h1>
          <p className="mt-0.5 text-sm text-slate-500">
            Manutenção preditiva e MLOps em tempo real
          </p>
        </div>
        <div className="flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 shadow-sm">
          <span
            className={
              sseStatus === "connected"
                ? "h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-500"
                : "h-1.5 w-1.5 rounded-full bg-amber-500"
            }
            aria-hidden="true"
          />
          <span className="text-[11px] font-medium text-slate-500">
            {sseStatus === "connected" ? "Telemetria ao vivo" : "Reconectando…"}
          </span>
        </div>
      </div>

      {/* ── 4 KPIs industriais ── */}
      <DevProfiler id="FleetKPIs">
        <FleetKPIs
          liveProbability={effectiveProb}
          effectiveRiskLevel={effectiveRiskLevel}
          latencyTelemetry={latencyTelemetry}
          isLoading={isLoading}
        />
      </DevProfiler>

      {/* ── Grid intermediário: tabela (2/3) + modelo (1/3), altura emparelhada.
          `flex-1` faz esta região absorver o resto da altura disponível. ── */}
      <div className="grid flex-1 grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <DevProfiler id="FleetHealthTable">
            <FleetHealthTable
              effectiveRiskLevel={effectiveRiskLevel}
              effectiveProb={effectiveProb}
              isLoading={isLoading}
              selectedId={selectedAssetId}
              onSelect={setSelectedAssetId}
            />
          </DevProfiler>
        </div>

        <div className="lg:col-span-1">
          <DevProfiler id="ModelStatusCard">
            <ModelStatusCard distribution={distribution} />
          </DevProfiler>
        </div>
      </div>

      {/* ── Eventos Recentes: full-width na base, lista em grid responsivo. ── */}
      <DevProfiler id="EventFeedCard">
        <EventFeedCard />
      </DevProfiler>
    </div>
  );
}
