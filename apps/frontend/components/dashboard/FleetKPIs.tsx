"use client";

/**
 * KPIs do cockpit operacional — 4 cards industriais com sparkline.
 *
 * Cada card mantém: título + valor grande + ícone tonal + sparkline 40–50px
 * (sem eixos, sem grid). A cor da sparkline reflecte a saúde do KPI:
 * verde = bom, âmbar = atenção, rosa = crítico, slate = neutro.
 *
 * Card 4 ("Modelo de IA Ativo") lê o estado real via listModels(); a
 * sparkline aí é neutra (confiança do modelo num histórico mock).
 *
 * RNF-58: decomposto em `components/dashboard/fleet-kpis/*` (constants,
 * KpiSparkline, KpiShell) + `hooks/use-latency-history.ts` (estado do
 * histórico de latência, incl. `LatencyTelemetry`, re-exportado aqui para
 * manter o contrato público). Nenhuma mudança de comportamento/DOM.
 */

import { useMemo } from "react";
import { AlertOctagon, Gauge, HeartPulse, ShieldAlert } from "lucide-react";

import { Skeleton } from "@/components/ui/skeleton";
import {
  useLatencyHistory,
  type LatencyTelemetry,
} from "@/hooks/use-latency-history";
import { getRiskLevel, type RiskLevel } from "@/lib/risk-thresholds";
import { cn } from "@/lib/utils";
import {
  ANDON_COLOR,
  LATENCY_HISTORY_SIZE,
  SPARK_ANOMALY,
  SPARK_HEALTH,
  type SparkTone,
} from "./fleet-kpis/constants";
import { KpiShell } from "./fleet-kpis/kpi-shell";

export type { LatencyTelemetry };

interface FleetKPIsProps {
  /** Probabilidade de falha do ativo ao vivo (0–1). */
  liveProbability: number;
  /** Nível de risco efectivo do ativo ao vivo. */
  effectiveRiskLevel: RiskLevel;
  /** Última leitura de latência reportada pelo backend. `null` → ainda nenhuma. */
  latencyTelemetry: LatencyTelemetry | null;
  isLoading: boolean;
}

export default function FleetKPIs({
  liveProbability,
  effectiveRiskLevel,
  latencyTelemetry,
  isLoading,
}: FleetKPIsProps) {
  // ── KPI 1: Saúde Global ──
  const liveHealth = Math.round((1 - liveProbability) * 100);
  const fleetHealth = Math.round((liveHealth + 95 + 92 + 96 + 88) / 5);

  // ── KPI 4: Latência da inferência ──
  const { latencyHistory, hasLatency, currentLatencyMs } =
    useLatencyHistory(latencyTelemetry);

  // Mapeia a faixa oficial (NORMAL/ALERTA/CRÍTICO) para o tom visual usado
  // tanto no ícone como na sparkline. Único ponto de verdade — qualquer
  // mudança nos limiares (lib/risk-thresholds) propaga automaticamente.
  const anomalyTone: SparkTone =
    getRiskLevel(liveProbability) === "CRÍTICO"
      ? "danger"
      : getRiskLevel(liveProbability) === "ALERTA"
      ? "warn"
      : "ok";

  // ── KPI 2: Ativos em Alerta ──
  const liveIsCritical = effectiveRiskLevel === "CRÍTICO";
  const liveIsAlert = effectiveRiskLevel === "ALERTA";

  // Matriz Andon: 5 blocos representando o estado de cada compressor.
  // O primeiro (APU-Trem-042) é reativo ao stream; os 4 demais seguem o
  // mesmo mock da FleetHealthTable para coerência visual entre componentes.
  const fleetAndon: ReadonlyArray<{ id: string; status: RiskLevel }> = useMemo(
    () => [
      { id: "APU-Trem-042", status: effectiveRiskLevel },
      { id: "APU-Trem-015", status: "NORMAL" },
      { id: "APU-Trem-023", status: "ALERTA" },
      { id: "APU-Trem-031", status: "NORMAL" },
      { id: "APU-Trem-055", status: "NORMAL" },
    ],
    [effectiveRiskLevel],
  );

  const inAlert = fleetAndon.filter((a) => a.status !== "NORMAL").length;

  // ── KPI 3: Anomalia Máxima ──
  const maxAnomalyPct = (liveProbability * 100).toFixed(1);

  // Sparkline da anomalia — extende a série mock com o valor actual no final
  // para que o gráfico reaja à leitura ao vivo do APU-Trem-042.
  const anomalySpark = useMemo(
    () => [...SPARK_ANOMALY.slice(0, -1), Math.round(liveProbability * 100)],
    [liveProbability],
  );

  return (
    <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
      <KpiShell
        title="Saúde Global da Frota"
        value={
          <span>
            {fleetHealth}
            <span className="ml-0.5 text-base font-medium text-slate-400">
              %
            </span>
          </span>
        }
        icon={HeartPulse}
        iconTone="ok"
        spark={SPARK_HEALTH}
        sparkTone="ok"
        isLoading={isLoading}
      />

      <KpiShell
        title="Ativos em Alerta"
        value={
          <span className="flex items-baseline gap-2">
            <span>{inAlert}</span>
            <span className="text-sm font-medium text-slate-400">/ 5</span>
          </span>
        }
        icon={ShieldAlert}
        iconTone={liveIsCritical ? "danger" : liveIsAlert ? "warn" : "ok"}
        isLoading={isLoading}
      >
        {/* Andon Board: cada bloco = 1 compressor. Status em scan rápido,
            sem precisar interpretar um número ou ler um label. */}
        <div
          className="flex h-10 w-full gap-1.5"
          role="group"
          aria-label="Estado da frota — matriz Andon"
        >
          {fleetAndon.map((asset) => (
            <div
              key={asset.id}
              title={`${asset.id}: ${asset.status}`}
              aria-label={`${asset.id}: ${asset.status}`}
              className={cn(
                "flex-1 rounded-sm border border-black/5 transition-colors",
                ANDON_COLOR[asset.status],
              )}
            />
          ))}
        </div>
      </KpiShell>

      <KpiShell
        title="Anomalia Máxima Atual"
        value={
          <span>
            {maxAnomalyPct}
            <span className="ml-0.5 text-base font-medium text-slate-400">
              %
            </span>
          </span>
        }
        icon={AlertOctagon}
        iconTone={anomalyTone}
        spark={anomalySpark}
        sparkTone={anomalyTone}
        isLoading={isLoading}
      />

      <KpiShell
        title="Latência de Inferência"
        value={
          currentLatencyMs !== null ? (
            <span>
              {currentLatencyMs}
              <span className="ml-0.5 text-base font-medium text-slate-400">
                ms
              </span>
            </span>
          ) : (
            // Sem leitura ainda — skeleton honesto em vez de "40 ms" fantasma.
            <Skeleton className="h-7 w-20" />
          )
        }
        subtitle={
          hasLatency
            ? `ao vivo · ${latencyHistory.length}/${LATENCY_HISTORY_SIZE} amostras`
            : "aguardando primeira inferência…"
        }
        icon={Gauge}
        iconTone="neutral"
        spark={latencyHistory}
        sparkTone="neutral"
        isLoading={isLoading}
      />
    </div>
  );
}
