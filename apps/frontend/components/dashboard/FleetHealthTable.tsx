"use client";

/**
 * Fleet Health Table — cockpit operacional.
 *
 * Linha 0  : APU-Trem-042 — dados reais (SSE + predição).
 * Linhas 1+: ativos simulados, marcados com tag discreta "Simulado".
 *
 * Cada linha exibe a saúde via <Progress> (verde/laranja/vermelho) em vez de
 * apenas texto, deixando o estado da frota legível em scan rápido.
 *
 * `React.memo` (RNF-39): props (`effectiveRiskLevel`/`effectiveRisk`) só
 * mudam de valor no poll de 5s ou em alerta WS — não no tick SSE de 1Hz.
 * Medido com React Profiler: sem memo, este componente re-renderizava a
 * ~1/tick SSE mesmo com os mesmos valores de props (ver
 * frontend_performance_report.md). `onSelect` é o setter de useState do
 * pai (`setSelectedAssetId`), garantidamente estável entre renders.
 *
 * RNF-58: decomposto em `components/dashboard/fleet-health-table/*` —
 * `MOCK_ASSETS`/`healthToRiskLevel` (mock-assets), `RiskBadge`,
 * `HealthCell`/`TableSkeleton`, `LiveAssetRow`, `MockAssetRow`. Nenhuma
 * mudança de comportamento/DOM.
 */

import { memo } from "react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { RiskLevel } from "@/hooks/use-sensor-data";
import { TableSkeleton } from "./fleet-health-table/health-cell";
import {
  LIVE_ASSET_ID,
  LiveAssetRow,
} from "./fleet-health-table/live-asset-row";
import { MockAssetRow } from "./fleet-health-table/mock-asset-row";
import { MOCK_ASSETS } from "./fleet-health-table/mock-assets";

// ── Component ────────────────────────────────────────────────────────────────

interface FleetHealthTableProps {
  effectiveRiskLevel: RiskLevel;
  /** Índice de risco suavizado do ativo ao vivo (0–1). */
  effectiveRisk: number;
  /** Probabilidade bruta da última predição — só no tooltip. */
  rawProbability: number;
  isLoading: boolean;
  selectedId: string;
  onSelect: (id: string) => void;
}

const COL_HEAD =
  "pb-2.5 text-xs font-semibold uppercase tracking-widest text-muted-foreground";

const FleetHealthTable = memo(function FleetHealthTable({
  effectiveRiskLevel,
  effectiveRisk,
  rawProbability,
  isLoading,
  selectedId,
  onSelect,
}: FleetHealthTableProps) {
  return (
    <Card className="flex h-full flex-col border border-border bg-card shadow-sm ring-0 short:gap-2 short:py-2">
      <CardHeader className="px-5 pb-3 pt-4 short:pb-1 short:pt-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-foreground">
            Saúde da Frota
          </CardTitle>
          <span className="text-xs text-muted-foreground">
            5 ativos · 1 em tempo real
          </span>
        </div>
      </CardHeader>

      <CardContent className="min-h-0 flex-1 overflow-y-auto px-5 pb-5">
        {isLoading ? (
          <TableSkeleton />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th scope="col" className={cn(COL_HEAD, "pl-5 text-left")}>
                    ID do Ativo
                  </th>
                  <th scope="col" className={cn(COL_HEAD, "text-left")}>
                    Status
                  </th>
                  <th scope="col" className={cn(COL_HEAD, "text-left")}>
                    Saúde
                  </th>
                  <th scope="col" className={cn(COL_HEAD, "text-right")}>
                    Risco
                  </th>
                  <th scope="col" className={cn(COL_HEAD, "pr-3 text-right")}>
                    Ação
                  </th>
                </tr>
              </thead>

              <tbody>
                <LiveAssetRow
                  effectiveRiskLevel={effectiveRiskLevel}
                  effectiveRisk={effectiveRisk}
                  rawProbability={rawProbability}
                  isSelected={selectedId === LIVE_ASSET_ID}
                  onSelect={onSelect}
                />

                {MOCK_ASSETS.map((asset) => (
                  <MockAssetRow
                    key={asset.id}
                    asset={asset}
                    isSelected={selectedId === asset.id}
                    onSelect={onSelect}
                  />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
});

export default FleetHealthTable;
