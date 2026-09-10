"use client";

/**
 * RNF-58: decomposto em `components/dashboard/asset-table/*` —
 * mock-assets, cells (RiskBadge/ProbabilityCell/SelectionBar),
 * TableSkeleton, LiveRow, MockRow. Nenhuma mudança de comportamento/DOM.
 */

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { RiskLevel } from "@/hooks/use-sensor-data";
import { LIVE_ASSET_ID, LiveRow } from "./asset-table/live-row";
import { MockRow } from "./asset-table/mock-row";
import { MOCK_ASSETS } from "./asset-table/mock-assets";
import { TableSkeleton } from "./asset-table/table-skeleton";

// ── Props ─────────────────────────────────────────────────────────────────

interface AssetTableProps {
  effectiveRiskLevel: RiskLevel;
  effectiveProb: number;
  tp2: number;
  oilTemp: number;
  isLoading: boolean;
  selectedId: string;
  onSelect: (id: string) => void;
}

const COL_HEAD =
  "pb-2.5 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground";

// ── Component ─────────────────────────────────────────────────────────────

export default function AssetTable({
  effectiveRiskLevel,
  effectiveProb,
  tp2,
  oilTemp,
  isLoading,
  selectedId,
  onSelect,
}: AssetTableProps) {
  return (
    <Card className="border-border bg-card">
      <CardHeader className="px-5 pb-3 pt-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-foreground/90">
            Ativos da Frota
          </CardTitle>
          <span className="text-[11px] text-muted-foreground">
            {5} ativos · 1 em tempo real
          </span>
        </div>
      </CardHeader>

      <CardContent className="px-5 pb-5">
        {isLoading ? (
          <TableSkeleton />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm" role="table">
              <thead>
                <tr className="border-b border-border">
                  <th className={cn(COL_HEAD, "text-left")}>ID do Ativo</th>
                  <th className={cn(COL_HEAD, "text-left")}>Status</th>
                  <th className={cn(COL_HEAD, "text-right")}>Prob. Falha</th>
                  <th className={cn(COL_HEAD, "text-right")}>TP2</th>
                  <th className={cn(COL_HEAD, "text-right")}>Temp. Óleo</th>
                  <th className={cn(COL_HEAD, "pr-3 text-right")}>Ação</th>
                </tr>
              </thead>

              <tbody>
                <LiveRow
                  effectiveRiskLevel={effectiveRiskLevel}
                  effectiveProb={effectiveProb}
                  tp2={tp2}
                  oilTemp={oilTemp}
                  isSelected={selectedId === LIVE_ASSET_ID}
                  onSelect={onSelect}
                />

                {MOCK_ASSETS.map((asset) => (
                  <MockRow
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
}
