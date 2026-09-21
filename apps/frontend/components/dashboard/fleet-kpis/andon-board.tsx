// ── Andon Board — RNF-58: extraído de FleetKPIs ─────────────────────────────

import { cn } from "@/lib/utils";
import type { RiskLevel } from "@/lib/risk-thresholds";
import { ANDON_COLOR } from "./constants";

export interface AndonAsset {
  id: string;
  status: RiskLevel;
}

export interface AndonBoardProps {
  assets: ReadonlyArray<AndonAsset>;
}

/** Cada bloco = 1 compressor. Status em scan rápido, sem precisar
 * interpretar um número ou ler um label. */
export function AndonBoard({ assets }: AndonBoardProps) {
  return (
    <div
      className="flex h-10 w-full gap-1.5"
      role="group"
      aria-label="Estado da frota — matriz Andon"
    >
      {assets.map((asset) => (
        <div
          key={asset.id}
          role="img"
          title={`${asset.id}: ${asset.status}`}
          aria-label={`${asset.id}: ${asset.status}`}
          className={cn(
            "flex-1 rounded-sm border border-black/5 transition-colors",
            ANDON_COLOR[asset.status],
          )}
        />
      ))}
    </div>
  );
}
