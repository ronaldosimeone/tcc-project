// ── Linha de ativo simulado — RNF-58: extraído de FleetHealthTable ──────────

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { HealthCell } from "./health-cell";
import type { MockAsset } from "./mock-assets";
import { RiskBadge } from "./risk-badge";

interface MockAssetRowProps {
  asset: MockAsset;
  isSelected: boolean;
  onSelect: (id: string) => void;
}

export function MockAssetRow({
  asset,
  isSelected,
  onSelect,
}: MockAssetRowProps) {
  return (
    <tr
      onClick={() => onSelect(asset.id)}
      className={cn(
        "cursor-pointer border-b border-border/60 transition-colors",
        isSelected ? "bg-muted/50" : "hover:bg-muted/60",
      )}
    >
      <td className="py-3 short:py-1.5 pl-5 pr-4">
        <button
          type="button"
          aria-pressed={isSelected}
          onClick={(e) => {
            e.stopPropagation();
            onSelect(asset.id);
          }}
          className="flex items-center gap-2 rounded-sm outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
        >
          <span
            className="h-1.5 w-1.5 shrink-0 rounded-full bg-muted-foreground/30"
            aria-hidden="true"
          />
          <span className="whitespace-nowrap font-mono text-xs text-foreground/80">
            {asset.id}
          </span>
          <Badge
            variant="outline"
            className="border-border bg-muted/50 px-1.5 py-0 text-[9px] font-medium tracking-wider text-muted-foreground"
          >
            SIMULADO
          </Badge>
        </button>
      </td>

      <td className="py-3 short:py-1.5 pr-4">
        <RiskBadge level={asset.riskLevel} />
      </td>

      <td className="py-3 short:py-1.5 pr-4">
        <HealthCell health={asset.health} trendPp={asset.trendPp} />
      </td>

      <td className="py-3 short:py-1.5 pr-4 text-right font-mono text-xs tabular-nums text-foreground/80">
        {(asset.prob * 100).toFixed(1)}%
      </td>

      <td className="py-3 short:py-1.5 pr-3 text-right">
        <span className="font-mono text-[10px] italic text-muted-foreground/70">
          há {asset.lastSeen}
        </span>
      </td>
    </tr>
  );
}
