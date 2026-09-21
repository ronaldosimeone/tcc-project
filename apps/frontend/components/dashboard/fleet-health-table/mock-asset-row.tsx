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
        "cursor-pointer border-b border-slate-100 transition-colors",
        isSelected ? "bg-slate-50" : "hover:bg-slate-50/60",
      )}
    >
      <td className="py-3 pl-5 pr-4">
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
            className="h-1.5 w-1.5 shrink-0 rounded-full bg-slate-300"
            aria-hidden="true"
          />
          <span className="font-mono text-xs text-slate-700">{asset.id}</span>
          <Badge
            variant="outline"
            className="border-slate-200 bg-slate-50 px-1.5 py-0 text-[9px] font-medium tracking-wider text-slate-500"
          >
            SIMULADO
          </Badge>
        </button>
      </td>

      <td className="py-3 pr-4">
        <RiskBadge level={asset.riskLevel} />
      </td>

      <td className="py-3 pr-4">
        <HealthCell health={asset.health} trendPp={asset.trendPp} />
      </td>

      <td className="py-3 pr-4 text-right font-mono text-xs tabular-nums text-slate-700">
        {(asset.prob * 100).toFixed(1)}%
      </td>

      <td className="py-3 pr-3 text-right">
        <span className="font-mono text-[10px] italic text-slate-400">
          há {asset.lastSeen}
        </span>
      </td>
    </tr>
  );
}
