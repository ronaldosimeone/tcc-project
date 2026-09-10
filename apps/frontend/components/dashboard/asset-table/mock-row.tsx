// ── Linha de ativo mockado — RNF-58: extraído de AssetTable ─────────────────

import { cn } from "@/lib/utils";
import { ProbabilityCell, RiskBadge, SelectionBar } from "./cells";
import type { MockAsset } from "./mock-assets";

interface MockRowProps {
  asset: MockAsset;
  isSelected: boolean;
  onSelect: (id: string) => void;
}

export function MockRow({ asset, isSelected, onSelect }: MockRowProps) {
  return (
    <tr
      onClick={() => onSelect(asset.id)}
      className={cn(
        "border-b border-border/50 cursor-pointer",
        "transition-colors duration-150",
        isSelected
          ? "bg-accent/40 border-l-4 border-primary"
          : "hover:bg-muted/50",
      )}
    >
      <td className="py-3 pr-4">
        <div className="flex items-center gap-2">
          <SelectionBar active={isSelected} />
          <span
            className="h-1.5 w-1.5 shrink-0 rounded-full bg-muted-foreground/30"
            aria-hidden="true"
          />
          <span className="font-mono text-xs text-muted-foreground">
            {asset.id}
          </span>
        </div>
      </td>

      <td className="py-3 pr-4">
        <RiskBadge level={asset.riskLevel} />
      </td>

      <td className="py-3 pr-4 text-right">
        <ProbabilityCell prob={asset.prob} riskLevel={asset.riskLevel} />
      </td>

      <td className="py-3 pr-4 text-right font-mono text-xs tabular-nums text-muted-foreground">
        {asset.tp2.toFixed(2)}{" "}
        <span className="text-muted-foreground/50">bar</span>
      </td>

      <td className="py-3 pr-4 text-right font-mono text-xs tabular-nums text-muted-foreground">
        {asset.oilTemp.toFixed(1)}{" "}
        <span className="text-muted-foreground/50">°C</span>
      </td>

      <td className="py-3 pr-3 text-right">
        <span className="font-mono text-[10px] italic text-muted-foreground/40">
          {asset.lastSeen} atrás
        </span>
      </td>
    </tr>
  );
}
