// ── Linha ao vivo (APU-Trem-042) — RNF-58: extraído de AssetTable ───────────

import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { RiskLevel } from "@/hooks/use-sensor-data";
import { ProbabilityCell, RiskBadge, SelectionBar } from "./cells";

export const LIVE_ASSET_ID = "APU-Trem-042";

interface LiveRowProps {
  effectiveRiskLevel: RiskLevel;
  effectiveProb: number;
  tp2: number;
  oilTemp: number;
  isSelected: boolean;
  onSelect: (id: string) => void;
}

export function LiveRow({
  effectiveRiskLevel,
  effectiveProb,
  tp2,
  oilTemp,
  isSelected,
  onSelect,
}: LiveRowProps) {
  return (
    <tr
      onClick={() => onSelect(LIVE_ASSET_ID)}
      className={cn(
        "group/row border-b border-border/50 cursor-pointer",
        "transition-colors duration-150",
        isSelected
          ? "bg-accent/40 border-l-4 border-primary"
          : "hover:bg-muted/50",
      )}
    >
      <td className="py-3 pr-4">
        <div className="flex items-center gap-2">
          <SelectionBar active={isSelected} />
          <span className="relative flex h-2 w-2 shrink-0">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-blue-400 opacity-75" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-blue-500" />
          </span>
          <span className="font-mono text-xs font-semibold text-foreground">
            APU-Trem-042
          </span>
          <Badge
            variant="outline"
            className="border-blue-500/30 bg-blue-500/10 px-1.5 py-0 text-[9px] font-bold tracking-wider text-blue-400"
          >
            LIVE
          </Badge>
        </div>
      </td>

      <td className="py-3 pr-4">
        <RiskBadge level={effectiveRiskLevel} />
      </td>

      <td className="py-3 pr-4 text-right">
        <ProbabilityCell prob={effectiveProb} riskLevel={effectiveRiskLevel} />
      </td>

      <td className="py-3 pr-4 text-right font-mono text-xs tabular-nums text-foreground">
        {tp2.toFixed(2)} <span className="text-muted-foreground">bar</span>
      </td>

      <td className="py-3 pr-4 text-right font-mono text-xs tabular-nums text-foreground">
        {oilTemp.toFixed(1)} <span className="text-muted-foreground">°C</span>
      </td>

      <td className="py-3 pr-3 text-right" onClick={(e) => e.stopPropagation()}>
        <Button
          asChild
          size="sm"
          variant="outline"
          className={cn(
            "h-7 gap-1.5 px-2.5 text-[11px] font-semibold",
            "border-primary/30 bg-primary/5 text-primary",
            "transition-colors duration-150 hover:bg-primary/10 hover:text-primary",
          )}
        >
          <Link href="/sensors/APU-Trem-042">
            Telemetria
            <ArrowRight className="h-3 w-3 transition-transform duration-150 group-hover/row:translate-x-0.5" />
          </Link>
        </Button>
      </td>
    </tr>
  );
}
