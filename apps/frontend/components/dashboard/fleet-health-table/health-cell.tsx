// ── HealthCell + TableSkeleton — RNF-58: extraído de FleetHealthTable.tsx ───

import { ArrowDownRight, ArrowUpRight } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { healthBarClass, healthTextClass } from "./mock-assets";

export interface HealthCellProps {
  health: number;
  /** Variação 24h em pontos percentuais; positivo melhora, negativo piora. */
  trendPp: number;
}

export function HealthCell({ health, trendPp }: HealthCellProps) {
  const TrendIcon = trendPp >= 0 ? ArrowUpRight : ArrowDownRight;
  // Seta da tendência: verde se melhorando, vermelho se piorando.
  // Aqui o vermelho é semântico ("piora") e não denota nível CRÍTICO.
  const trendCls = trendPp >= 0 ? "text-emerald-600" : "text-red-600";
  return (
    <div className="flex items-center gap-2">
      <Progress
        value={health}
        className={cn("h-1.5 w-24 bg-slate-100", healthBarClass(health))}
      />
      <span
        className={cn(
          "min-w-[3ch] font-mono text-xs font-semibold tabular-nums",
          healthTextClass(health),
        )}
      >
        {health}%
      </span>
      <TrendIcon
        className={cn("h-3 w-3 shrink-0", trendCls)}
        aria-label={`Tendência ${trendPp >= 0 ? "positiva" : "negativa"}`}
      />
    </div>
  );
}

export function TableSkeleton() {
  return (
    <div className="space-y-3">
      {Array.from({ length: 5 }).map((_, i) => (
        <div key={i} className="flex items-center gap-4 py-2.5">
          <Skeleton className="h-4 w-28" />
          <Skeleton className="h-5 w-20" />
          <Skeleton className="ml-auto h-1.5 w-32" />
          <Skeleton className="h-4 w-12" />
        </div>
      ))}
    </div>
  );
}
