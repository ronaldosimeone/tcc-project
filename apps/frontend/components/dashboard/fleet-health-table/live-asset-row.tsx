// ── Linha "ao vivo" (APU-Trem-042) — RNF-58: extraído de FleetHealthTable ───
//
// Cores estritas por nível: ALERTA → âmbar, CRÍTICO → vermelho + pulse,
// NORMAL → cinza neutro. Nenhuma mudança de comportamento/DOM.

import Link from "next/link";
import { ArrowRight } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { RiskLevel } from "@/hooks/use-sensor-data";
import { HealthCell } from "./health-cell";
import { RiskBadge } from "./risk-badge";

export const LIVE_ASSET_ID = "APU-Trem-042";

interface LiveAssetRowProps {
  effectiveRiskLevel: RiskLevel;
  effectiveProb: number;
  isSelected: boolean;
  onSelect: (id: string) => void;
}

export function LiveAssetRow({
  effectiveRiskLevel,
  effectiveProb,
  isSelected,
  onSelect,
}: LiveAssetRowProps) {
  const liveHealth = Math.round((1 - effectiveProb) * 100);
  // Tendência da linha LIVE: deriva do nível de risco corrente para dar
  // feedback imediato no scan visual (sem necessitar de histórico real).
  const liveTrendPp =
    effectiveRiskLevel === "CRÍTICO"
      ? -8.0
      : effectiveRiskLevel === "ALERTA"
      ? -2.5
      : 1.2;
  const liveIsCritical = effectiveRiskLevel === "CRÍTICO";
  const liveIsAlert = effectiveRiskLevel === "ALERTA";

  return (
    <tr
      onClick={() => onSelect(LIVE_ASSET_ID)}
      className={cn(
        "group/row cursor-pointer border-b border-slate-100 transition-colors",
        liveIsCritical
          ? "bg-red-50 hover:bg-red-100"
          : liveIsAlert
          ? "bg-amber-50 hover:bg-amber-100"
          : isSelected
          ? "bg-slate-50"
          : "hover:bg-slate-50/60",
      )}
    >
      <td className="py-3 pl-5 pr-4">
        <button
          type="button"
          aria-pressed={isSelected}
          onClick={(e) => {
            e.stopPropagation();
            onSelect(LIVE_ASSET_ID);
          }}
          className="flex items-center gap-2 rounded-sm outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
        >
          <span className="relative flex h-2 w-2 shrink-0" aria-hidden="true">
            {/* Dot tonal: vermelho em CRÍTICO, âmbar em ALERTA, azul "LIVE"
                discreto em operação normal. */}
            <span
              className={cn(
                "absolute inline-flex h-full w-full rounded-full opacity-75 animate-ping",
                liveIsCritical
                  ? "bg-red-500"
                  : liveIsAlert
                  ? "bg-amber-500"
                  : "bg-blue-400",
              )}
            />
            <span
              className={cn(
                "relative inline-flex h-2 w-2 rounded-full",
                liveIsCritical
                  ? "bg-red-600"
                  : liveIsAlert
                  ? "bg-amber-600"
                  : "bg-blue-500",
              )}
            />
          </span>
          <span className="font-mono text-xs font-semibold text-slate-900">
            APU-Trem-042
          </span>
          <Badge
            variant="outline"
            className={cn(
              "px-1.5 py-0 text-[9px] font-bold tracking-wider",
              liveIsCritical
                ? "animate-pulse border-red-300 bg-red-100 text-red-700"
                : liveIsAlert
                ? "border-amber-300 bg-amber-100 text-amber-700"
                : "border-blue-200 bg-blue-50 text-blue-700",
            )}
          >
            LIVE
          </Badge>
        </button>
      </td>

      <td className="py-3 pr-4">
        <RiskBadge level={effectiveRiskLevel} />
      </td>

      <td className="py-3 pr-4">
        <HealthCell health={liveHealth} trendPp={liveTrendPp} />
      </td>

      <td className="py-3 pr-4 text-right font-mono text-xs tabular-nums text-slate-900">
        {(effectiveProb * 100).toFixed(1)}%
      </td>

      <td className="py-3 pr-3 text-right" onClick={(e) => e.stopPropagation()}>
        <Link
          href={`/sensors/${LIVE_ASSET_ID}`}
          className="inline-flex items-center gap-1 rounded-md border border-slate-200 px-2 py-1 text-[11px] font-semibold text-slate-700 transition-colors hover:bg-slate-50"
        >
          Telemetria
          <ArrowRight className="h-3 w-3" />
        </Link>
      </td>
    </tr>
  );
}
