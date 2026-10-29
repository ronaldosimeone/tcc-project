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
  effectiveRisk: number;
  rawProbability: number;
  isSelected: boolean;
  onSelect: (id: string) => void;
}

export function LiveAssetRow({
  effectiveRiskLevel,
  effectiveRisk,
  rawProbability,
  isSelected,
  onSelect,
}: LiveAssetRowProps) {
  const liveHealth = Math.round((1 - effectiveRisk) * 100);
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
        "group/row cursor-pointer border-b border-border/60 transition-colors",
        liveIsCritical
          ? "bg-red-500/10 hover:bg-red-500/15"
          : liveIsAlert
          ? "bg-amber-500/10 hover:bg-amber-500/15"
          : isSelected
          ? "bg-muted/50"
          : "hover:bg-muted/60",
      )}
    >
      <td className="py-3 short:py-1.5 pl-5 pr-4">
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
          <span className="whitespace-nowrap font-mono text-xs font-semibold text-foreground">
            APU-Trem-042
          </span>
          <Badge
            variant="outline"
            className={cn(
              "px-1.5 py-0 text-xs font-bold tracking-wider",
              liveIsCritical
                ? "animate-pulse border-red-500/30 bg-red-500/15 text-red-700 dark:text-red-400"
                : liveIsAlert
                ? "border-amber-500/30 bg-amber-500/15 text-amber-700 dark:text-amber-400"
                : "border-blue-500/30 bg-blue-500/10 text-blue-700 dark:text-blue-400",
            )}
          >
            AO VIVO
          </Badge>
        </button>
      </td>

      <td className="py-3 short:py-1.5 pr-4">
        <RiskBadge level={effectiveRiskLevel} />
      </td>

      <td className="py-3 short:py-1.5 pr-4">
        <HealthCell
          health={liveHealth}
          trendPp={liveTrendPp}
          riskLevel={effectiveRiskLevel}
        />
      </td>

      <td
        className="py-3 short:py-1.5 pr-4 text-right font-mono text-xs tabular-nums text-foreground"
        title={`Probabilidade bruta do modelo: ${(rawProbability * 100).toFixed(
          1,
        )}%`}
      >
        {(effectiveRisk * 100).toFixed(1)}%
      </td>

      <td
        className="py-3 short:py-1.5 pr-3 text-right"
        onClick={(e) => e.stopPropagation()}
      >
        <Link
          href={`/sensors/${LIVE_ASSET_ID}`}
          className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs font-semibold text-foreground/80 transition-colors hover:bg-muted/50"
        >
          Telemetria
          <ArrowRight className="h-3 w-3" />
        </Link>
      </td>
    </tr>
  );
}
