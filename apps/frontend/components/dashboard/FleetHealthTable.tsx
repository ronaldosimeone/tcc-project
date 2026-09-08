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
 * `React.memo` (RNF-39): props (`effectiveRiskLevel`/`effectiveProb`) só
 * mudam de valor no poll de 5s ou em alerta WS — não no tick SSE de 1Hz.
 * Medido com React Profiler: sem memo, este componente re-renderizava a
 * ~1/tick SSE mesmo com os mesmos valores de props (ver
 * frontend_performance_report.md). `onSelect` é o setter de useState do
 * pai (`setSelectedAssetId`), garantidamente estável entre renders.
 */

import Link from "next/link";
import { memo, type ComponentType } from "react";
import {
  AlertTriangle,
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  CheckCircle2,
  XCircle,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { RISK_THRESHOLDS } from "@/lib/risk-thresholds";
import { cn } from "@/lib/utils";
import type { RiskLevel } from "@/hooks/use-sensor-data";

// ── Mock data ────────────────────────────────────────────────────────────────

export interface MockAsset {
  id: string;
  riskLevel: RiskLevel;
  /** Saúde sintética [0–100] derivada do mock. */
  health: number;
  /** Variação de saúde nas últimas 24h em pontos percentuais (+/−). */
  trendPp: number;
  prob: number;
  lastSeen: string;
}

// Invariante mantida: prob === (100 - health) / 100. Saúde + Risco somam 100.
export const MOCK_ASSETS: MockAsset[] = [
  {
    id: "APU-Trem-015",
    riskLevel: "NORMAL",
    health: 95,
    trendPp: 1.2,
    prob: 0.05,
    lastSeen: "2 min",
  },
  {
    id: "APU-Trem-023",
    riskLevel: "ALERTA",
    health: 62,
    trendPp: -4.5,
    prob: 0.38,
    lastSeen: "1 min",
  },
  {
    id: "APU-Trem-031",
    riskLevel: "NORMAL",
    health: 96,
    trendPp: 0.3,
    prob: 0.04,
    lastSeen: "3 min",
  },
  {
    id: "APU-Trem-055",
    riskLevel: "NORMAL",
    health: 88,
    trendPp: -1.1,
    prob: 0.12,
    lastSeen: "4 min",
  },
];

// ── Helpers ──────────────────────────────────────────────────────────────────

const RISK_CONFIG: Record<
  RiskLevel,
  {
    icon: ComponentType<{ className?: string }>;
    badgeClass: string;
  }
> = {
  NORMAL: {
    icon: CheckCircle2,
    badgeClass: "border-emerald-200 bg-emerald-50 text-emerald-700",
  },
  ALERTA: {
    icon: AlertTriangle,
    badgeClass: "border-amber-200 bg-amber-50 text-amber-700",
  },
  CRÍTICO: {
    icon: XCircle,
    badgeClass: "border-red-200 bg-red-50 text-red-700",
  },
};

/**
 * Mapeia saúde [0–100] no nível de risco oficial.
 *
 * Invariante: `prob = (100 - health) / 100`, portanto:
 *   - health ≥ 65  ↔ prob < 0.35 → NORMAL
 *   - 35 ≤ h < 65  ↔ 0.35 ≤ prob < 0.65 → ALERTA
 *   - health < 35  ↔ prob ≥ 0.65 → CRÍTICO
 * As fronteiras 65/35 são derivadas de RISK_THRESHOLDS — não duplicar.
 */
function healthToRiskLevel(health: number): RiskLevel {
  const probEquivalent = (100 - health) / 100;
  if (probEquivalent < RISK_THRESHOLDS.ALERT) return "NORMAL";
  if (probEquivalent < RISK_THRESHOLDS.CRITICAL) return "ALERTA";
  return "CRÍTICO";
}

/** Cor da barra de progresso por nível de risco. */
function healthBarClass(health: number): string {
  const level = healthToRiskLevel(health);
  if (level === "NORMAL") return "[&>*]:bg-emerald-500";
  if (level === "ALERTA") return "[&>*]:bg-amber-500";
  return "[&>*]:bg-red-500";
}

/** Cor do texto do percentual por nível de risco. */
function healthTextClass(health: number): string {
  const level = healthToRiskLevel(health);
  if (level === "NORMAL") return "text-emerald-600";
  if (level === "ALERTA") return "text-amber-700";
  return "text-red-700";
}

function RiskBadge({ level }: { level: RiskLevel }) {
  const { icon: Icon, badgeClass } = RISK_CONFIG[level];
  return (
    <Badge
      variant="outline"
      className={cn("gap-1 text-[11px] font-semibold", badgeClass)}
    >
      <Icon className="h-3 w-3" />
      {level}
    </Badge>
  );
}

interface HealthCellProps {
  health: number;
  /** Variação 24h em pontos percentuais; positivo melhora, negativo piora. */
  trendPp: number;
}

function HealthCell({ health, trendPp }: HealthCellProps) {
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

function TableSkeleton() {
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

// ── Component ────────────────────────────────────────────────────────────────

interface FleetHealthTableProps {
  effectiveRiskLevel: RiskLevel;
  effectiveProb: number;
  isLoading: boolean;
  selectedId: string;
  onSelect: (id: string) => void;
}

const LIVE_ASSET_ID = "APU-Trem-042";

const FleetHealthTable = memo(function FleetHealthTable({
  effectiveRiskLevel,
  effectiveProb,
  isLoading,
  selectedId,
  onSelect,
}: FleetHealthTableProps) {
  const liveHealth = Math.round((1 - effectiveProb) * 100);
  // Tendência da linha LIVE: deriva do nível de risco corrente para dar
  // feedback imediato no scan visual (sem necessitar de histórico real).
  const liveTrendPp =
    effectiveRiskLevel === "CRÍTICO"
      ? -8.0
      : effectiveRiskLevel === "ALERTA"
      ? -2.5
      : 1.2;
  const isLiveSelected = selectedId === LIVE_ASSET_ID;
  const liveIsCritical = effectiveRiskLevel === "CRÍTICO";
  const liveIsAlert = effectiveRiskLevel === "ALERTA";

  const COL_HEAD =
    "pb-2.5 text-[10px] font-semibold uppercase tracking-widest text-slate-500";

  return (
    <Card className="flex h-full flex-col border border-slate-200 bg-white shadow-sm ring-0">
      <CardHeader className="px-5 pb-3 pt-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-slate-900">
            Saúde da Frota
          </CardTitle>
          <span className="text-[11px] text-slate-500">
            5 ativos · 1 em tempo real
          </span>
        </div>
      </CardHeader>

      <CardContent className="px-5 pb-5">
        {isLoading ? (
          <TableSkeleton />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200">
                  <th className={cn(COL_HEAD, "pl-5 text-left")}>
                    ID do Ativo
                  </th>
                  <th className={cn(COL_HEAD, "text-left")}>Status</th>
                  <th className={cn(COL_HEAD, "text-left")}>Saúde</th>
                  <th className={cn(COL_HEAD, "text-right")}>Risco</th>
                  <th className={cn(COL_HEAD, "pr-3 text-right")}>Ação</th>
                </tr>
              </thead>

              <tbody>
                {/* ── Linha ao vivo: APU-Trem-042 ──
                    Cores estritas por nível:
                      ALERTA  → âmbar (não destaca demais)
                      CRÍTICO → vermelho + pulse (atenção imediata)
                      NORMAL  → cinza neutro */}
                <tr
                  onClick={() => onSelect(LIVE_ASSET_ID)}
                  className={cn(
                    "group/row cursor-pointer border-b border-slate-100 transition-colors",
                    liveIsCritical
                      ? "bg-red-50 hover:bg-red-100"
                      : liveIsAlert
                      ? "bg-amber-50 hover:bg-amber-100"
                      : isLiveSelected
                      ? "bg-slate-50"
                      : "hover:bg-slate-50/60",
                  )}
                >
                  <td className="py-3 pl-5 pr-4">
                    <div className="flex items-center gap-2">
                      <span className="relative flex h-2 w-2 shrink-0">
                        {/* Dot tonal: vermelho em CRÍTICO, âmbar em ALERTA,
                            azul "LIVE" discreto em operação normal. */}
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
                    </div>
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

                  <td
                    className="py-3 pr-3 text-right"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <Link
                      href={`/sensors/${LIVE_ASSET_ID}`}
                      className="inline-flex items-center gap-1 rounded-md border border-slate-200 px-2 py-1 text-[11px] font-semibold text-slate-700 transition-colors hover:bg-slate-50"
                    >
                      Telemetria
                      <ArrowRight className="h-3 w-3" />
                    </Link>
                  </td>
                </tr>

                {/* ── Linhas simuladas ── */}
                {MOCK_ASSETS.map((asset) => {
                  const isMockSelected = selectedId === asset.id;
                  return (
                    <tr
                      key={asset.id}
                      onClick={() => onSelect(asset.id)}
                      className={cn(
                        "cursor-pointer border-b border-slate-100 transition-colors",
                        isMockSelected ? "bg-slate-50" : "hover:bg-slate-50/60",
                      )}
                    >
                      <td className="py-3 pl-5 pr-4">
                        <div className="flex items-center gap-2">
                          <span
                            className="h-1.5 w-1.5 shrink-0 rounded-full bg-slate-300"
                            aria-hidden="true"
                          />
                          <span className="font-mono text-xs text-slate-700">
                            {asset.id}
                          </span>
                          <Badge
                            variant="outline"
                            className="border-slate-200 bg-slate-50 px-1.5 py-0 text-[9px] font-medium tracking-wider text-slate-500"
                          >
                            SIMULADO
                          </Badge>
                        </div>
                      </td>

                      <td className="py-3 pr-4">
                        <RiskBadge level={asset.riskLevel} />
                      </td>

                      <td className="py-3 pr-4">
                        <HealthCell
                          health={asset.health}
                          trendPp={asset.trendPp}
                        />
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
                })}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
});

export default FleetHealthTable;
