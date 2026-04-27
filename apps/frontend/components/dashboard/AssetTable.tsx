"use client";

import type { ComponentType } from "react";
import Link from "next/link";
import { ArrowRight, CheckCircle2, AlertTriangle, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import type { RiskLevel } from "@/hooks/use-sensor-data";

// ── Props ─────────────────────────────────────────────────────────────────

interface AssetTableProps {
  effectiveRiskLevel: RiskLevel;
  effectiveProb: number;
  tp2: number;
  oilTemp: number;
  isLoading: boolean;
}

// ── Mock data ─────────────────────────────────────────────────────────────

interface MockAsset {
  id: string;
  riskLevel: RiskLevel;
  prob: number;
  tp2: number;
  oilTemp: number;
  lastSeen: string;
}

const MOCK_ASSETS: MockAsset[] = [
  {
    id: "APU-Trem-015",
    riskLevel: "NORMAL",
    prob: 0.123,
    tp2: 8.2,
    oilTemp: 72.0,
    lastSeen: "2 min",
  },
  {
    id: "APU-Trem-023",
    riskLevel: "ALERTA",
    prob: 0.456,
    tp2: 7.8,
    oilTemp: 81.3,
    lastSeen: "1 min",
  },
  {
    id: "APU-Trem-031",
    riskLevel: "NORMAL",
    prob: 0.089,
    tp2: 8.4,
    oilTemp: 68.5,
    lastSeen: "3 min",
  },
  {
    id: "APU-Trem-055",
    riskLevel: "NORMAL",
    prob: 0.221,
    tp2: 8.1,
    oilTemp: 74.1,
    lastSeen: "4 min",
  },
];

// ── Helpers ───────────────────────────────────────────────────────────────

const RISK_CONFIG: Record<
  RiskLevel,
  {
    icon: ComponentType<{ className?: string }>;
    badgeClass: string;
    barClass: string;
  }
> = {
  NORMAL: {
    icon: CheckCircle2,
    badgeClass: "border-green-500/40 bg-green-500/10 text-green-400",
    barClass: "bg-green-500",
  },
  ALERTA: {
    icon: AlertTriangle,
    badgeClass: "border-amber-500/40 bg-amber-500/10 text-amber-400",
    barClass: "bg-amber-500",
  },
  CRÍTICO: {
    icon: XCircle,
    badgeClass: "border-red-500/40 bg-red-500/10 text-red-400",
    barClass: "bg-red-500",
  },
};

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

function ProbabilityCell({
  prob,
  riskLevel,
}: {
  prob: number;
  riskLevel: RiskLevel;
}) {
  const { barClass } = RISK_CONFIG[riskLevel];
  return (
    <div className="flex flex-col items-end gap-1">
      <span className="font-mono text-xs tabular-nums text-foreground">
        {(prob * 100).toFixed(1)}%
      </span>
      <div
        className="h-1 w-16 overflow-hidden rounded-full bg-muted"
        aria-hidden="true"
      >
        <div
          className={cn(
            "h-full rounded-full transition-[width] duration-500",
            barClass,
          )}
          style={{ width: `${Math.min(100, prob * 100)}%` }}
        />
      </div>
    </div>
  );
}

// ── Loading skeleton ───────────────────────────────────────────────────────

function TableSkeleton() {
  return (
    <div className="space-y-3">
      {Array.from({ length: 5 }).map((_, i) => (
        <div key={i} className="flex items-center gap-4 py-2.5">
          <Skeleton className="h-4 w-28" />
          <Skeleton className="h-5 w-20" />
          <Skeleton className="ml-auto h-4 w-12" />
          <Skeleton className="h-4 w-14" />
          <Skeleton className="h-4 w-14" />
          <Skeleton className="h-7 w-24" />
        </div>
      ))}
    </div>
  );
}

// ── Component ─────────────────────────────────────────────────────────────

export default function AssetTable({
  effectiveRiskLevel,
  effectiveProb,
  tp2,
  oilTemp,
  isLoading,
}: AssetTableProps) {
  const COL_HEAD =
    "pb-2.5 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground";

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
                  <th className={cn(COL_HEAD, "text-right")}>Ação</th>
                </tr>
              </thead>

              <tbody>
                {/* ── Linha real: APU-Trem-042 ── */}
                <tr
                  className={cn(
                    "group/row border-b border-border/50",
                    "transition-colors duration-150 hover:bg-muted/20",
                  )}
                >
                  <td className="py-3 pr-4">
                    <div className="flex items-center gap-2.5">
                      {/* Pulsing live dot */}
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
                    <ProbabilityCell
                      prob={effectiveProb}
                      riskLevel={effectiveRiskLevel}
                    />
                  </td>

                  <td className="py-3 pr-4 text-right font-mono text-xs tabular-nums text-foreground">
                    {tp2.toFixed(2)}{" "}
                    <span className="text-muted-foreground">bar</span>
                  </td>

                  <td className="py-3 pr-4 text-right font-mono text-xs tabular-nums text-foreground">
                    {oilTemp.toFixed(1)}{" "}
                    <span className="text-muted-foreground">°C</span>
                  </td>

                  <td className="py-3 text-right">
                    <Button
                      asChild
                      size="sm"
                      variant="outline"
                      className={cn(
                        "h-7 gap-1.5 px-2.5 text-[11px] font-semibold",
                        "border-primary/30 bg-primary/5 text-primary",
                        "transition-colors duration-150",
                        "hover:bg-primary/10 hover:text-primary",
                      )}
                    >
                      <Link href="/sensors/APU-Trem-042">
                        Telemetria
                        <ArrowRight className="h-3 w-3 transition-transform duration-150 group-hover/row:translate-x-0.5" />
                      </Link>
                    </Button>
                  </td>
                </tr>

                {/* ── Linhas mockadas ── */}
                {MOCK_ASSETS.map((asset) => (
                  <tr
                    key={asset.id}
                    className={cn(
                      "border-b border-border/50 last:border-0",
                      "transition-colors duration-150 hover:bg-muted/10",
                    )}
                  >
                    <td className="py-3 pr-4">
                      <div className="flex items-center gap-2.5">
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
                      <ProbabilityCell
                        prob={asset.prob}
                        riskLevel={asset.riskLevel}
                      />
                    </td>

                    <td className="py-3 pr-4 text-right font-mono text-xs tabular-nums text-muted-foreground">
                      {asset.tp2.toFixed(2)}{" "}
                      <span className="text-muted-foreground/50">bar</span>
                    </td>

                    <td className="py-3 pr-4 text-right font-mono text-xs tabular-nums text-muted-foreground">
                      {asset.oilTemp.toFixed(1)}{" "}
                      <span className="text-muted-foreground/50">°C</span>
                    </td>

                    <td className="py-3 text-right">
                      <span className="font-mono text-[10px] italic text-muted-foreground/40">
                        {asset.lastSeen} atrás
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
