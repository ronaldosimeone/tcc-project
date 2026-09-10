// ── Células/badges reutilizadas — RNF-58: extraído de AssetTable ────────────

import type { ComponentType } from "react";
import { AlertTriangle, CheckCircle2, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { RiskLevel } from "@/hooks/use-sensor-data";

export const RISK_CONFIG: Record<
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

export function RiskBadge({ level }: { level: RiskLevel }) {
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

export function ProbabilityCell({
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

export function SelectionBar({ active }: { active: boolean }) {
  return (
    <span
      className={cn(
        "h-5 w-[3px] shrink-0 rounded-full transition-all duration-200",
        active ? "bg-primary" : "bg-transparent",
      )}
      aria-hidden="true"
    />
  );
}
