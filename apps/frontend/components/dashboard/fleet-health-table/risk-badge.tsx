// ── RiskBadge — RNF-58: extraído de FleetHealthTable.tsx ────────────────────

import type { ComponentType } from "react";
import { AlertTriangle, CheckCircle2, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { RiskLevel } from "@/hooks/use-sensor-data";
import { cn } from "@/lib/utils";

const RISK_CONFIG: Record<
  RiskLevel,
  {
    icon: ComponentType<{ className?: string }>;
    badgeClass: string;
  }
> = {
  NORMAL: {
    icon: CheckCircle2,
    badgeClass:
      "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  },
  ALERTA: {
    icon: AlertTriangle,
    badgeClass:
      "border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-400",
  },
  CRÍTICO: {
    icon: XCircle,
    badgeClass:
      "border-red-500/30 bg-red-500/10 text-red-700 dark:text-red-400",
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
