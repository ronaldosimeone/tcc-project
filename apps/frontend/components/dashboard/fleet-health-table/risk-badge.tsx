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
