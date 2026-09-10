// ── Badge de risco — RNF-58: extraído de alert-panel.tsx ────────────────────

import { AlertTriangle, CheckCircle2, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { RiskLevel } from "@/hooks/use-sensor-data";

export const RISK_CONFIG: Record<
  RiskLevel,
  {
    badgeClass: string;
    icon: React.ComponentType<{ className?: string }>;
  }
> = {
  NORMAL: {
    badgeClass: "border-green-500/40 bg-green-500/10 text-green-400",
    icon: CheckCircle2,
  },
  ALERTA: {
    badgeClass: "border-amber-500/40 bg-amber-500/10 text-amber-400",
    icon: AlertTriangle,
  },
  CRÍTICO: {
    badgeClass: "border-red-500/40 bg-red-500/10 text-red-400",
    icon: XCircle,
  },
};

export function RiskBadge({ level }: { level: RiskLevel }) {
  const { icon: Icon, badgeClass } = RISK_CONFIG[level];
  return (
    <Badge
      variant="outline"
      className={cn("gap-1 px-2 py-0.5 text-[10px] font-bold", badgeClass)}
    >
      <Icon className="h-3 w-3" />
      {level}
    </Badge>
  );
}
