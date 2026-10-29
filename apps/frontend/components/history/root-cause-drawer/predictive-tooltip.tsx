// ── Tooltip do mini-chart preditivo ─────────────────────────────────────────

import { CRITICAL_RISK } from "@/lib/risk-thresholds";
import { cn } from "@/lib/utils";

interface ChartTooltipProps {
  active?: boolean;
  label?: string;
  payload?: Array<{ value: number; payload?: { raw?: number } }>;
}

export function PredictiveTooltip({
  active,
  label,
  payload,
}: ChartTooltipProps) {
  if (!active || !payload?.length) return null;
  const value = payload[0].value;
  const raw = payload[0].payload?.raw;
  return (
    <div className="rounded-md border border-border bg-popover px-2 py-1.5 text-popover-foreground shadow-lg">
      <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </p>
      <p
        className={cn(
          "text-xs font-bold tabular-nums",
          value >= CRITICAL_RISK ? "text-destructive" : "text-foreground",
        )}
      >
        Risco {(value * 100).toFixed(1)}%
      </p>
      {raw !== undefined && (
        <p className="text-xs tabular-nums text-muted-foreground">
          P(falha) bruta {(raw * 100).toFixed(1)}%
        </p>
      )}
    </div>
  );
}
