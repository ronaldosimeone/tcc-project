// ── Tooltip do mini-chart preditivo ─────────────────────────────────────────

import { cn } from "@/lib/utils";

interface ChartTooltipProps {
  active?: boolean;
  label?: string;
  payload?: Array<{ value: number }>;
  threshold: number;
}

export function PredictiveTooltip({
  active,
  label,
  payload,
  threshold,
}: ChartTooltipProps) {
  if (!active || !payload?.length) return null;
  const value = payload[0].value;
  return (
    <div className="rounded-md border border-border bg-popover px-2 py-1.5 text-popover-foreground shadow-lg">
      <p className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </p>
      <p
        className={cn(
          "text-xs font-bold tabular-nums",
          value >= threshold ? "text-destructive" : "text-foreground",
        )}
      >
        {(value * 100).toFixed(1)}% prob.
      </p>
    </div>
  );
}
