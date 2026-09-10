// ── Tooltip do mini-chart preditivo — RNF-58: extraído de RootCauseDrawer ───

import { cn } from "@/lib/utils";
import { CRITICAL_THRESHOLD } from "./constants";

interface ChartTooltipProps {
  active?: boolean;
  label?: string;
  payload?: Array<{ value: number }>;
}

export function PredictiveTooltip({
  active,
  label,
  payload,
}: ChartTooltipProps) {
  if (!active || !payload?.length) return null;
  const value = payload[0].value;
  const pct = (value * 100).toFixed(1);
  const isCritical = value >= CRITICAL_THRESHOLD;
  return (
    <div className="rounded-md bg-slate-900 px-2 py-1.5 text-white shadow-lg">
      <p className="text-[10px] font-medium uppercase tracking-wider text-slate-400">
        {label}
      </p>
      <p
        className={cn(
          "text-xs font-bold tabular-nums",
          isCritical ? "text-red-400" : "text-amber-300",
        )}
      >
        {pct}% prob.
      </p>
    </div>
  );
}
