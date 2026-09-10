// ── Tooltip — RNF-58: extraído de AssetRadarChart ───────────────────────────

import { memo } from "react";

interface TooltipEntry {
  name: string;
  value: number;
  color: string;
}

interface RadarTooltipProps {
  active?: boolean;
  label?: string;
  payload?: TooltipEntry[];
}

export const ChartTooltip = memo(function ChartTooltip({
  active,
  label,
  payload,
}: RadarTooltipProps) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-border/80 bg-popover/90 px-3 py-2 shadow-md backdrop-blur-md">
      <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {label}
      </p>
      {payload.map((entry) => (
        <div key={entry.name} className="flex items-center gap-2 text-xs">
          <span
            className="inline-block h-2 w-2 rounded-full"
            style={{ background: entry.color }}
          />
          <span className="text-muted-foreground">{entry.name}</span>
          <span className="ml-auto font-bold tabular-nums text-foreground">
            {entry.value.toFixed(0)}
            <span className="font-normal text-muted-foreground">/100</span>
          </span>
        </div>
      ))}
    </div>
  );
});

// Static JSX element — Recharts clones it with runtime tooltip props.
export const TOOLTIP_CONTENT = <ChartTooltip />;
