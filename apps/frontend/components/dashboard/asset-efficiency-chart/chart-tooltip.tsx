// ── Tooltip — RNF-58: extraído de AssetEfficiencyChart ──────────────────────

import { memo } from "react";

interface TooltipEntry {
  name: string;
  value: number;
  color: string;
  dataKey: string;
}

interface TooltipProps {
  active?: boolean;
  label?: string;
  payload?: TooltipEntry[];
}

export const ChartTooltip = memo(function ChartTooltip({
  active,
  label,
  payload,
}: TooltipProps) {
  if (!active || !payload?.length) return null;

  const carga = payload.find((p) => p.dataKey === "carga")?.value ?? 0;
  const ocioso = payload.find((p) => p.dataKey === "ocioso")?.value ?? 0;
  const total = carga + ocioso;
  const efficiency = total > 0 ? ((carga / total) * 100).toFixed(0) : "0";

  return (
    <div className="rounded-lg border border-border/80 bg-popover/90 px-3 py-2 shadow-md backdrop-blur-md">
      <div className="mb-2 flex items-center justify-between gap-4">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          {label}
        </p>
        <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] font-bold tabular-nums text-foreground">
          {efficiency}% efic.
        </span>
      </div>
      {payload.map((entry) => (
        <div key={entry.name} className="flex items-center gap-2 text-xs">
          <span
            className="inline-block h-2 w-2 rounded-full"
            style={{ background: entry.color }}
          />
          <span className="text-muted-foreground">{entry.name}</span>
          <span className="ml-auto font-bold tabular-nums text-foreground">
            {entry.value}h
          </span>
        </div>
      ))}
      <div className="mt-1.5 border-t border-border/60 pt-1.5">
        <div className="flex items-center justify-between text-xs">
          <span className="text-muted-foreground">Total</span>
          <span className="font-bold tabular-nums text-foreground">
            {total}h
          </span>
        </div>
      </div>
    </div>
  );
});

export const TOOLTIP_CONTENT = <ChartTooltip />;
