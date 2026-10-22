// ── Tooltip do gráfico de pressão — extraído de main-area-chart.tsx (RNF-58).

import { memo } from "react";
import { formatFixed } from "@/lib/format";

interface AreaTooltipEntry {
  name: string;
  value: number;
  color: string;
}

interface AreaTooltipProps {
  active?: boolean;
  label?: string;
  payload?: AreaTooltipEntry[];
}

const AreaChartTooltip = memo(function AreaChartTooltip({
  active,
  label,
  payload,
}: AreaTooltipProps) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-border bg-card px-3 py-2 shadow-md">
      <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {label}
      </p>
      {payload.map((e) => (
        <div key={e.name} className="flex items-center gap-2 text-xs">
          <span
            className="h-2 w-2 rounded-full"
            style={{ background: e.color }}
          />
          <span className="text-muted-foreground">{e.name}</span>
          {/* Antes: valor cru do sensor (ex.: 8.977999687194824 bar). */}
          <span className="ml-auto font-bold tabular-nums text-foreground">
            {formatFixed(e.value, 2)} bar
          </span>
        </div>
      ))}
    </div>
  );
});

export const AREA_TOOLTIP = <AreaChartTooltip />;
