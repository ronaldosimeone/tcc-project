// ── Legenda das séries — RNF-58: extraído de sensor-chart.tsx ───────────────

import { memo } from "react";
import { LEGEND_ITEMS, LINE_COLORS } from "./constants";

export const ChartLegend = memo(function ChartLegend() {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1">
      {LEGEND_ITEMS.map(({ key, label }) => (
        <div key={key} className="flex items-center gap-1.5">
          <span
            className="inline-block h-2 w-4 rounded-full"
            style={{ background: LINE_COLORS[key] }}
          />
          <span className="text-[10px] text-muted-foreground">{label}</span>
        </div>
      ))}
    </div>
  );
});
