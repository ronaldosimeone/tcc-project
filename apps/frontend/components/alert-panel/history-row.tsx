// ── Linha do histórico — RNF-58: extraído de alert-panel.tsx ────────────────

import { formatClock } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { PredictionHistoryEntry } from "@/hooks/use-prediction-history";
import { RiskBadge } from "./risk-badge";

export function HistoryRow({ entry }: { entry: PredictionHistoryEntry }) {
  const pct = (entry.failure_probability * 100).toFixed(1);
  const time = formatClock(entry.timestamp);

  const probColor =
    entry.riskLevel === "CRÍTICO"
      ? "text-red-400"
      : entry.riskLevel === "ALERTA"
      ? "text-amber-400"
      : "text-muted-foreground";

  return (
    <div
      className="flex items-center gap-2 border-b border-border/40 px-3 py-2 last:border-0"
      data-risk={entry.riskLevel}
    >
      <span className="w-16 shrink-0 font-mono text-[10px] text-muted-foreground/60">
        {time}
      </span>
      <RiskBadge level={entry.riskLevel} />
      <span
        className={cn(
          "ml-auto font-mono text-xs font-semibold tabular-nums",
          probColor,
        )}
      >
        {pct}%
      </span>
    </div>
  );
}
