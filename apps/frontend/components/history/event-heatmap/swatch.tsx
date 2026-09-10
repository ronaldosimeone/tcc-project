// ── Swatch da legenda — RNF-58: extraído de EventHeatmap ────────────────────

import { cn } from "@/lib/utils";

interface SwatchProps {
  className: string;
  label: string;
}

export function Swatch({ className, label }: SwatchProps) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className={cn("h-3 w-3 rounded-sm border border-black/5", className)}
        aria-hidden="true"
      />
      <span className="font-mono">{label}</span>
    </span>
  );
}
