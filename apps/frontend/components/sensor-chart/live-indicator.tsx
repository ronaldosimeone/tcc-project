// ── Indicador "Live" — RNF-58: extraído de sensor-chart.tsx ─────────────────

export function LiveIndicator() {
  return (
    <span className="flex items-center gap-1.5">
      <span className="relative flex h-2 w-2">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-400 opacity-75" />
        <span className="relative inline-flex h-2 w-2 rounded-full bg-green-500" />
      </span>
      <span className="text-[10px] font-semibold uppercase tracking-wider text-green-400">
        Live
      </span>
    </span>
  );
}
