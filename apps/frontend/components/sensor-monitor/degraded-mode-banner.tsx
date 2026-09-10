// ── Banner modo degradado — RNF-58: extraído de sensor-monitor.tsx ──────────

import { WifiOff } from "lucide-react";

export function DegradedModeBanner({ attempt }: { attempt: number }) {
  return (
    <div
      role="status"
      aria-live="polite"
      className="flex items-center gap-3 rounded-xl border border-amber-400/40 bg-amber-400/8 px-4 py-3 text-amber-700"
    >
      <WifiOff className="h-4 w-4 shrink-0" />
      <div className="min-w-0 flex-1">
        <p className="text-sm font-bold">Modo Degradado</p>
        <p className="text-xs text-amber-600">
          Exibindo últimos dados conhecidos · Reconexão {attempt} em curso…
        </p>
      </div>
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-500" />
    </div>
  );
}
