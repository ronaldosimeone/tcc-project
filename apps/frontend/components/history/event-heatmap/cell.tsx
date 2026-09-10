// ── Célula individual (+ tooltip CSS-only) — RNF-58: extraído de
// EventHeatmap ─────────────────────────────────────────────────────────────

import { cn } from "@/lib/utils";
import type { CellData, Day, Shift } from "./helpers";
import { intensityClass, intensityLabel } from "./helpers";

interface CellProps {
  cell: CellData;
  shift: Shift;
  day: Day;
}

export function Cell({ cell, shift, day }: CellProps) {
  const { total: count, critical, alert } = cell;
  const label = `${day} ${shift}: ${count} ${
    count === 1 ? "evento" : "eventos"
  } (${intensityLabel(count)})`;

  return (
    // `group` + `relative` permite uma tooltip CSS-only via `group-hover`.
    // Sem libs externas — funciona em qualquer browser moderno.
    <div className="group relative">
      <div
        role="img"
        aria-label={label}
        className={cn(
          "h-4 w-full rounded-sm transition-colors",
          intensityClass(count),
        )}
      />

      {/* Tooltip: aparece acima da célula no hover, segue o cursor.
          `pointer-events-none` evita que ele intercepte hovers vizinhos.
          Centralizada via `left-1/2 -translate-x-1/2` para alinhamento óptico. */}
      <div
        role="tooltip"
        className={cn(
          "pointer-events-none absolute bottom-full left-1/2 z-50 mb-2 -translate-x-1/2",
          "hidden min-w-[10rem] rounded-md bg-slate-900 px-2 py-2 text-white shadow-lg",
          "group-hover:block",
        )}
      >
        <p className="text-[10px] font-medium uppercase tracking-wider text-slate-400">
          {day} · {shift}
        </p>
        <p className="mt-0.5 text-sm font-bold tabular-nums">
          {count} {count === 1 ? "ocorrência" : "ocorrências"}
        </p>
        {count > 0 && (
          <div className="mt-1 flex items-center gap-2 text-[11px] text-slate-200">
            <span className="inline-flex items-center gap-1">
              <span className="h-1.5 w-1.5 rounded-full bg-red-500" />
              <span className="tabular-nums">{critical} críticas</span>
            </span>
            <span className="text-slate-600">|</span>
            <span className="inline-flex items-center gap-1">
              <span className="h-1.5 w-1.5 rounded-full bg-amber-500" />
              <span className="tabular-nums">{alert} alertas</span>
            </span>
          </div>
        )}
        {/* Seta apontando para a célula */}
        <span
          aria-hidden="true"
          className="absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent border-t-slate-900"
        />
      </div>
    </div>
  );
}
