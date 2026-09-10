// ── Linha de turno (7 células) — RNF-58: extraído de EventHeatmap ───────────

import { Cell } from "./cell";
import { DAYS, type CellData, type Shift } from "./helpers";

interface RowGroupProps {
  shift: Shift;
  cells: ReadonlyArray<CellData>;
}

export function RowGroup({ shift, cells }: RowGroupProps) {
  return (
    <>
      <span className="self-center pr-2 font-mono text-[10px] uppercase tracking-wider text-slate-400">
        {shift}
      </span>
      {DAYS.map((day, colIdx) => {
        const cell = cells[colIdx] ?? { total: 0, critical: 0, alert: 0 };
        return (
          <Cell key={`${shift}-${day}`} cell={cell} shift={shift} day={day} />
        );
      })}
    </>
  );
}
