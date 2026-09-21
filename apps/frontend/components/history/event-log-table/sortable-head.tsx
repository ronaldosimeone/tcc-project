// ── Cabeçalho ordenável — RNF-58: extraído de EventLogTable ─────────────────

import { ArrowDown, ArrowUp, ArrowUpDown } from "lucide-react";
import { cn } from "@/lib/utils";
import type { SortColumn, SortState } from "./constants";

function SortIndicator({
  column,
  sort,
}: {
  column: SortColumn;
  sort: SortState;
}) {
  if (sort.column !== column)
    return <ArrowUpDown className="h-3 w-3 opacity-25" />;
  return sort.direction === "asc" ? (
    <ArrowUp className="h-3 w-3 text-primary" />
  ) : (
    <ArrowDown className="h-3 w-3 text-primary" />
  );
}

export function SortableHead({
  label,
  column,
  sort,
  onSort,
  align = "left",
}: {
  label: string;
  column: SortColumn;
  sort: SortState;
  onSort: (col: SortColumn) => void;
  align?: "left" | "right";
}) {
  const ariaSort: "ascending" | "descending" | "none" =
    sort.column !== column
      ? "none"
      : sort.direction === "asc"
      ? "ascending"
      : "descending";

  return (
    <th
      scope="col"
      aria-sort={ariaSort}
      className={cn(
        "pb-2.5 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground",
        align === "right" ? "text-right" : "text-left",
      )}
    >
      <button
        onClick={() => onSort(column)}
        className="flex items-center gap-1 transition-colors hover:text-foreground"
      >
        {label}
        <SortIndicator column={column} sort={sort} />
      </button>
    </th>
  );
}
