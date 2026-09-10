"use client";

/**
 * RNF-58: decomposto em `components/history/event-log-table/*` —
 * constants (config de severidade/tipo), helpers (formatTimestamp),
 * SortableHead, EmptyState, EventRow, EventLogPagination. Este arquivo
 * mantém só o estado de ordenação/paginação e a orquestração da tabela.
 * Nenhuma mudança de comportamento/DOM.
 */

import { useCallback, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { HistoryEvent } from "@/lib/history-mock";
import {
  PAGE_SIZE,
  SEVERITY_ORDER,
  type SortColumn,
  type SortState,
} from "./event-log-table/constants";
import { EmptyState } from "./event-log-table/empty-state";
import { EventRow } from "./event-log-table/event-row";
import { EventLogPagination } from "./event-log-table/pagination";
import { SortableHead } from "./event-log-table/sortable-head";

// ── Component ──────────────────────────────────────────────────────────────

interface EventLogTableProps {
  events: HistoryEvent[];
  /** Disparado ao clicar numa linha — opcional; abre o RootCauseDrawer. */
  onSelectEvent?: (event: HistoryEvent) => void;
}

const COL_HEAD =
  "pb-2.5 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground text-left";

export default function EventLogTable({
  events,
  onSelectEvent,
}: EventLogTableProps) {
  const [sort, setSort] = useState<SortState>({
    column: "timestamp",
    direction: "desc",
  });
  const [page, setPage] = useState(1);

  const toggleSort = useCallback((column: SortColumn) => {
    setSort((prev) => ({
      column,
      direction:
        prev.column === column && prev.direction === "desc" ? "asc" : "desc",
    }));
    setPage(1);
  }, []);

  const sorted = useMemo(() => {
    return [...events].sort((a, b) => {
      const dir = sort.direction === "asc" ? 1 : -1;
      switch (sort.column) {
        case "timestamp":
          return (
            dir *
            (new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
          );
        case "equipment":
          return dir * a.equipment.localeCompare(b.equipment);
        case "type":
          return dir * a.type.localeCompare(b.type);
        case "severity":
          return (
            dir * (SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity])
          );
        default:
          return 0;
      }
    });
  }, [events, sort]);

  const totalPages = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages);
  const paginated = sorted.slice(
    (safePage - 1) * PAGE_SIZE,
    safePage * PAGE_SIZE,
  );
  const from = sorted.length === 0 ? 0 : (safePage - 1) * PAGE_SIZE + 1;
  const to = Math.min(safePage * PAGE_SIZE, sorted.length);

  const pageNumbers = Array.from({ length: totalPages }, (_, i) => i + 1)
    .filter((p) => p === 1 || p === totalPages || Math.abs(p - safePage) <= 1)
    .reduce<(number | "…")[]>((acc, p, i, arr) => {
      if (i > 0 && (p as number) - (arr[i - 1] as number) > 1) acc.push("…");
      acc.push(p);
      return acc;
    }, []);

  return (
    <Card className="border-border bg-card">
      <CardHeader className="px-5 pb-3 pt-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-foreground/90">
            Log de Eventos
          </CardTitle>
          <span className="text-[11px] text-muted-foreground">
            {sorted.length === 0
              ? "0 eventos"
              : `${from}–${to} de ${sorted.length} eventos`}
          </span>
        </div>
      </CardHeader>

      <CardContent className="px-5 pb-5">
        {paginated.length === 0 ? (
          <EmptyState />
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-sm" role="table">
                <thead>
                  <tr className="border-b border-border">
                    <SortableHead
                      label="Data / Hora"
                      column="timestamp"
                      sort={sort}
                      onSort={toggleSort}
                    />
                    <SortableHead
                      label="Equipamento"
                      column="equipment"
                      sort={sort}
                      onSort={toggleSort}
                    />
                    <SortableHead
                      label="Tipo"
                      column="type"
                      sort={sort}
                      onSort={toggleSort}
                    />
                    <SortableHead
                      label="Severidade"
                      column="severity"
                      sort={sort}
                      onSort={toggleSort}
                    />
                    <th className={COL_HEAD}>Duração</th>
                    <th className={COL_HEAD}>Descrição</th>
                  </tr>
                </thead>

                <tbody>
                  {paginated.map((event) => (
                    <EventRow
                      key={event.id}
                      event={event}
                      onSelectEvent={onSelectEvent}
                    />
                  ))}
                </tbody>
              </table>
            </div>

            {totalPages > 1 && (
              <EventLogPagination
                safePage={safePage}
                totalPages={totalPages}
                pageNumbers={pageNumbers}
                onPrev={() => setPage((p) => Math.max(1, p - 1))}
                onNext={() => setPage((p) => Math.min(totalPages, p + 1))}
                onPageSelect={setPage}
              />
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
