"use client";

import { useState, useMemo, useCallback } from "react";
import {
  AlertOctagon,
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  ArrowUpDown,
  CheckCircle2,
  ClipboardCheck,
  FileSearch,
  Zap,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { HistoryEvent, Severity, EventType } from "@/lib/history-mock";

// ── Constants ──────────────────────────────────────────────────────────────

const PAGE_SIZE = 10;

const SEVERITY_ORDER: Record<Severity, number> = {
  CRÍTICO: 2,
  ALERTA: 1,
  NORMAL: 0,
};

const SEVERITY_CONFIG: Record<
  Severity,
  { className: string; icon: typeof CheckCircle2 }
> = {
  CRÍTICO: {
    className: "border-red-500/40 bg-red-500/10 text-red-400",
    icon: AlertOctagon,
  },
  ALERTA: {
    className: "border-amber-500/40 bg-amber-500/10 text-amber-400",
    icon: AlertTriangle,
  },
  NORMAL: {
    className: "border-green-500/40 bg-green-500/10 text-green-400",
    icon: CheckCircle2,
  },
};

const EVENT_TYPE_CONFIG: Record<
  EventType,
  { className: string; icon: typeof CheckCircle2 }
> = {
  Falha: { className: "text-red-400", icon: AlertOctagon },
  Alerta: { className: "text-amber-400", icon: Zap },
  Diagnóstico: { className: "text-green-400", icon: ClipboardCheck },
};

// ── Types ──────────────────────────────────────────────────────────────────

type SortColumn = "timestamp" | "equipment" | "type" | "severity";
type SortDir = "asc" | "desc";

interface SortState {
  column: SortColumn;
  direction: SortDir;
}

// ── Helpers ────────────────────────────────────────────────────────────────

function formatTimestamp(iso: string): string {
  return new Intl.DateTimeFormat("pt-BR", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  })
    .format(new Date(iso))
    .replace(".", "");
}

// ── Sub-components ─────────────────────────────────────────────────────────

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

function SortableHead({
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
  return (
    <th
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

function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-3 py-16 text-center">
      <FileSearch className="h-10 w-10 text-muted-foreground/25" />
      <p className="text-sm font-medium text-muted-foreground">
        Nenhum evento encontrado
      </p>
      <p className="text-xs text-muted-foreground/60">
        Ajuste os filtros para ampliar a busca
      </p>
    </div>
  );
}

// ── Component ──────────────────────────────────────────────────────────────

interface EventLogTableProps {
  events: HistoryEvent[];
}

export default function EventLogTable({ events }: EventLogTableProps) {
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

  const COL_HEAD =
    "pb-2.5 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground text-left";

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
                  {paginated.map((event) => {
                    const sevCfg = SEVERITY_CONFIG[event.severity];
                    const typeCfg = EVENT_TYPE_CONFIG[event.type];
                    const SevIcon = sevCfg.icon;
                    const TypeIcon = typeCfg.icon;

                    return (
                      <tr
                        key={event.id}
                        className="group/row border-b border-border/40 last:border-0 transition-colors duration-150 hover:bg-muted/20"
                      >
                        {/* Timestamp */}
                        <td className="py-3 pr-5">
                          <span className="whitespace-nowrap font-mono text-xs tabular-nums text-muted-foreground">
                            {formatTimestamp(event.timestamp)}
                          </span>
                        </td>

                        {/* Equipment */}
                        <td className="py-3 pr-5">
                          <span className="font-mono text-xs font-medium text-foreground">
                            {event.equipment}
                          </span>
                        </td>

                        {/* Type */}
                        <td className="py-3 pr-5">
                          <div
                            className={cn(
                              "flex items-center gap-1.5 text-xs font-medium",
                              typeCfg.className,
                            )}
                          >
                            <TypeIcon className="h-3.5 w-3.5 shrink-0" />
                            {event.type}
                          </div>
                        </td>

                        {/* Severity */}
                        <td className="py-3 pr-5">
                          <Badge
                            variant="outline"
                            className={cn(
                              "gap-1 text-[11px] font-semibold",
                              sevCfg.className,
                            )}
                          >
                            <SevIcon className="h-3 w-3" />
                            {event.severity}
                          </Badge>
                        </td>

                        {/* Duration */}
                        <td className="py-3 pr-5">
                          <span className="whitespace-nowrap font-mono text-xs tabular-nums text-muted-foreground">
                            {event.duration}
                          </span>
                        </td>

                        {/* Description */}
                        <td className="max-w-sm py-3">
                          <span className="line-clamp-2 text-xs leading-relaxed text-foreground/70 transition-colors group-hover/row:text-foreground/90">
                            {event.description}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="mt-5 flex items-center justify-between border-t border-border pt-4">
                <Button
                  variant="outline"
                  size="sm"
                  className="h-8 text-xs"
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={safePage === 1}
                >
                  Anterior
                </Button>

                <div className="flex items-center gap-1">
                  {pageNumbers.map((p, i) =>
                    p === "…" ? (
                      <span
                        key={`ellipsis-${i}`}
                        className="px-1 text-xs text-muted-foreground"
                      >
                        …
                      </span>
                    ) : (
                      <button
                        key={p}
                        onClick={() => setPage(p as number)}
                        className={cn(
                          "flex h-7 w-7 items-center justify-center rounded-md text-xs font-medium transition-colors",
                          safePage === p
                            ? "bg-primary/10 text-primary ring-1 ring-inset ring-primary/20"
                            : "text-muted-foreground hover:bg-accent hover:text-foreground",
                        )}
                      >
                        {p}
                      </button>
                    ),
                  )}
                </div>

                <Button
                  variant="outline"
                  size="sm"
                  className="h-8 text-xs"
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={safePage === totalPages}
                >
                  Próxima
                </Button>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
