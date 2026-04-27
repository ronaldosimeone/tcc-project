"use client";

import { Search, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { Severity } from "@/lib/history-mock";

const EQUIPMENTS = [
  "APU-Trem-042",
  "APU-Trem-015",
  "APU-Trem-023",
  "APU-Trem-031",
  "APU-Trem-055",
];

const PERIODS = [
  { label: "Últimos 7 dias", value: "7" },
  { label: "Últimos 14 dias", value: "14" },
  { label: "Últimos 30 dias", value: "30" },
  { label: "Todo o período", value: "all" },
];

export interface FilterState {
  search: string;
  period: string;
  severity: Severity | "all";
  equipment: string;
}

export const DEFAULT_FILTERS: FilterState = {
  search: "",
  period: "30",
  severity: "all",
  equipment: "all",
};

interface HistoryFiltersProps {
  filters: FilterState;
  onChange: (filters: FilterState) => void;
}

export default function HistoryFilters({
  filters,
  onChange,
}: HistoryFiltersProps) {
  const selectBase = cn(
    "h-9 cursor-pointer appearance-none rounded-md border border-border bg-card",
    "pl-3 pr-8 text-sm text-foreground",
    "focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/40",
    "transition-colors duration-150",
  );

  const update = <K extends keyof FilterState>(key: K, value: FilterState[K]) =>
    onChange({ ...filters, [key]: value });

  const hasActive =
    filters.search !== "" ||
    filters.period !== "30" ||
    filters.severity !== "all" ||
    filters.equipment !== "all";

  return (
    <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-card p-3">
      {/* Search */}
      <div className="relative min-w-52 flex-1">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
        <input
          type="text"
          placeholder="Buscar eventos ou equipamentos…"
          value={filters.search}
          onChange={(e) => update("search", e.target.value)}
          className={cn(
            "h-9 w-full rounded-md border border-border bg-background",
            "pl-8 pr-3 text-sm placeholder:text-muted-foreground/50",
            "focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/40",
            "transition-colors duration-150",
          )}
        />
      </div>

      {/* Period */}
      <div className="relative">
        <select
          value={filters.period}
          onChange={(e) => update("period", e.target.value)}
          className={selectBase}
        >
          {PERIODS.map((p) => (
            <option key={p.value} value={p.value}>
              {p.label}
            </option>
          ))}
        </select>
        <span className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-xs text-muted-foreground">
          ▾
        </span>
      </div>

      {/* Severity */}
      <div className="relative">
        <select
          value={filters.severity}
          onChange={(e) =>
            update("severity", e.target.value as Severity | "all")
          }
          className={selectBase}
        >
          <option value="all">Todas as severidades</option>
          <option value="CRÍTICO">Crítico</option>
          <option value="ALERTA">Alerta</option>
          <option value="NORMAL">Normal</option>
        </select>
        <span className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-xs text-muted-foreground">
          ▾
        </span>
      </div>

      {/* Equipment */}
      <div className="relative">
        <select
          value={filters.equipment}
          onChange={(e) => update("equipment", e.target.value)}
          className={cn(selectBase, "font-mono")}
        >
          <option value="all" className="font-sans">
            Todos os equipamentos
          </option>
          {EQUIPMENTS.map((eq) => (
            <option key={eq} value={eq}>
              {eq}
            </option>
          ))}
        </select>
        <span className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-xs text-muted-foreground">
          ▾
        </span>
      </div>

      {/* Reset — only visible when filters are active */}
      {hasActive && (
        <Button
          variant="ghost"
          size="sm"
          className="h-9 gap-1.5 text-xs text-muted-foreground hover:text-foreground"
          onClick={() => onChange(DEFAULT_FILTERS)}
        >
          <RotateCcw className="h-3.5 w-3.5" />
          Limpar
        </Button>
      )}
    </div>
  );
}
