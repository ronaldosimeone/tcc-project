"use client";

import { Clock, FileDown, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";

interface HistoryHeaderProps {
  totalEvents: number;
  filteredCount: number;
}

export default function HistoryHeader({
  totalEvents,
  filteredCount,
}: HistoryHeaderProps) {
  const isFiltered = filteredCount !== totalEvents;

  return (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div>
        <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
          PredictIQ · MetroPT-3
        </p>
        <h1 className="mt-0.5 text-xl font-bold tracking-tight text-foreground">
          Histórico & Relatórios
        </h1>
        <p className="mt-1 flex items-center gap-1.5 text-sm text-muted-foreground">
          <Clock className="h-3.5 w-3.5 shrink-0" />
          Últimos 30 dias ·{" "}
          {isFiltered ? (
            <>
              <span className="font-semibold text-foreground">
                {filteredCount}
              </span>{" "}
              de{" "}
              <span className="font-semibold text-foreground">
                {totalEvents}
              </span>{" "}
              eventos
            </>
          ) : (
            <>
              <span className="font-semibold text-foreground">
                {totalEvents}
              </span>{" "}
              eventos registrados
            </>
          )}
        </p>
      </div>

      <div className="flex items-center gap-2">
        <Button variant="outline" size="sm" className="h-9 gap-2 text-xs">
          <FileDown className="h-3.5 w-3.5" />
          Exportar CSV
        </Button>
        <Button
          variant="outline"
          size="sm"
          className="h-9 gap-2 border-primary/30 bg-primary/5 text-xs text-primary hover:bg-primary/10 hover:text-primary"
        >
          <FileText className="h-3.5 w-3.5" />
          Exportar PDF
        </Button>
      </div>
    </div>
  );
}
