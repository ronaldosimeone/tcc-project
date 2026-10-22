"use client";

import { useState } from "react";
import { Clock, FileDown, FileText, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { HistoryEvent } from "@/lib/history-types";

type ExportState = "csv" | "pdf" | null;

interface HistoryHeaderProps {
  totalEvents: number;
  filteredCount: number;
  /** Rótulo do período do filtro (ex.: "Últimos 30 dias"). */
  periodLabel: string;
  /** Eventos exportados no CSV (os do recorte filtrado). */
  events: HistoryEvent[];
}

function downloadCsv(events: HistoryEvent[]) {
  const SEP = ";";
  const BOM = "﻿"; // UTF-8 BOM — garante leitura correta no Excel

  const header = [
    "ID",
    "Timestamp",
    "Equipamento",
    "Tipo",
    "Severidade",
    "Duração",
    "Pico P(falha)",
    "Status OS",
    "Causa confirmada",
  ].join(SEP);

  const rows = events.map((e) =>
    [
      e.id,
      e.timestamp,
      e.equipment,
      e.type,
      e.severity,
      e.duration,
      e.peakProbability.toFixed(3),
      e.workOrderStatus ?? "",
      e.isFalseAlarm === true ? "falso alarme" : e.confirmedCause ?? "",
    ].join(SEP),
  );

  const csv = BOM + [header, ...rows].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);

  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "historico_eventos_predictiq.csv";
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

export default function HistoryHeader({
  totalEvents,
  filteredCount,
  events,
  periodLabel,
}: HistoryHeaderProps) {
  const [isExporting, setIsExporting] = useState<ExportState>(null);
  const isFiltered = filteredCount !== totalEvents;

  const handleExport = (type: "csv" | "pdf") => {
    if (isExporting) return;
    setIsExporting(type);

    if (type === "csv") {
      downloadCsv(events);
      setTimeout(() => setIsExporting(null), 1500);
    } else {
      setTimeout(() => {
        window.print();
        setIsExporting(null);
      }, 400);
    }
  };

  return (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div>
        <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          PredictIQ · MetroPT-3
        </p>
        <h1 className="mt-0.5 text-xl font-bold tracking-tight text-foreground">
          Histórico & Relatórios
        </h1>
        <p className="mt-1 flex items-center gap-1.5 text-sm text-muted-foreground">
          <Clock className="h-3.5 w-3.5 shrink-0" />
          {periodLabel} ·{" "}
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
        <Button
          variant="outline"
          size="sm"
          className="h-9 gap-2 text-xs"
          disabled={isExporting !== null}
          onClick={() => handleExport("csv")}
        >
          {isExporting === "csv" ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <FileDown className="h-3.5 w-3.5" />
          )}
          {isExporting === "csv" ? "Exportando…" : "Exportar CSV"}
        </Button>

        <Button
          variant="outline"
          size="sm"
          className="h-9 gap-2 border-primary/30 bg-primary/5 text-xs text-primary hover:bg-primary/10 hover:text-primary"
          disabled={isExporting !== null}
          onClick={() => handleExport("pdf")}
        >
          {isExporting === "pdf" ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <FileText className="h-3.5 w-3.5" />
          )}
          {isExporting === "pdf" ? "Exportando…" : "Exportar PDF"}
        </Button>
      </div>
    </div>
  );
}
