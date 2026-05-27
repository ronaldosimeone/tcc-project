"use client";

/**
 * Dashboard de Histórico.
 *
 * Estrutura vertical:
 *   1. Sessão analítica — Header + KPIs + Chart + Heatmap. Ocupa exatamente
 *      a altura disponível do viewport (≈ 100dvh) via flex-col + min-height
 *      no container externo. Chart e Heatmap dividem o espaço residual em
 *      duas colunas no desktop (flex-1 cada).
 *   2. Filtros + tabela de eventos — fluem abaixo, acessíveis por scroll.
 *   3. RootCauseDrawer — overlay lateral aberto pelo clique em qualquer row.
 */

import { useState, useMemo } from "react";

import HistoryHeader from "@/components/history/HistoryHeader";
import HistoryKPIs from "@/components/history/HistoryKPIs";
import AlertFrequencyChart from "@/components/history/AlertFrequencyChart";
import EventHeatmap from "@/components/history/EventHeatmap";
import TopEquipamentos from "@/components/history/TopEquipamentos";
import TiposEvento from "@/components/history/TiposEvento";
import HistoryFilters, {
  type FilterState,
  DEFAULT_FILTERS,
} from "@/components/history/HistoryFilters";
import EventLogTable from "@/components/history/EventLogTable";
import RootCauseDrawer from "@/components/history/RootCauseDrawer";
import { HISTORY_EVENTS, type HistoryEvent } from "@/lib/history-mock";

export default function HistoryDashboard() {
  const [filters, setFilters] = useState<FilterState>(DEFAULT_FILTERS);
  const [selectedEvent, setSelectedEvent] = useState<HistoryEvent | null>(null);

  const filtered = useMemo(() => {
    const periodDays =
      filters.period === "all" ? Infinity : parseInt(filters.period, 10);
    const cutoff = new Date("2026-04-27T23:59:59.000Z");

    if (isFinite(periodDays)) {
      cutoff.setUTCDate(cutoff.getUTCDate() - periodDays);
      cutoff.setUTCHours(0, 0, 0, 0);
    } else {
      cutoff.setUTCFullYear(2000);
    }

    return HISTORY_EVENTS.filter((event) => {
      if (new Date(event.timestamp) < cutoff) return false;
      if (filters.severity !== "all" && event.severity !== filters.severity)
        return false;
      if (filters.equipment !== "all" && event.equipment !== filters.equipment)
        return false;
      if (filters.search) {
        const q = filters.search.toLowerCase();
        if (
          !event.description.toLowerCase().includes(q) &&
          !event.equipment.toLowerCase().includes(q)
        ) {
          return false;
        }
      }
      return true;
    });
  }, [filters]);

  return (
    <>
      <div className="flex flex-col gap-6 px-4 py-6 lg:px-8">
        {/* ── Sessão analítica: ocupa min-h 100dvh menos o padding vertical
            (py-6 = 1.5rem topo + 1.5rem base = 3rem). Tudo aqui dentro flui
            em flex-col; o grid intermediário (chart + heatmap) usa `flex-1`
            para absorver o espaço residual e evitar área branca. ── */}
        <section className="flex min-h-[calc(100dvh-3rem)] flex-col gap-6">
          <HistoryHeader
            totalEvents={HISTORY_EVENTS.length}
            filteredCount={filtered.length}
          />

          {/* Todos os 5 componentes analíticos consomem `filtered` como fonte
              única — derivam suas métricas via useMemo internamente. */}
          <HistoryKPIs events={filtered} />

          <div className="grid flex-1 grid-cols-1 gap-6 lg:grid-cols-12">
            <div className="flex min-h-0 flex-col lg:col-span-7">
              <AlertFrequencyChart events={filtered} />
            </div>

            <div className="flex min-h-0 flex-col gap-4 lg:col-span-5">
              <div className="min-h-0 flex-1">
                <EventHeatmap events={filtered} />
              </div>
              <div className="grid min-h-0 flex-1 grid-cols-2 gap-4">
                <TopEquipamentos events={filtered} />
                <TiposEvento events={filtered} />
              </div>
            </div>
          </div>
        </section>

        {/* ── Filtros + tabela de eventos: fluem abaixo da sessão analítica. ── */}
        <HistoryFilters filters={filters} onChange={setFilters} />
        <EventLogTable events={filtered} onSelectEvent={setSelectedEvent} />
      </div>

      {/* ── Drawer lateral — controlado por selectedEvent. ── */}
      <RootCauseDrawer
        event={selectedEvent}
        onClose={() => setSelectedEvent(null)}
      />
    </>
  );
}
