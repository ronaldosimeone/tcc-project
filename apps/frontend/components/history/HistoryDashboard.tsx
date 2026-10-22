"use client";

/**
 * Dashboard de Histórico — dados reais (sem mock).
 *
 * Fonte: `GET /v1/events` (episódios de alarme RN-03 persistidos pelo
 * pipeline de inferência de cada ativo da frota) + `GET /v1/kpis` (MTBF,
 * MTTR, disponibilidade e precisão de alarmes — só falhas confirmadas pelo
 * técnico na OS contam).
 *
 * Estrutura vertical:
 *   1. Sessão analítica — Header + KPIs + Chart + Heatmap.
 *      Os filtros ficam logo abaixo do header: KPIs e gráficos também
 *      usam os eventos filtrados.
 *   2. Tabela de eventos.
 *   3. RootCauseDrawer — causa raiz congelada, leituras reais e a OS.
 *
 * O período vem do servidor (refetch); severidade/equipamento/busca são
 * filtros locais sobre os eventos já carregados.
 */

import { useMemo, useState } from "react";

import AlertFrequencyChart from "@/components/history/AlertFrequencyChart";
import EventHeatmap from "@/components/history/EventHeatmap";
import EventLogTable from "@/components/history/EventLogTable";
import HistoryFilters, {
  DEFAULT_FILTERS,
  chartDays,
  periodLabel,
  type FilterState,
} from "@/components/history/HistoryFilters";
import HistoryHeader from "@/components/history/HistoryHeader";
import HistoryKPIs from "@/components/history/HistoryKPIs";
import RootCauseDrawer from "@/components/history/RootCauseDrawer";
import TiposEvento from "@/components/history/TiposEvento";
import TopEquipamentos from "@/components/history/TopEquipamentos";
import {
  HistoryEmptyState,
  HistoryErrorState,
  HistorySkeleton,
} from "@/components/history/history-states";
import { useHistoryData } from "@/hooks/use-history-data";
import type { HistoryEvent } from "@/lib/history-types";

export default function HistoryDashboard() {
  const [filters, setFilters] = useState<FilterState>(DEFAULT_FILTERS);
  const [selectedEvent, setSelectedEvent] = useState<HistoryEvent | null>(null);
  const { assets, events, kpis, state, error, reload } = useHistoryData(
    filters.period,
  );

  const filtered = useMemo(() => {
    const q = filters.search.toLowerCase();
    return events.filter((event) => {
      if (filters.severity !== "all" && event.severity !== filters.severity)
        return false;
      if (filters.equipment !== "all" && event.equipment !== filters.equipment)
        return false;
      if (
        q &&
        !event.description.toLowerCase().includes(q) &&
        !event.equipment.toLowerCase().includes(q)
      ) {
        return false;
      }
      return true;
    });
  }, [events, filters]);

  return (
    <>
      <div className="flex flex-col gap-6 px-4 py-6 lg:px-8">
        <section className="flex min-h-[calc(100dvh-3rem)] flex-col gap-6">
          <HistoryHeader
            totalEvents={events.length}
            filteredCount={filtered.length}
            events={filtered}
            periodLabel={periodLabel(filters.period)}
          />

          {/* Filtros antes dos KPIs e gráficos: todos usam os eventos
              filtrados (antes ficavam abaixo deles, longe do que mudam). */}
          <HistoryFilters
            filters={filters}
            onChange={setFilters}
            equipments={
              assets.length > 0
                ? assets
                : [...new Set(events.map((e) => e.equipment))].sort()
            }
          />

          {state === "loading" && <HistorySkeleton />}
          {state === "error" && (
            <HistoryErrorState message={error ?? ""} onRetry={reload} />
          )}
          {state === "ready" && events.length === 0 && <HistoryEmptyState />}

          {state === "ready" && events.length > 0 && (
            <>
              <HistoryKPIs events={filtered} kpis={kpis} />

              <div className="grid flex-1 grid-cols-1 gap-6 lg:grid-cols-12">
                <div className="flex min-h-0 flex-col lg:col-span-7">
                  <AlertFrequencyChart
                    events={filtered}
                    days={chartDays(filters.period)}
                  />
                </div>

                <div className="flex min-h-0 flex-col gap-4 lg:col-span-5">
                  <div className="min-h-0 flex-1">
                    <EventHeatmap events={filtered} />
                  </div>
                  <div className="grid min-h-0 flex-1 grid-cols-2 gap-4">
                    <TopEquipamentos events={filtered} roster={assets} />
                    <TiposEvento events={filtered} />
                  </div>
                </div>
              </div>
            </>
          )}
        </section>

        <EventLogTable events={filtered} onSelectEvent={setSelectedEvent} />
      </div>

      <RootCauseDrawer
        event={selectedEvent}
        onClose={() => setSelectedEvent(null)}
        onChanged={reload}
      />
    </>
  );
}
