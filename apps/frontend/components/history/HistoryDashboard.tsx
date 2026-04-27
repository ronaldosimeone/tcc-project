"use client";

import { useState, useMemo } from "react";
import HistoryHeader from "@/components/history/HistoryHeader";
import AlertFrequencyChart from "@/components/history/AlertFrequencyChart";
import HistoryFilters, {
  type FilterState,
  DEFAULT_FILTERS,
} from "@/components/history/HistoryFilters";
import EventLogTable from "@/components/history/EventLogTable";
import { HISTORY_EVENTS, generateDailyAlertCounts } from "@/lib/history-mock";

export default function HistoryDashboard() {
  const [filters, setFilters] = useState<FilterState>(DEFAULT_FILTERS);

  const chartData = useMemo(
    () => generateDailyAlertCounts(HISTORY_EVENTS, 14),
    [],
  );

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
    <div className="flex flex-col gap-6 p-6">
      <HistoryHeader
        totalEvents={HISTORY_EVENTS.length}
        filteredCount={filtered.length}
      />
      <AlertFrequencyChart data={chartData} />
      <HistoryFilters filters={filters} onChange={setFilters} />
      <EventLogTable events={filtered} />
    </div>
  );
}
