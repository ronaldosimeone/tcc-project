// ── Bucketização diária — RNF-58: extraído de AlertFrequencyChart ───────────

import dayjs from "dayjs";

import { formatDayMonth, localDayKey } from "@/lib/format";
import type { HistoryEvent } from "@/lib/history-types";

// Literais slate — evita tokens hsl(var(--...)) que renderizam transparente
// pelo problema conhecido de mistura oklch/hsl no tema.
export const AXIS_TICK = { fontSize: 12, fill: "#94a3b8" } as const;
export const GRID_STROKE = "var(--border)";

export interface DayBucket {
  /** Rótulo "DD/MM" exibido no eixo X. */
  date: string;
  /** Chave ISO "YYYY-MM-DD" usada para ordenação. */
  iso: string;
  alerta: number;
  critico: number;
  normal: number;
}

/**
 * Agrupa o array de eventos por dia ISO. Garante presença de todos os dias
 * num intervalo de 14 dias terminando no evento mais recente — assim o eixo
 * X mantém continuidade visual mesmo quando um dia não tem ocorrências.
 */
export function bucketEventsByDay(
  events: HistoryEvent[],
  days = 14,
): DayBucket[] {
  if (events.length === 0) return [];

  // Encontra o dia "mais recente" no recorte para fechar a janela de 14 dias.
  // Dias no fuso do operador (antes UTC: eventos das 21h–24h no Brasil
  // caíam no dia seguinte).
  const latestMs = events.reduce(
    (max, e) => Math.max(max, dayjs(e.timestamp).valueOf()),
    0,
  );
  const latest = dayjs(latestMs).startOf("day");

  const buckets = new Map<string, DayBucket>();
  for (let i = days - 1; i >= 0; i--) {
    const d = latest.subtract(i, "day");
    const iso = localDayKey(d.valueOf());
    buckets.set(iso, {
      iso,
      date: formatDayMonth(d.valueOf()),
      alerta: 0,
      critico: 0,
      normal: 0,
    });
  }

  for (const e of events) {
    const iso = localDayKey(e.timestamp);
    const bucket = buckets.get(iso);
    if (!bucket) continue; // fora da janela
    if (e.severity === "CRÍTICO") bucket.critico++;
    else if (e.severity === "ALERTA") bucket.alerta++;
    else bucket.normal++;
  }

  return Array.from(buckets.values()).sort((a, b) => (a.iso < b.iso ? -1 : 1));
}
