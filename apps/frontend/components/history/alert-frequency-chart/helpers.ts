// ── Bucketização diária — RNF-58: extraído de AlertFrequencyChart ───────────

import type { HistoryEvent } from "@/lib/history-mock";

// Literais slate — evita tokens hsl(var(--...)) que renderizam transparente
// pelo problema conhecido de mistura oklch/hsl no tema.
export const AXIS_TICK = { fontSize: 11, fill: "#94a3b8" } as const;
export const GRID_STROKE = "#e2e8f0";

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
export function bucketEventsByDay(events: HistoryEvent[]): DayBucket[] {
  if (events.length === 0) return [];

  // Encontra o dia "mais recente" no recorte para fechar a janela de 14 dias.
  const latestMs = events.reduce(
    (max, e) => Math.max(max, new Date(e.timestamp).getTime()),
    0,
  );
  const latest = new Date(latestMs);
  latest.setUTCHours(0, 0, 0, 0);

  const buckets = new Map<string, DayBucket>();
  for (let i = 13; i >= 0; i--) {
    const d = new Date(latest);
    d.setUTCDate(d.getUTCDate() - i);
    const iso = d.toISOString().slice(0, 10);
    buckets.set(iso, {
      iso,
      date: `${String(d.getUTCDate()).padStart(2, "0")}/${String(
        d.getUTCMonth() + 1,
      ).padStart(2, "0")}`,
      alerta: 0,
      critico: 0,
      normal: 0,
    });
  }

  for (const e of events) {
    const iso = e.timestamp.slice(0, 10);
    const bucket = buckets.get(iso);
    if (!bucket) continue; // fora da janela de 14 dias
    if (e.severity === "CRÍTICO") bucket.critico++;
    else if (e.severity === "ALERTA") bucket.alerta++;
    else bucket.normal++;
  }

  return Array.from(buckets.values()).sort((a, b) => (a.iso < b.iso ? -1 : 1));
}
