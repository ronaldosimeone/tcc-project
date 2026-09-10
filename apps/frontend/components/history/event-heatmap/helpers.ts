// ── Configuração dos eixos + helpers — RNF-58: extraído de EventHeatmap ─────

import type { HistoryEvent } from "@/lib/history-mock";

export const SHIFTS = ["00h", "06h", "12h", "18h"] as const;
export const DAYS = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"] as const;

export type Shift = (typeof SHIFTS)[number];
export type Day = (typeof DAYS)[number];

/**
 * Classe Tailwind tonal por densidade. Cores sólidas + sombra interna sutil
 * (`shadow-inner ring-1 ring-inset ring-black/5`) dão sensação de relevo
 * sem comprometer a leitura imediata da intensidade.
 */
export function intensityClass(count: number): string {
  if (count === 0) return "bg-slate-100 ring-1 ring-inset ring-black/[0.03]";
  if (count <= 2)
    return "bg-emerald-500 shadow-inner shadow-emerald-700/30 ring-1 ring-inset ring-emerald-600/30";
  if (count <= 5)
    return "bg-amber-500 shadow-inner shadow-amber-700/30 ring-1 ring-inset ring-amber-600/30";
  return "bg-red-500 shadow-inner shadow-red-700/30 ring-1 ring-inset ring-red-600/40 animate-pulse";
}

/** Label semântico para acessibilidade (`aria-label` + `title`). */
export function intensityLabel(count: number): string {
  if (count === 0) return "sem ocorrências";
  if (count <= 2) return "operação normal";
  if (count <= 5) return "alerta";
  return "crítico";
}

/**
 * Bucketiza eventos em matriz 4 (turnos) × 7 (dias). Em cada célula
 * registamos a contagem total e a partição por severidade — usada pelo
 * tooltip do hover sem precisar de recomputo.
 */
export interface CellData {
  total: number;
  critical: number;
  alert: number;
}

export function buildMatrix(events: HistoryEvent[]): CellData[][] {
  const empty = (): CellData => ({ total: 0, critical: 0, alert: 0 });
  // Indexação interna: row 0..3 = 00/06/12/18h; col 0..6 = Seg..Dom.
  const matrix: CellData[][] = Array.from({ length: SHIFTS.length }, () =>
    Array.from({ length: DAYS.length }, empty),
  );

  for (const e of events) {
    const dt = new Date(e.timestamp);
    if (Number.isNaN(dt.getTime())) continue;

    // Turno: 0–5h → row 0, 6–11h → 1, 12–17h → 2, 18–23h → 3.
    const row = Math.floor(dt.getUTCHours() / 6);
    if (row < 0 || row > 3) continue;

    // getUTCDay() retorna 0 = Domingo, 1 = Segunda, ..., 6 = Sábado.
    // Nosso eixo é Seg..Dom (índice 0 = Seg). Mapeamento: (day + 6) % 7.
    const col = (dt.getUTCDay() + 6) % 7;

    const cell = matrix[row][col];
    cell.total++;
    if (e.severity === "CRÍTICO") cell.critical++;
    else if (e.severity === "ALERTA") cell.alert++;
  }

  return matrix;
}
