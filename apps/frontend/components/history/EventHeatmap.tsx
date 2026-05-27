"use client";

/**
 * Event Heatmap — matriz de calor operacional (Turnos × Dias da semana).
 *
 * Recharts não oferece heatmap nativo; implementação pura com CSS Grid +
 * Tailwind, mantendo o estilo SCADA do resto do dashboard.
 *
 * Eixos
 * -----
 *   Linhas  : Turnos (00h, 06h, 12h, 18h) — 4 slots de 6 horas
 *   Colunas : Dias da semana (Seg → Dom) — 7 colunas
 *
 * Cores (severidade por densidade de eventos no turno)
 * ----------------------------------------------------
 *   0          → slate-100  (sem ocorrências)
 *   1–2        → emerald-500 (normal)
 *   3–5        → amber-500   (alerta)
 *   ≥ 6        → red-500 + animate-pulse (crítico)
 */

import { useMemo } from "react";
import { Activity } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { HistoryEvent } from "@/lib/history-mock";

// ── Configuração dos eixos ───────────────────────────────────────────────────

const SHIFTS = ["00h", "06h", "12h", "18h"] as const;
const DAYS = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"] as const;

type Shift = (typeof SHIFTS)[number];
type Day = (typeof DAYS)[number];

// ── Helpers de cor ───────────────────────────────────────────────────────────

/**
 * Classe Tailwind tonal por densidade. Cores sólidas + sombra interna sutil
 * (`shadow-inner ring-1 ring-inset ring-black/5`) dão sensação de relevo
 * sem comprometer a leitura imediata da intensidade.
 */
function intensityClass(count: number): string {
  if (count === 0) return "bg-slate-100 ring-1 ring-inset ring-black/[0.03]";
  if (count <= 2)
    return "bg-emerald-500 shadow-inner shadow-emerald-700/30 ring-1 ring-inset ring-emerald-600/30";
  if (count <= 5)
    return "bg-amber-500 shadow-inner shadow-amber-700/30 ring-1 ring-inset ring-amber-600/30";
  return "bg-red-500 shadow-inner shadow-red-700/30 ring-1 ring-inset ring-red-600/40 animate-pulse";
}

/** Label semântico para acessibilidade (`aria-label` + `title`). */
function intensityLabel(count: number): string {
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
interface CellData {
  total: number;
  critical: number;
  alert: number;
}

function buildMatrix(events: HistoryEvent[]): CellData[][] {
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

// ── Componente ──────────────────────────────────────────────────────────────

interface EventHeatmapProps {
  events: HistoryEvent[];
}

export default function EventHeatmap({ events }: EventHeatmapProps) {
  const matrix = useMemo(() => buildMatrix(events), [events]);
  const totalEvents = useMemo(
    () => matrix.flat().reduce((sum, c) => sum + c.total, 0),
    [matrix],
  );

  return (
    <Card className="flex h-full flex-col overflow-visible border border-slate-200 bg-white shadow-sm ring-0">
      <CardHeader className="px-5 pb-2 pt-4">
        <div className="flex items-baseline justify-between">
          <CardTitle className="flex items-center gap-2 text-sm font-semibold text-slate-900">
            <Activity className="h-4 w-4 text-slate-500" />
            Mapa de Calor de Ocorrências
          </CardTitle>
          <p className="text-[10px] font-medium uppercase tracking-wider text-slate-400">
            Turno × Dia da semana · {totalEvents} eventos
          </p>
        </div>
      </CardHeader>

      <CardContent className="flex flex-1 flex-col justify-center px-5 pb-5">
        {/* Eixos discretos: cabeçalho de dias no topo, turnos à esquerda.
            Grid 8 cols = 1 (turno label) + 7 (dias). */}
        <div className="grid grid-cols-[max-content_repeat(7,minmax(0,1fr))] gap-1.5">
          {/* Linha 0: cabeçalho de dias */}
          <span aria-hidden="true" />
          {DAYS.map((day) => (
            <span
              key={day}
              className="text-center font-mono text-[10px] uppercase tracking-wider text-slate-400"
            >
              {day}
            </span>
          ))}

          {/* Linhas dos turnos */}
          {SHIFTS.map((shift, rowIdx) => (
            <RowGroup key={shift} shift={shift} cells={matrix[rowIdx] ?? []} />
          ))}
        </div>

        {/* Legenda — escala de intensidade. */}
        <div className="mt-4 flex items-center gap-3 border-t border-slate-100 pt-3 text-[10px] text-slate-500">
          <span className="font-medium uppercase tracking-wider text-slate-400">
            Intensidade
          </span>
          <Swatch className="bg-slate-100" label="0" />
          <Swatch className="bg-emerald-500" label="1–2" />
          <Swatch className="bg-amber-500" label="3–5" />
          <Swatch className="bg-red-500" label="≥ 6" />
        </div>
      </CardContent>
    </Card>
  );
}

// ── Sub-componentes ─────────────────────────────────────────────────────────

interface RowGroupProps {
  shift: Shift;
  cells: ReadonlyArray<CellData>;
}

function RowGroup({ shift, cells }: RowGroupProps) {
  return (
    <>
      <span className="self-center pr-2 font-mono text-[10px] uppercase tracking-wider text-slate-400">
        {shift}
      </span>
      {DAYS.map((day, colIdx) => {
        const cell = cells[colIdx] ?? { total: 0, critical: 0, alert: 0 };
        return (
          <Cell key={`${shift}-${day}`} cell={cell} shift={shift} day={day} />
        );
      })}
    </>
  );
}

interface CellProps {
  cell: CellData;
  shift: Shift;
  day: Day;
}

function Cell({ cell, shift, day }: CellProps) {
  const { total: count, critical, alert } = cell;
  const label = `${day} ${shift}: ${count} ${
    count === 1 ? "evento" : "eventos"
  } (${intensityLabel(count)})`;

  return (
    // `group` + `relative` permite uma tooltip CSS-only via `group-hover`.
    // Sem libs externas — funciona em qualquer browser moderno.
    <div className="group relative">
      <div
        role="img"
        aria-label={label}
        className={cn(
          "h-4 w-full rounded-sm transition-colors",
          intensityClass(count),
        )}
      />

      {/* Tooltip: aparece acima da célula no hover, segue o cursor.
          `pointer-events-none` evita que ele intercepte hovers vizinhos.
          Centralizada via `left-1/2 -translate-x-1/2` para alinhamento óptico. */}
      <div
        role="tooltip"
        className={cn(
          "pointer-events-none absolute bottom-full left-1/2 z-50 mb-2 -translate-x-1/2",
          "hidden min-w-[10rem] rounded-md bg-slate-900 px-2 py-2 text-white shadow-lg",
          "group-hover:block",
        )}
      >
        <p className="text-[10px] font-medium uppercase tracking-wider text-slate-400">
          {day} · {shift}
        </p>
        <p className="mt-0.5 text-sm font-bold tabular-nums">
          {count} {count === 1 ? "ocorrência" : "ocorrências"}
        </p>
        {count > 0 && (
          <div className="mt-1 flex items-center gap-2 text-[11px] text-slate-200">
            <span className="inline-flex items-center gap-1">
              <span className="h-1.5 w-1.5 rounded-full bg-red-500" />
              <span className="tabular-nums">{critical} críticas</span>
            </span>
            <span className="text-slate-600">|</span>
            <span className="inline-flex items-center gap-1">
              <span className="h-1.5 w-1.5 rounded-full bg-amber-500" />
              <span className="tabular-nums">{alert} alertas</span>
            </span>
          </div>
        )}
        {/* Seta apontando para a célula */}
        <span
          aria-hidden="true"
          className="absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent border-t-slate-900"
        />
      </div>
    </div>
  );
}

interface SwatchProps {
  className: string;
  label: string;
}

function Swatch({ className, label }: SwatchProps) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className={cn("h-3 w-3 rounded-sm border border-black/5", className)}
        aria-hidden="true"
      />
      <span className="font-mono">{label}</span>
    </span>
  );
}
