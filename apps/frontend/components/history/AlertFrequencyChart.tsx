"use client";

import { memo, useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { TrendingUp } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { HistoryEvent } from "@/lib/history-mock";

// Literais slate — evita tokens hsl(var(--...)) que renderizam transparente
// pelo problema conhecido de mistura oklch/hsl no tema.
const AXIS_TICK = { fontSize: 11, fill: "#94a3b8" } as const;
const GRID_STROKE = "#e2e8f0";

interface TooltipEntry {
  dataKey: string;
  value: number;
}

interface CustomTooltipProps {
  active?: boolean;
  label?: string;
  payload?: TooltipEntry[];
}

const CustomTooltip = memo(function CustomTooltip({
  active,
  label,
  payload,
}: CustomTooltipProps) {
  if (!active || !payload?.length) return null;

  const critico = payload.find((p) => p.dataKey === "critico")?.value ?? 0;
  const alerta = payload.find((p) => p.dataKey === "alerta")?.value ?? 0;
  const total = critico + alerta;

  return (
    <div className="rounded-lg border border-border/80 bg-popover/95 px-3 py-2 shadow-xl backdrop-blur-sm">
      <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {label}
      </p>
      {total === 0 ? (
        <p className="text-xs text-muted-foreground">Sem ocorrências</p>
      ) : (
        <>
          {critico > 0 && (
            <div className="flex items-center gap-2 text-xs">
              <span className="inline-block h-2 w-2 rounded-full bg-red-500" />
              <span className="text-muted-foreground">Crítico</span>
              <span className="ml-auto font-bold tabular-nums text-red-500">
                {critico}
              </span>
            </div>
          )}
          {alerta > 0 && (
            <div className="flex items-center gap-2 text-xs">
              <span className="inline-block h-2 w-2 rounded-full bg-amber-500" />
              <span className="text-muted-foreground">Alerta</span>
              <span className="ml-auto font-bold tabular-nums text-amber-500">
                {alerta}
              </span>
            </div>
          )}
          <div className="mt-1.5 border-t border-border/60 pt-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-muted-foreground">Total</span>
              <span className="font-bold tabular-nums text-foreground">
                {total}
              </span>
            </div>
          </div>
        </>
      )}
    </div>
  );
});

const TOOLTIP_CONTENT = <CustomTooltip />;

// ── Tipo interno do agrupamento diário ──────────────────────────────────────

interface DayBucket {
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
function bucketEventsByDay(events: HistoryEvent[]): DayBucket[] {
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

interface AlertFrequencyChartProps {
  events: HistoryEvent[];
}

export default function AlertFrequencyChart({
  events,
}: AlertFrequencyChartProps) {
  const data = useMemo(() => bucketEventsByDay(events), [events]);
  const totalAlerts = data.reduce((acc, d) => acc + d.alerta + d.critico, 0);
  const peakDay = data.reduce(
    (max, d) => Math.max(max, d.alerta + d.critico),
    0,
  );

  return (
    <Card className="flex h-full flex-col border border-slate-200 bg-white shadow-sm ring-0">
      <CardHeader className="px-5 pb-2 pt-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <CardTitle className="flex items-center gap-2 text-sm font-semibold text-foreground/90">
              <TrendingUp className="h-4 w-4 text-primary" />
              Frequência de Ocorrências
            </CardTitle>
            <p className="mt-0.5 text-[11px] text-muted-foreground">
              Alertas e falhas por dia · Últimos 14 dias
            </p>
          </div>

          <div className="flex items-center gap-6 text-right">
            <div>
              <p className="font-mono text-2xl font-bold tabular-nums text-foreground">
                {totalAlerts}
              </p>
              <p className="text-[10px] text-muted-foreground">
                total no período
              </p>
            </div>
            <div>
              <p className="font-mono text-2xl font-bold tabular-nums tracking-tight text-amber-600 dark:text-amber-400">
                {peakDay}
              </p>
              <p className="text-[10px] text-muted-foreground">pico diário</p>
            </div>
          </div>
        </div>
      </CardHeader>

      {/* `flex-1 min-h-[200px]` faz o gráfico esticar verticalmente até o
          espaço residual do grid (100dvh layout) sem cair abaixo do mínimo
          legível em viewports baixos. */}
      <CardContent className="flex flex-1 min-h-[200px] flex-col px-5 pb-4">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            margin={{ top: 8, right: 4, bottom: 0, left: -10 }}
            barSize={14}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              vertical={false}
              stroke={GRID_STROKE}
            />
            <XAxis
              dataKey="date"
              tick={AXIS_TICK}
              tickLine={false}
              axisLine={false}
              interval={1}
            />
            <YAxis
              stroke="#94a3b8"
              fontSize={11}
              tickLine={false}
              axisLine={false}
              allowDecimals={false}
              width={28}
            />
            <Tooltip
              content={TOOLTIP_CONTENT}
              cursor={{ fill: "#f1f5f9", fillOpacity: 0.6 }}
            />
            <Bar dataKey="critico" name="Crítico" stackId="a" fill="#ef4444" />
            <Bar
              dataKey="alerta"
              name="Alerta"
              stackId="a"
              fill="#f59e0b"
              radius={[3, 3, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>

        <div className="mt-3 flex items-center justify-end gap-5 border-t border-border pt-3">
          <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
            <span className="inline-block h-2 w-2 rounded-full bg-red-400" />
            Falha Crítica
          </div>
          <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
            <span className="inline-block h-2 w-2 rounded-full bg-amber-400" />
            Alerta
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
