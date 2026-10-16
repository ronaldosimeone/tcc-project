// ── Janela real de P(falha) em torno do evento (−2 h … +30 min) ───────────────

import { memo, useMemo } from "react";
import dayjs from "dayjs";
import {
  Area,
  AreaChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { SensorPoint } from "@/lib/history-api";
import { PredictiveTooltip } from "./predictive-tooltip";

interface PredictiveWindowChartProps {
  points: SensorPoint[];
  startedAt: string;
  threshold: number;
}

/** Rótulo relativo ao início do episódio: segundos em janelas curtas
 * (episódios recentes, < 10 min de dados), minutos/horas nas longas. */
const relativeLabel = (
  ts: string,
  startedAt: string,
  shortSpan: boolean,
): string => {
  const seconds = dayjs(ts).diff(dayjs(startedAt), "second");
  if (seconds === 0) return "0";
  const sign = seconds < 0 ? "-" : "+";
  const abs = Math.abs(seconds);
  if (shortSpan) {
    return abs >= 60
      ? `${sign}${Math.floor(abs / 60)}m${String(abs % 60).padStart(2, "0")}s`
      : `${sign}${abs}s`;
  }
  const minutes = Math.round(abs / 60);
  return minutes >= 60
    ? `${sign}${Math.floor(minutes / 60)}h${String(minutes % 60).padStart(
        2,
        "0",
      )}`
    : `${sign}${minutes}min`;
};

export const PredictiveWindowChart = memo(function PredictiveWindowChart({
  points,
  startedAt,
  threshold,
}: PredictiveWindowChartProps) {
  const chartData = useMemo(() => {
    const span =
      points.length > 1
        ? dayjs(points[points.length - 1].timestamp).diff(
            dayjs(points[0].timestamp),
            "second",
          )
        : 0;
    const shortSpan = span < 600;
    return points.map((p) => ({
      t: relativeLabel(p.timestamp, startedAt, shortSpan),
      p: p.failure_probability,
    }));
  }, [points, startedAt]);

  if (chartData.length === 0) {
    return (
      <section className="rounded-lg border border-dashed border-border p-4 text-xs text-muted-foreground">
        Nenhuma leitura armazenada na janela deste evento.
      </section>
    );
  }

  return (
    <section>
      <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
        Probabilidade de falha · 2 h antes até 30 min depois (leituras reais)
      </p>
      <div className="rounded-lg border border-border bg-card p-3">
        <table className="sr-only">
          <caption>
            Probabilidade de falha registrada ao redor do evento (limiar{" "}
            {Math.round(threshold * 100)}%)
          </caption>
          <thead>
            <tr>
              <th scope="col">Tempo relativo ao início</th>
              <th scope="col">Probabilidade</th>
            </tr>
          </thead>
          <tbody>
            {chartData.map((point, i) => (
              <tr key={`${point.t}-${i}`}>
                <th scope="row">{point.t}</th>
                <td>{Math.round(point.p * 100)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div aria-hidden="true" className="contents">
          <ResponsiveContainer width="100%" height={160}>
            <AreaChart
              accessibilityLayer={false}
              data={chartData}
              margin={{ top: 8, right: 8, bottom: 0, left: 0 }}
            >
              <defs>
                <linearGradient
                  id="drawer-prob-grad"
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="1"
                >
                  <stop
                    offset="0%"
                    stopColor="var(--destructive)"
                    stopOpacity={0.35}
                  />
                  <stop
                    offset="100%"
                    stopColor="var(--destructive)"
                    stopOpacity={0}
                  />
                </linearGradient>
              </defs>
              <XAxis
                dataKey="t"
                stroke="var(--muted-foreground)"
                fontSize={10}
                tickLine={false}
                axisLine={false}
                minTickGap={24}
              />
              <YAxis
                stroke="var(--muted-foreground)"
                fontSize={10}
                tickLine={false}
                axisLine={false}
                domain={[0, 1]}
                tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
                width={40}
              />
              <Tooltip content={<PredictiveTooltip threshold={threshold} />} />
              <ReferenceLine
                y={threshold}
                stroke="var(--destructive)"
                strokeDasharray="4 3"
                strokeWidth={1.5}
                label={{
                  value: `LIMIAR ${Math.round(threshold * 100)}%`,
                  position: "insideTopRight",
                  fontSize: 9,
                  fill: "var(--destructive)",
                }}
              />
              <Area
                type="monotone"
                dataKey="p"
                stroke="var(--destructive)"
                strokeWidth={2}
                fill="url(#drawer-prob-grad)"
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  );
});
