// ── Sparkline genérica — RNF-58: extraído de FleetKPIs ───────────────────────

import { useMemo } from "react";
import { Area, AreaChart, ResponsiveContainer } from "recharts";
import { SPARK_COLOR, type SparkTone } from "./constants";

interface KpiSparklineProps {
  data: ReadonlyArray<number>;
  tone: SparkTone;
}

/** Mini AreaChart — 44px, sem eixos, sem grid. Decorativo. */
export function KpiSparkline({ data, tone }: KpiSparklineProps) {
  const color = SPARK_COLOR[tone];
  const gradId = `kpi-spark-${tone}`;
  const chartData = useMemo(() => data.map((v, i) => ({ i, v })), [data]);

  return (
    // aria-hidden + accessibilityLayer={false} (RNF-66 §12/§13): puramente
    // decorativo — o valor real já está em texto no KpiShell. Sem os dois,
    // o Recharts 3.x expõe o <svg> como tabIndex=0 role="application" SEM
    // nome acessível (accessibilityLayer default-on), um foco fantasma que
    // só aparece em browser real — jsdom nunca renderiza o SVG (dimensões
    // 0x0 sem ResizeObserver real), por isso passou despercebido no jest-axe.
    // `display: contents` (className="contents") no wrapper: some da árvore
    // de layout (o ResponsiveContainer mede o pai original normalmente),
    // mas `aria-hidden` continua a esconder toda a subárvore do leitor de tela.
    <div aria-hidden="true" className="contents">
      <ResponsiveContainer width="100%" height={44}>
        <AreaChart
          accessibilityLayer={false}
          data={chartData}
          margin={{ top: 4, right: 0, left: 0, bottom: 0 }}
        >
          <defs>
            <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.32} />
              <stop offset="100%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <Area
            type="monotone"
            dataKey="v"
            stroke={color}
            strokeWidth={1.75}
            fill={`url(#${gradId})`}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
