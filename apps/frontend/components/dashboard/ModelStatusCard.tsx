"use client";

/**
 * Status do modelo activo + distribuição de saúde da frota.
 *
 * Layout: cabeçalho com modelo activo e confiança (mock) à esquerda,
 * donut radial à direita mostrando quantos ativos estão em cada faixa
 * (saudável/atenção/crítico). Donut clean — sem grid, sem legenda flutuante.
 */

import { memo, useEffect, useMemo, useState } from "react";
import { Brain, ShieldCheck } from "lucide-react";
import { Cell, Pie, PieChart, ResponsiveContainer } from "recharts";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { listModels } from "@/lib/api-client";
import { formatModelName } from "@/lib/model-name";

interface ModelStatusCardProps {
  /** Distribuição de saúde da frota (precomputada). */
  distribution: {
    healthy: number;
    warning: number;
    critical: number;
  };
  /** Confiança simulada do modelo activo [0–1] — placeholder. */
  confidence?: number;
}

const COLORS = {
  healthy: "#10b981",
  warning: "#f59e0b",
  critical: "#f43f5e",
} as const;

/**
 * `React.memo`: `distribution` já chega memoizada do pai (useMemo em
 * FleetDashboard, keyed em `effectiveRiskLevel`) — só muda de referência a
 * cada 5s (poll) ou alerta WS, não a cada tick SSE de 1Hz. Medido com React
 * Profiler: sem memo, este componente ainda re-renderizava a ~1/tick SSE
 * mesmo com `distribution` referencialmente igual (ver
 * frontend_performance_report.md).
 */
const ModelStatusCard = memo(function ModelStatusCard({
  distribution,
  confidence = 0.92,
}: ModelStatusCardProps) {
  const [activeModel, setActiveModel] = useState<string | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let alive = true;
    listModels()
      .then((res) => {
        if (alive) setActiveModel(res.active_model);
      })
      .catch(() => {
        if (alive) setError(true);
      });
    return () => {
      alive = false;
    };
  }, []);

  const total =
    distribution.healthy + distribution.warning + distribution.critical;

  const pieData = useMemo(
    () => [
      { name: "Saudáveis", value: distribution.healthy, color: COLORS.healthy },
      { name: "Atenção", value: distribution.warning, color: COLORS.warning },
      { name: "Crítico", value: distribution.critical, color: COLORS.critical },
    ],
    [distribution],
  );

  return (
    <Card className="flex h-full flex-col border border-slate-200 bg-white shadow-sm ring-0">
      <CardHeader className="px-5 pb-3 pt-4">
        <CardTitle className="flex items-center gap-2 text-sm font-semibold text-slate-900">
          <Brain className="h-4 w-4 text-slate-500" />
          Status do Modelo
        </CardTitle>
      </CardHeader>

      <CardContent className="flex flex-1 flex-col px-5 pb-5">
        {/* ── Cabeçalho: modelo activo + confiança ── */}
        <div className="flex items-start justify-between gap-3 border-b border-slate-100 pb-3">
          <div className="min-w-0 flex-1">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-500">
              Inferência activa
            </p>
            {error ? (
              <p className="mt-1 text-sm text-slate-400">indisponível</p>
            ) : activeModel ? (
              <>
                <p className="mt-1 text-base font-semibold text-slate-900">
                  {formatModelName(activeModel)}
                </p>
                <p className="font-mono text-[10px] uppercase tracking-wider text-slate-400">
                  {activeModel}
                </p>
              </>
            ) : (
              <Skeleton className="mt-1 h-5 w-32" />
            )}
          </div>
          <Badge
            variant="outline"
            className="gap-1 border-emerald-200 bg-emerald-50 text-[10px] font-semibold text-emerald-700"
          >
            <ShieldCheck className="h-3 w-3" />
            {(confidence * 100).toFixed(1)}% conf.
          </Badge>
        </div>

        {/* ── Donut: distribuição de saúde ── */}
        <div className="mt-3 flex items-center gap-4">
          <div className="relative h-24 w-24 shrink-0">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  innerRadius={28}
                  outerRadius={44}
                  paddingAngle={2}
                  dataKey="value"
                  stroke="none"
                  isAnimationActive={false}
                >
                  {pieData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
            <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
              <span className="font-mono text-lg font-bold tabular-nums leading-none text-slate-900">
                {total}
              </span>
              <span className="text-[9px] uppercase tracking-wider text-slate-400">
                ativos
              </span>
            </div>
          </div>

          <ul className="flex flex-1 flex-col gap-1.5 text-xs">
            {pieData.map((slice) => (
              <li key={slice.name} className="flex items-center gap-2">
                <span
                  className="h-2 w-2 shrink-0 rounded-sm"
                  style={{ backgroundColor: slice.color }}
                  aria-hidden="true"
                />
                <span className="flex-1 text-slate-700">{slice.name}</span>
                <span className="font-mono font-semibold tabular-nums text-slate-900">
                  {slice.value}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </CardContent>
    </Card>
  );
});

export default ModelStatusCard;
