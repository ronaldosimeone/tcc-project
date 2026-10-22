"use client";

/**
 * Status do modelo ativo + distribuição de saúde da frota.
 *
 * Layout: cabeçalho com o modelo ativo e o seu limiar de decisão calibrado
 * (o mesmo que define CRÍTICO em toda a UI); abaixo, ocupando o resto do
 * card, o donut com quantos ativos estão em cada faixa e a legenda.
 */

import { memo, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { Brain, Gauge } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { listModels } from "@/lib/api-client";
import { formatFixed } from "@/lib/format";
import { formatModelName } from "@/lib/model-name";

// RNF-74/75 — code splitting: `recharts` (~100KB gzip) sai do First Load JS
// do Dashboard, vira chunk próprio. `ssr: true` (default, mantido
// explícito) — achado real medido nesta task: com `ssr:false`, o donut só
// aparece depois de um round-trip extra do chunk DEPOIS da hidratação, e o
// Lighthouse mediu isso como regressão real de LCP/TTI (~450-650ms pior,
// reproduzido em 3 execuções) porque o elemento visualmente relevante
// (o card acima da dobra) só estabiliza depois desse fetch. Com SSR, o
// HTML já chega com o donut renderizado — o chunk do recharts carrega em
// paralelo em segundo plano só para hidratar, sem atrasar LCP. O bundle
// ainda sai do First Load JS do cliente (o ganho de RNF-74 é o mesmo) —
// só o QUANDO renderiza no servidor que muda.
const ModelStatusDonut = dynamic(
  () => import("./model-status-donut").then((m) => m.ModelStatusDonut),
  {
    ssr: true,
    loading: () => <Skeleton className="h-full w-full rounded-full" />,
  },
);

interface ModelStatusCardProps {
  /** Distribuição de saúde da frota (precomputada). */
  distribution: {
    healthy: number;
    warning: number;
    critical: number;
  };
}

const COLORS = {
  healthy: "#10b981",
  warning: "#f59e0b",
  critical: "#f43f5e",
} as const;

interface ActiveModel {
  name: string;
  threshold: number | null;
}

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
}: ModelStatusCardProps) {
  const [model, setModel] = useState<ActiveModel | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let alive = true;
    listModels()
      .then((res) => {
        if (alive && res.active_model) {
          setModel({
            name: res.active_model,
            threshold: res.active_threshold ?? null,
          });
        }
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
    <Card className="flex h-full flex-col border border-border bg-card shadow-sm ring-0 short:gap-2 short:py-2">
      <CardHeader className="px-5 pb-3 pt-4 short:pb-1 short:pt-2">
        <CardTitle className="flex items-center gap-2 text-sm font-semibold text-foreground">
          <Brain className="h-4 w-4 text-muted-foreground" />
          Status do Modelo
        </CardTitle>
      </CardHeader>

      <CardContent className="flex min-h-0 flex-1 flex-col overflow-y-auto px-5 pb-5">
        {/* ── Modelo ativo + limiar de decisão ── */}
        <div className="flex flex-wrap items-start justify-between gap-x-3 gap-y-2 border-b border-border/60 pb-3">
          <div className="min-w-0 flex-1">
            <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
              Inferência ativa
            </p>
            {error ? (
              <p className="mt-1 text-sm text-muted-foreground">indisponível</p>
            ) : model ? (
              <>
                <p className="mt-1 text-lg font-semibold leading-tight text-foreground">
                  {formatModelName(model.name)}
                </p>
                <p className="mt-0.5 font-mono text-xs uppercase tracking-wider text-muted-foreground">
                  {model.name}
                </p>
              </>
            ) : (
              <Skeleton className="mt-1 h-6 w-32" />
            )}
          </div>
          {model?.threshold != null && (
            <Badge
              variant="outline"
              className="gap-1 text-xs font-semibold"
              title="Probabilidade a partir da qual o modelo decide 'falha' (CRÍTICO)"
            >
              <Gauge className="h-3.5 w-3.5" />
              Limiar {formatFixed(model.threshold, 3)}
            </Badge>
          )}
        </div>

        {/* ── Donut + legenda ocupam o resto do card (antes: donut fixo de
            96 px e metade do card vazia no desktop). ── */}
        <div className="flex min-h-0 flex-1 flex-wrap items-center justify-center gap-x-6 gap-y-3 pt-3">
          <div className="relative aspect-square h-full max-h-44 min-h-24 shrink-0 short:max-h-28">
            <ModelStatusDonut pieData={pieData} />
            <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
              <span className="font-mono text-2xl font-bold tabular-nums leading-none text-foreground">
                {total}
              </span>
              <span className="text-xs uppercase tracking-wider text-muted-foreground">
                ativos
              </span>
            </div>
          </div>

          <ul className="flex min-w-32 flex-col gap-2 text-sm">
            {pieData.map((slice) => (
              <li key={slice.name} className="flex items-center gap-2">
                <span
                  className="h-2.5 w-2.5 shrink-0 rounded-sm"
                  style={{ backgroundColor: slice.color }}
                  aria-hidden="true"
                />
                <span className="flex-1 text-foreground/80">{slice.name}</span>
                <span className="ml-4 font-mono text-base font-semibold tabular-nums text-foreground">
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
