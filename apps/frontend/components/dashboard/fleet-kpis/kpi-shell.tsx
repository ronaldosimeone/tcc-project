// ── Shell de card — RNF-58: extraído de FleetKPIs ────────────────────────────

import type { ComponentType, ReactNode } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { TONE_BG, type SparkTone } from "./constants";
import { KpiSparkline } from "./kpi-sparkline";

export interface KpiShellProps {
  title: string;
  value: ReactNode;
  /** Linha de contexto opcional abaixo do valor (ex.: percentil). */
  subtitle?: string;
  icon: ComponentType<{ className?: string }>;
  iconTone: SparkTone;
  /** Série da sparkline. Omitir para cards que renderizam um visual customizado. */
  spark?: ReadonlyArray<number>;
  sparkTone?: SparkTone;
  /**
   * Visual alternativo no rodapé do card (ex.: Andon Board). Quando provido,
   * substitui a sparkline. `spark` e `children` são mutuamente exclusivos —
   * `children` vence se ambos forem passados.
   */
  children?: ReactNode;
  isLoading?: boolean;
}

export function KpiShell({
  title,
  value,
  subtitle,
  icon: Icon,
  iconTone,
  spark,
  sparkTone,
  children,
  isLoading,
}: KpiShellProps) {
  // children > sparkline. Cards podem optar por nenhum visual de rodapé
  // simplesmente omitindo ambos.
  const footerVisual: ReactNode = children ? (
    <div className="mt-3 h-11">{children}</div>
  ) : spark ? (
    <div className="mt-2 -mb-1 h-11">
      <KpiSparkline data={spark} tone={sparkTone ?? "neutral"} />
    </div>
  ) : null;
  return (
    <Card className="border border-slate-200 bg-white shadow-sm ring-0">
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-500">
              {title}
            </p>
            {isLoading ? (
              <Skeleton className="mt-2 h-7 w-20" />
            ) : (
              <>
                <div className="mt-1 font-mono text-2xl font-bold tracking-tight tabular-nums text-slate-900">
                  {value}
                </div>
                {subtitle && (
                  <p className="mt-0.5 text-[10px] text-slate-500">
                    {subtitle}
                  </p>
                )}
              </>
            )}
          </div>
          <div className={cn("shrink-0 rounded-lg p-2", TONE_BG[iconTone])}>
            <Icon className="h-4 w-4" />
          </div>
        </div>
        {footerVisual}
      </CardContent>
    </Card>
  );
}
