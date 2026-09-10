// ── Tooltip customizado — RNF-58: extraído de AlertFrequencyChart ───────────

import { memo } from "react";

interface TooltipEntry {
  dataKey: string;
  value: number;
}

interface CustomTooltipProps {
  active?: boolean;
  label?: string;
  payload?: TooltipEntry[];
}

export const CustomTooltip = memo(function CustomTooltip({
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

export const TOOLTIP_CONTENT = <CustomTooltip />;
