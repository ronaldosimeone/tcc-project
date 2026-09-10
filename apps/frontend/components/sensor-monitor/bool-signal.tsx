"use client";

// ── Painel de Sinais Booleanos — RNF-58: extraído de sensor-monitor.tsx ─────

import { memo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import {
  BOOL_SIGNALS,
  type BoolSignalMode,
  type BooleanPanelProps,
} from "./constants";

export interface BoolSignalProps {
  label: string;
  value: number;
  mode?: BoolSignalMode;
  isCount?: boolean; // para Caudal Impulses
}

export function BoolSignal({
  label,
  value,
  mode = "normal",
  isCount = false,
}: BoolSignalProps) {
  const isOn = isCount ? value > 0 : value === 1;
  const isAlert =
    mode === "alertWhenOn" ? isOn : mode === "alertWhenOff" ? !isOn : false;
  const isSuccess = !isAlert && isOn;

  const dotCls = isAlert
    ? "bg-red-500 shadow-sm shadow-red-400/40 animate-pulse"
    : isSuccess
    ? "bg-emerald-500 shadow-sm shadow-emerald-400/30"
    : "bg-slate-300";

  const valueCls = isAlert
    ? "text-red-700 font-extrabold"
    : isSuccess
    ? "text-emerald-600"
    : "text-slate-400";

  const displayValue = isCount
    ? value > 0
      ? String(value)
      : "—"
    : isOn
    ? "ON"
    : "OFF";

  return (
    <div
      className={cn(
        "flex items-center justify-between rounded-md border p-2 transition-colors duration-200",
        isAlert ? "border-red-300 bg-red-50" : "border-slate-200 bg-white",
      )}
    >
      <span
        className={cn(
          "truncate text-[10px] font-semibold uppercase tracking-wide",
          isAlert ? "text-red-700" : "text-slate-500",
        )}
      >
        {label}
      </span>
      <div className="flex shrink-0 items-center gap-1">
        <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", dotCls)} />
        <span className={cn("text-[10px] font-bold tabular-nums", valueCls)}>
          {displayValue}
        </span>
      </div>
    </div>
  );
}

export const BooleanPanel = memo(function BooleanPanel(
  props: BooleanPanelProps,
) {
  return (
    <Card className="flex h-full flex-col border-slate-200">
      <CardHeader className="shrink-0 px-4 pb-2 pt-4">
        <CardTitle className="text-sm font-semibold text-foreground/90">
          Sinais Digitais
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col p-3 pt-0">
        <div className="grid h-full grid-cols-2 grid-rows-4 gap-2">
          {BOOL_SIGNALS.map(({ key, label, mode, isCount }) => (
            <BoolSignal
              key={key}
              label={label}
              value={props[key]}
              mode={mode}
              isCount={isCount}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  );
});
