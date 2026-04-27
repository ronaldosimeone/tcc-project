"use client";

import type { ComponentType } from "react";
import {
  Activity,
  AlertOctagon,
  Cpu,
  Minus,
  TrendingDown,
  TrendingUp,
  Wifi,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import type { RiskLevel } from "@/hooks/use-sensor-data";
import type { SSEStatus } from "@/hooks/use-sse";

interface FleetKPIsProps {
  effectiveRiskLevel: RiskLevel;
  sseStatus: SSEStatus;
  isLoading: boolean;
}

interface TrendConfig {
  value: string;
  direction: "up" | "down" | "neutral";
  positive: boolean;
}

interface KpiCardProps {
  title: string;
  value: string;
  sub: string;
  icon: ComponentType<{ className?: string }>;
  isLoading?: boolean;
  trend?: TrendConfig;
  accentClass: string;
  iconBgClass: string;
  iconTextClass: string;
}

function Trend({ value, direction, positive }: TrendConfig) {
  const colorClass =
    direction === "neutral"
      ? "text-muted-foreground"
      : positive
      ? "text-green-400"
      : "text-red-400";

  const Icon =
    direction === "up"
      ? TrendingUp
      : direction === "down"
      ? TrendingDown
      : Minus;

  return (
    <span
      className={cn(
        "flex items-center gap-0.5 text-[11px] font-semibold",
        colorClass,
      )}
    >
      <Icon className="h-3 w-3" />
      {value}
    </span>
  );
}

function KpiCard({
  title,
  value,
  sub,
  icon: Icon,
  isLoading,
  trend,
  accentClass,
  iconBgClass,
  iconTextClass,
}: KpiCardProps) {
  return (
    <Card
      className={cn(
        "relative overflow-hidden border-border bg-card",
        "transition-all duration-200 hover:shadow-lg hover:shadow-foreground/5",
        "focus-within:ring-2 focus-within:ring-primary/50",
      )}
    >
      {/* Accent bar */}
      <div
        className={cn("absolute left-0 top-0 h-full w-0.5", accentClass)}
        aria-hidden="true"
      />

      <CardContent className="p-4 pl-5">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
              {title}
            </p>

            {isLoading ? (
              <div className="mt-2 space-y-1.5">
                <Skeleton className="h-9 w-16" />
                <Skeleton className="h-3 w-32" />
              </div>
            ) : (
              <>
                <div className="mt-1 flex items-baseline gap-2">
                  <p className="font-mono text-4xl font-bold tabular-nums tracking-tight text-foreground">
                    {value}
                  </p>
                  {trend && <Trend {...trend} />}
                </div>
                <p className="mt-0.5 text-[11px] leading-snug text-muted-foreground">
                  {sub}
                </p>
              </>
            )}
          </div>

          <div className={cn("shrink-0 rounded-xl p-2.5", iconBgClass)}>
            <Icon className={cn("h-5 w-5", iconTextClass)} />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default function FleetKPIs({
  effectiveRiskLevel,
  sseStatus,
  isLoading,
}: FleetKPIsProps) {
  const isCritical = effectiveRiskLevel === "CRÍTICO";
  const isAlert = effectiveRiskLevel === "ALERTA";
  const isOnline = sseStatus === "connected";

  const riskAccent = isCritical
    ? "bg-red-500"
    : isAlert
    ? "bg-amber-500"
    : "bg-green-500";
  const riskIconBg = isCritical
    ? "bg-red-500/10"
    : isAlert
    ? "bg-amber-500/10"
    : "bg-green-500/10";
  const riskIconText = isCritical
    ? "text-red-400"
    : isAlert
    ? "text-amber-400"
    : "text-green-400";
  const riskSub = isCritical
    ? "APU-Trem-042 · Atenção imediata"
    : isAlert
    ? "APU-Trem-042 · Degradação detectada"
    : "Todos os ativos operacionais";

  return (
    <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
      <KpiCard
        title="Ativos Monitorados"
        value="5"
        sub="Compressores MetroPT-3"
        icon={Cpu}
        accentClass="bg-primary"
        iconBgClass="bg-primary/10"
        iconTextClass="text-primary"
        trend={{ value: "estável", direction: "neutral", positive: true }}
      />

      <KpiCard
        title="Em Risco Crítico"
        value={isCritical ? "1" : isAlert ? "1" : "0"}
        sub={riskSub}
        icon={AlertOctagon}
        isLoading={isLoading}
        accentClass={riskAccent}
        iconBgClass={riskIconBg}
        iconTextClass={riskIconText}
        trend={
          isCritical
            ? { value: "crítico", direction: "up", positive: false }
            : isAlert
            ? { value: "alerta", direction: "up", positive: false }
            : { value: "normal", direction: "neutral", positive: true }
        }
      />

      <KpiCard
        title="Sensores Online"
        value={isOnline ? "5" : "4"}
        sub={isOnline ? "Telemetria 1 Hz ativa" : "1 em reconexão SSE"}
        icon={Wifi}
        isLoading={isLoading}
        accentClass={isOnline ? "bg-green-500" : "bg-amber-500"}
        iconBgClass={isOnline ? "bg-green-500/10" : "bg-amber-500/10"}
        iconTextClass={isOnline ? "text-green-400" : "text-amber-400"}
        trend={
          isOnline
            ? { value: "online", direction: "neutral", positive: true }
            : { value: "-1", direction: "down", positive: false }
        }
      />

      <KpiCard
        title="Uptime Médio (30d)"
        value="97.4%"
        sub="Meta: 95% · SLA cumprido"
        icon={Activity}
        accentClass="bg-green-500"
        iconBgClass="bg-green-500/10"
        iconTextClass="text-green-400"
        trend={{ value: "+0.3 pp", direction: "up", positive: true }}
      />
    </div>
  );
}
