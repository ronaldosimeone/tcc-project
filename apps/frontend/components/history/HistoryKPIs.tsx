"use client";

/**
 * KPIs operacionais derivados do array `events` recebido via prop —
 * fonte única (logs filtrados) com a EventLogTable. Sem mocks internos.
 *
 * Métricas:
 *   - MTTR  : duração média dos eventos CRÍTICO (proxy para tempo de
 *             recuperação a partir de uma falha).
 *   - Top   : equipamento com maior número de ocorrências no recorte.
 *   - Down  : downtime acumulado = soma das durações de CRÍTICO + ALERTA.
 */

import { useMemo, type ComponentType } from "react";
import { Activity, AlertTriangle, Clock } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import {
  formatMinutes,
  parseDurationMinutes,
  type HistoryEvent,
} from "@/lib/history-mock";
import { cn } from "@/lib/utils";

interface KpiTileProps {
  title: string;
  value: string;
  subtitle: string;
  icon: ComponentType<{ className?: string }>;
  iconTone: "neutral" | "ok" | "warn" | "danger";
}

const TONE: Record<KpiTileProps["iconTone"], string> = {
  neutral: "bg-slate-100 text-slate-600",
  ok: "bg-emerald-50 text-emerald-600",
  warn: "bg-amber-50 text-amber-600",
  danger: "bg-rose-50 text-rose-600",
};

function KpiTile({
  title,
  value,
  subtitle,
  icon: Icon,
  iconTone,
}: KpiTileProps) {
  return (
    <Card className="border border-slate-200 bg-white shadow-sm ring-0">
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-500">
              {title}
            </p>
            <p className="mt-1 truncate font-mono text-xl font-bold tracking-tight tabular-nums text-slate-900">
              {value}
            </p>
            <p className="mt-0.5 text-[11px] leading-snug text-slate-500">
              {subtitle}
            </p>
          </div>
          <div className={cn("shrink-0 rounded-lg p-2", TONE[iconTone])}>
            <Icon className="h-4 w-4" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

interface HistoryKPIsProps {
  events: HistoryEvent[];
}

export default function HistoryKPIs({ events }: HistoryKPIsProps) {
  const stats = useMemo(() => {
    // ── MTTR: média das durações dos eventos CRÍTICO ──
    const criticalDurations = events
      .filter((e) => e.severity === "CRÍTICO")
      .map((e) => parseDurationMinutes(e.duration));
    const mttrMin =
      criticalDurations.length > 0
        ? criticalDurations.reduce((a, b) => a + b, 0) /
          criticalDurations.length
        : 0;

    // ── Ativo mais crítico: equipamento com mais ocorrências (qualquer sev) ──
    const eqCount = new Map<string, number>();
    for (const e of events) {
      eqCount.set(e.equipment, (eqCount.get(e.equipment) ?? 0) + 1);
    }
    let topEquipment = "—";
    let topCount = 0;
    for (const [eq, c] of eqCount.entries()) {
      if (c > topCount) {
        topCount = c;
        topEquipment = eq;
      }
    }
    const topPct =
      events.length > 0 ? Math.round((topCount / events.length) * 100) : 0;

    // ── Downtime: soma das durações de CRÍTICO + ALERTA ──
    const downtimeMin = events
      .filter((e) => e.severity === "CRÍTICO" || e.severity === "ALERTA")
      .reduce((sum, e) => sum + parseDurationMinutes(e.duration), 0);

    return { mttrMin, topEquipment, topPct, downtimeMin };
  }, [events]);

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
      <KpiTile
        title="Tempo Médio de Recuperação (MTTR)"
        value={stats.mttrMin > 0 ? formatMinutes(stats.mttrMin) : "—"}
        subtitle={`Média de ${
          events.filter((e) => e.severity === "CRÍTICO").length
        } eventos críticos`}
        icon={Clock}
        iconTone="neutral"
      />
      <KpiTile
        title="Ativo mais Crítico"
        value={stats.topEquipment}
        subtitle={
          stats.topPct > 0
            ? `Concentra ${stats.topPct}% das ocorrências`
            : "Sem eventos no recorte"
        }
        icon={AlertTriangle}
        iconTone="danger"
      />
      <KpiTile
        title="Total de Downtime"
        value={stats.downtimeMin > 0 ? formatMinutes(stats.downtimeMin) : "—"}
        subtitle="Crítico + Alerta no recorte filtrado"
        icon={Activity}
        iconTone="warn"
      />
    </div>
  );
}
