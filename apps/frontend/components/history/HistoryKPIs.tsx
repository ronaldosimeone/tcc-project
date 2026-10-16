"use client";

/**
 * KPIs de confiabilidade — calculados pelo backend (`GET /v1/kpis`) a partir
 * de dados ROTULADOS: só eventos cuja OS foi fechada como falha confirmada
 * entram em MTBF/MTTR/disponibilidade; falsos alarmes entram na precisão e
 * na taxa semanal. OEE não é exibido: exige contagens de produção que um
 * compressor não registra (o backend devolve `oee = null` com a justificativa).
 *
 * "Ativo mais crítico" é derivado dos eventos filtrados (mesma fonte da tabela).
 */

import { useMemo, type ComponentType } from "react";
import {
  Activity,
  AlertTriangle,
  Clock,
  Gauge,
  ShieldCheck,
  Timer,
} from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import type { KpiReportDto } from "@/lib/history-api";
import { formatHours, type HistoryEvent } from "@/lib/history-types";
import { cn } from "@/lib/utils";

interface KpiTileProps {
  title: string;
  value: string;
  subtitle: string;
  icon: ComponentType<{ className?: string }>;
  iconTone: "neutral" | "ok" | "warn" | "danger";
}

const TONE: Record<KpiTileProps["iconTone"], string> = {
  neutral: "bg-muted text-muted-foreground",
  ok: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400",
  warn: "bg-amber-500/10 text-amber-600 dark:text-amber-400",
  danger: "bg-rose-500/10 text-rose-600 dark:text-rose-400",
};

function KpiTile({
  title,
  value,
  subtitle,
  icon: Icon,
  iconTone,
}: KpiTileProps) {
  return (
    <Card className="border border-border bg-card shadow-sm ring-0">
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
              {title}
            </p>
            <p className="mt-1 truncate font-mono text-xl font-bold tracking-tight tabular-nums text-foreground">
              {value}
            </p>
            <p className="mt-0.5 text-[11px] leading-snug text-muted-foreground">
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

const pct = (v: number | null): string =>
  v === null ? "—" : `${(v * 100).toFixed(1)}%`;

const topEquipment = (
  events: HistoryEvent[],
): { equipment: string; share: number } => {
  const count = new Map<string, number>();
  for (const e of events)
    count.set(e.equipment, (count.get(e.equipment) ?? 0) + 1);
  let equipment = "—";
  let n = 0;
  for (const [eq, c] of count.entries()) {
    if (c > n) {
      n = c;
      equipment = eq;
    }
  }
  return {
    equipment,
    share: events.length > 0 ? Math.round((n / events.length) * 100) : 0,
  };
};

const reliabilityTiles = (k: KpiReportDto | null): KpiTileProps[] => {
  const labelled = k ? k.confirmed_failures + k.false_alarms : 0;
  return [
    {
      title: "MTBF",
      value: formatHours(k?.mtbf_hours ?? null),
      subtitle: `${
        k?.confirmed_failures ?? 0
      } falhas confirmadas em ${formatHours(
        k?.observed_hours ?? null,
      )} observadas`,
      icon: Timer,
      iconTone: "ok",
    },
    {
      title: "MTTR",
      value: formatHours(k?.mttr_hours ?? null),
      subtitle: "Média das OS fechadas de falhas confirmadas",
      icon: Clock,
      iconTone: "neutral",
    },
    {
      title: "Disponibilidade",
      value: pct(k?.availability ?? null),
      subtitle: "MTBF / (MTBF + MTTR)",
      icon: Gauge,
      iconTone: "ok",
    },
    {
      title: "Precisão dos alarmes",
      value: pct(k?.alarm_precision ?? null),
      subtitle:
        labelled > 0
          ? `${labelled} eventos rotulados · ${
              k?.unlabeled_events ?? 0
            } sem rótulo`
          : "Nenhuma OS fechada com veredito ainda",
      icon: ShieldCheck,
      iconTone: "neutral",
    },
    {
      title: "Falsos alarmes / semana",
      value: k ? k.false_alarms_per_week.toFixed(2) : "—",
      subtitle: `${k?.false_alarms ?? 0} confirmados pelo técnico`,
      icon: Activity,
      iconTone: "warn",
    },
  ];
};

interface HistoryKPIsProps {
  events: HistoryEvent[];
  kpis: KpiReportDto | null;
}

export default function HistoryKPIs({ events, kpis }: HistoryKPIsProps) {
  const top = useMemo(() => topEquipment(events), [events]);
  const tiles = useMemo(() => reliabilityTiles(kpis), [kpis]);

  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-6">
      {tiles.map((t) => (
        <KpiTile key={t.title} {...t} />
      ))}
      <KpiTile
        title="Ativo mais crítico"
        value={top.equipment}
        subtitle={
          top.share > 0
            ? `Concentra ${top.share}% dos episódios`
            : "Sem eventos no recorte"
        }
        icon={AlertTriangle}
        iconTone="danger"
      />
    </div>
  );
}
