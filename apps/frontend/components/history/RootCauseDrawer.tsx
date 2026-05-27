"use client";

/**
 * Root Cause Drawer — análise preditiva de um evento histórico.
 *
 * Abre da direita ao clicar numa linha do EventLogTable. Conteúdo:
 *  1. Cabeçalho — descrição do alerta + badge de severidade.
 *  2. Janela de 2h antes da falha — mini LineChart (Recharts) com
 *     ReferenceLine tracejada marcando o threshold de probabilidade.
 *  3. Causa raiz da IA — bloco indigo com o diagnóstico mockado.
 *  4. Timeline — 3 pontos: Alerta inicial → Degradação crítica → Falha.
 *
 * O drawer é totalmente client-side (puro Tailwind, sem libs extra além
 * do Sheet já existente). Recebe `event` (ou null) e `onClose`.
 */

import { useMemo } from "react";
import {
  AlertOctagon,
  AlertTriangle,
  CheckCircle2,
  Sparkles,
} from "lucide-react";
import {
  Area,
  AreaChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Badge } from "@/components/ui/badge";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { cn } from "@/lib/utils";
import type { HistoryEvent, Severity } from "@/lib/history-mock";

// ── Severity tokens (literais — evita tokens hsl quebrados) ─────────────────

const SEVERITY_BADGE: Record<Severity, string> = {
  CRÍTICO: "border-red-200 bg-red-50 text-red-700",
  ALERTA: "border-amber-200 bg-amber-50 text-amber-700",
  NORMAL: "border-emerald-200 bg-emerald-50 text-emerald-700",
};

const SEVERITY_ICON: Record<
  Severity,
  React.ComponentType<{ className?: string }>
> = {
  CRÍTICO: AlertOctagon,
  ALERTA: AlertTriangle,
  NORMAL: CheckCircle2,
};

// ── Mock: 2h antes da falha (1 ponto / 10 min = 13 pontos) ──────────────────
// Linha que começa estável, oscila no meio e dispara perto do fim, cruzando
// o threshold (0.65 → CRÍTICO). Determinístico — independente do `event`.

const PREDICTIVE_WINDOW = [
  { t: "-2h", p: 0.12 },
  { t: "-1h50", p: 0.14 },
  { t: "-1h40", p: 0.18 },
  { t: "-1h30", p: 0.22 },
  { t: "-1h20", p: 0.21 },
  { t: "-1h10", p: 0.27 },
  { t: "-1h", p: 0.31 },
  { t: "-50min", p: 0.38 },
  { t: "-40min", p: 0.42 },
  { t: "-30min", p: 0.51 },
  { t: "-20min", p: 0.62 },
  { t: "-10min", p: 0.78 },
  { t: "0", p: 0.91 },
] as const;

const CRITICAL_THRESHOLD = 0.65;

// ── Tooltip do mini-chart ────────────────────────────────────────────────────

interface ChartTooltipProps {
  active?: boolean;
  label?: string;
  payload?: Array<{ value: number }>;
}

function PredictiveTooltip({ active, label, payload }: ChartTooltipProps) {
  if (!active || !payload?.length) return null;
  const value = payload[0].value;
  const pct = (value * 100).toFixed(1);
  const isCritical = value >= CRITICAL_THRESHOLD;
  return (
    <div className="rounded-md bg-slate-900 px-2 py-1.5 text-white shadow-lg">
      <p className="text-[10px] font-medium uppercase tracking-wider text-slate-400">
        {label}
      </p>
      <p
        className={cn(
          "text-xs font-bold tabular-nums",
          isCritical ? "text-red-400" : "text-amber-300",
        )}
      >
        {pct}% prob.
      </p>
    </div>
  );
}

// ── Timeline ────────────────────────────────────────────────────────────────

interface TimelineStep {
  label: string;
  detail: string;
  tone: "warn" | "danger" | "neutral";
}

const TIMELINE: ReadonlyArray<TimelineStep> = [
  {
    label: "Alerta Inicial",
    detail: "Vibração anômala detectada no rolamento principal",
    tone: "warn",
  },
  {
    label: "Degradação Crítica",
    detail: "Pressão DV cruza o threshold de 65% de risco",
    tone: "danger",
  },
  {
    label: "Falha do Equipamento",
    detail: "Compressor desliga; intervenção mecânica acionada",
    tone: "neutral",
  },
];

const TIMELINE_DOT: Record<TimelineStep["tone"], string> = {
  warn: "bg-amber-500 ring-amber-200",
  danger: "bg-red-500 ring-red-200 animate-pulse",
  neutral: "bg-slate-400 ring-slate-200",
};

// ── Componente principal ────────────────────────────────────────────────────

interface RootCauseDrawerProps {
  event: HistoryEvent | null;
  onClose: () => void;
}

export default function RootCauseDrawer({
  event,
  onClose,
}: RootCauseDrawerProps) {
  const open = event !== null;
  const SevIcon = event ? SEVERITY_ICON[event.severity] : AlertOctagon;
  const badgeCls = event
    ? SEVERITY_BADGE[event.severity]
    : SEVERITY_BADGE.ALERTA;

  // Memoizado para evitar re-mount do Recharts a cada hover.
  const chartData = useMemo(() => [...PREDICTIVE_WINDOW], []);

  return (
    <Sheet
      open={open}
      onOpenChange={(o) => {
        if (!o) onClose();
      }}
    >
      <SheetContent
        side="right"
        className="w-full overflow-y-auto sm:max-w-[500px]"
      >
        <SheetHeader>
          <div className="flex items-start justify-between gap-3 pr-8">
            <div className="min-w-0 flex-1">
              <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-500">
                Análise de Causa Raiz
              </p>
              <SheetTitle className="mt-0.5 text-base font-semibold text-slate-900">
                {event?.description ?? "—"}
              </SheetTitle>
              <SheetDescription className="mt-1 font-mono text-[11px] text-slate-500">
                {event?.equipment ?? ""} · {event?.timestamp ?? ""}
              </SheetDescription>
            </div>
            {event && (
              <Badge
                variant="outline"
                className={cn("gap-1 text-[11px] font-semibold", badgeCls)}
              >
                <SevIcon className="h-3 w-3" />
                {event.severity}
              </Badge>
            )}
          </div>
        </SheetHeader>

        {event && (
          <div className="flex flex-col gap-5 px-6 pb-6">
            {/* ── Janela preditiva (2h antes) ── */}
            <section>
              <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-slate-500">
                Janela preditiva · 2 horas antes da falha
              </p>
              <div className="rounded-lg border border-slate-200 bg-white p-3">
                <ResponsiveContainer width="100%" height={160}>
                  <AreaChart
                    data={chartData}
                    margin={{ top: 8, right: 8, bottom: 0, left: -16 }}
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
                          stopColor="#f43f5e"
                          stopOpacity={0.35}
                        />
                        <stop
                          offset="100%"
                          stopColor="#f43f5e"
                          stopOpacity={0}
                        />
                      </linearGradient>
                    </defs>
                    <XAxis
                      dataKey="t"
                      stroke="#94a3b8"
                      fontSize={10}
                      tickLine={false}
                      axisLine={false}
                      interval={1}
                    />
                    <YAxis
                      stroke="#94a3b8"
                      fontSize={10}
                      tickLine={false}
                      axisLine={false}
                      domain={[0, 1]}
                      tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
                      width={32}
                    />
                    <Tooltip content={<PredictiveTooltip />} />
                    <ReferenceLine
                      y={CRITICAL_THRESHOLD}
                      stroke="#ef4444"
                      strokeDasharray="4 3"
                      strokeWidth={1.5}
                      label={{
                        value: "CRÍTICO 65%",
                        position: "insideTopRight",
                        fontSize: 9,
                        fill: "#ef4444",
                      }}
                    />
                    <Area
                      type="monotone"
                      dataKey="p"
                      stroke="#f43f5e"
                      strokeWidth={2}
                      fill="url(#drawer-prob-grad)"
                      isAnimationActive={false}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </section>

            {/* ── Causa raiz da IA ── */}
            <section
              className="rounded-lg border border-slate-200 border-l-4 border-l-indigo-500 bg-indigo-50/50 p-4"
              aria-label="Diagnóstico da IA"
            >
              <div className="flex items-start gap-3">
                <span className="mt-0.5 shrink-0 rounded-md bg-indigo-100 p-1.5 text-indigo-600">
                  <Sparkles className="h-3.5 w-3.5" />
                </span>
                <div className="min-w-0 flex-1">
                  <p className="text-[10px] font-bold uppercase tracking-wider text-indigo-700">
                    Análise Preditiva
                  </p>
                  <p className="mt-1 text-sm leading-relaxed text-slate-700">
                    A anomalia foi precedida por{" "}
                    <span className="font-semibold text-slate-900">
                      vibração anômala no rolamento principal
                    </span>{" "}
                    cerca de{" "}
                    <span className="font-semibold text-slate-900">
                      45 minutos
                    </span>{" "}
                    antes do evento crítico. Padrão compatível com{" "}
                    <span className="font-semibold text-slate-900">
                      desgaste de material
                    </span>
                    .
                  </p>
                </div>
              </div>
            </section>

            {/* ── Timeline 3 passos ── */}
            <section>
              <p className="mb-3 text-[10px] font-semibold uppercase tracking-widest text-slate-500">
                Linha do tempo do incidente
              </p>
              <ol className="relative space-y-3">
                {/* Linha vertical conectora */}
                <span
                  aria-hidden="true"
                  className="absolute left-[5px] top-2 bottom-2 w-px bg-slate-200"
                />
                {TIMELINE.map((step, i) => (
                  <li
                    key={step.label}
                    className="relative flex items-start gap-3 pl-1"
                  >
                    <span
                      className={cn(
                        "mt-1 inline-flex h-2.5 w-2.5 shrink-0 rounded-full ring-4",
                        TIMELINE_DOT[step.tone],
                      )}
                      aria-hidden="true"
                    />
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-semibold text-slate-900">
                        {i + 1}. {step.label}
                      </p>
                      <p className="text-xs text-slate-500">{step.detail}</p>
                    </div>
                  </li>
                ))}
              </ol>
            </section>
          </div>
        )}
      </SheetContent>
    </Sheet>
  );
}
