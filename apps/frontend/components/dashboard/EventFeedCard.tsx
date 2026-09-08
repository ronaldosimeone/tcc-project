"use client";
/* eslint-disable react-hooks/purity --
   `Date.now()` no useMemo é intencional: o feed renderiza timestamps
   relativos ("há 2 min") frescos a cada mensagem WS, comportamento
   incompatível com a expectativa de pureza determinística da regra. */

/**
 * Feed de eventos recentes do cockpit.
 *
 * Alimenta-se de duas fontes:
 *  - Alertas vivos do WebSocket (use-alert-websocket) — sempre que existem,
 *    aparecem no topo.
 *  - Eventos estáticos de demonstração — preenchem a lista quando o WS está
 *    em silêncio, mantendo o cockpit visualmente populado para showcase do TCC.
 */

import { memo, useMemo, type ComponentType } from "react";
import { AlertTriangle, CheckCircle2, Info, Radio, Wrench } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { RISK_THRESHOLDS } from "@/lib/risk-thresholds";
import { cn } from "@/lib/utils";
import { useAlertWebSocket } from "@/hooks/use-alert-websocket";

/** Tempo relativo em pt-BR — "agora", "há 2 min", "há 3 h". */
function fromNow(ts: number, now: number = Date.now()): string {
  const diffSec = Math.max(0, Math.floor((now - ts) / 1000));
  if (diffSec < 30) return "agora";
  if (diffSec < 60) return `há ${diffSec}s`;
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `há ${diffMin} min`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `há ${diffHr}h`;
  const diffDay = Math.floor(diffHr / 24);
  return `há ${diffDay}d`;
}

type EventTone = "warn" | "ok" | "info" | "wrench";

interface FeedEvent {
  id: string;
  assetId: string;
  message: string;
  ts: number; // epoch ms
  tone: EventTone;
  /** Probabilidade de falha [0–1] — usada para gating de explicabilidade. */
  probability?: number;
}

// ── Explicabilidade da IA (Feature Importance mock) ─────────────────────────
// Para eventos críticos, simulamos a saída de SHAP/feature-importance do
// modelo. As 3 explicações alternam-se de forma determinística pelo hash
// do assetId, dando consistência por ativo entre re-renders.
const EXPLANATIONS: readonly string[] = [
  "Vibração elevada (3.2g) e Pressão atípica (8.1 bar).",
  "Temperatura do óleo acima do baseline (+12°C) e corrente irregular do motor.",
  "Ciclo de carga prolongado (>40s) combinado com queda na pressão DV.",
];

function hashAssetId(id: string): number {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) | 0;
  return Math.abs(h);
}

function explanationFor(assetId: string): string {
  return EXPLANATIONS[hashAssetId(assetId) % EXPLANATIONS.length];
}

function isCriticalEvent(evt: FeedEvent): boolean {
  if (
    evt.probability !== undefined &&
    evt.probability >= RISK_THRESHOLDS.CRITICAL
  )
    return true;
  return evt.tone === "warn";
}

const TONE_ICON: Record<EventTone, ComponentType<{ className?: string }>> = {
  warn: AlertTriangle,
  ok: CheckCircle2,
  info: Info,
  wrench: Wrench,
};

const TONE_COLOR: Record<EventTone, string> = {
  warn: "text-amber-600",
  ok: "text-emerald-600",
  info: "text-slate-500",
  wrench: "text-blue-600",
};

const TONE_BG: Record<EventTone, string> = {
  warn: "bg-amber-50",
  ok: "bg-emerald-50",
  info: "bg-slate-100",
  wrench: "bg-blue-50",
};

// Borda esquerda colorida — padrão visual SCADA para sinalizar severidade
// imediatamente no scan vertical. Eventos críticos têm a sua própria cor.
const TONE_BORDER: Record<EventTone, string> = {
  warn: "border-l-amber-500",
  ok: "border-l-emerald-500",
  info: "border-l-slate-300",
  wrench: "border-l-blue-500",
};

const CRITICAL_BORDER = "border-l-red-500";

// Eventos fixos para preencher o feed quando não há alertas vivos.
// Timestamps relativos ao "agora" para parecer vivo no demo.
const DEMO_EVENTS: Array<Omit<FeedEvent, "ts"> & { ageMin: number }> = [
  {
    id: "demo-1",
    assetId: "APU-Trem-023",
    message: "Risco de anomalia elevado",
    tone: "warn",
    ageMin: 2,
  },
  {
    id: "demo-2",
    assetId: "APU-Trem-011",
    message: "Sinal estabilizado",
    tone: "ok",
    ageMin: 7,
  },
  {
    id: "demo-3",
    assetId: "APU-Trem-031",
    message: "Manutenção preventiva concluída",
    tone: "wrench",
    ageMin: 18,
  },
  {
    id: "demo-4",
    assetId: "APU-Trem-055",
    message: "TP2 abaixo do esperado",
    tone: "info",
    ageMin: 31,
  },
  {
    id: "demo-5",
    assetId: "APU-Trem-015",
    message: "Ciclo de carga/descarga normal",
    tone: "ok",
    ageMin: 44,
  },
];

// Defesa em profundidade: o hook já corta em QUEUE_MAX=5, mas mantemos
// um teto duro aqui caso o feed evolua para acumular histórico próprio.
const MAX_FEED_EVENTS = 50;
// Quantos eventos efectivamente renderizamos na lista compacta.
// 10 linhas cabem confortavelmente dentro do h-[260px] com scroll suave.
const VISIBLE_FEED_EVENTS = 10;

/** Horário curto (HH:mm:ss) — formato típico de log industrial / SCADA. */
function formatClockTime(ts: number): string {
  return new Date(ts).toLocaleTimeString("pt-BR", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

/**
 * `React.memo`: componente sem props — todo o estado (`alerts`) vem de
 * `useAlertWebSocket()` internamente, uma fonte 100% independente do tick
 * SSE de 1Hz que re-renderiza o `FleetDashboard` pai. Sem memo, ainda assim
 * re-renderizava a cada tick SSE (medido com React Profiler — ver
 * frontend_performance_report.md) mesmo sem nenhuma prop para justificar.
 */
const EventFeedCard = memo(function EventFeedCard() {
  const { alerts } = useAlertWebSocket();

  const events: FeedEvent[] = useMemo(() => {
    const now = Date.now();

    const liveEvents: FeedEvent[] = alerts
      .slice(0, MAX_FEED_EVENTS)
      .map((a, i) => ({
        // ID estável por message_id; index só desempata colisões na render.
        id: `live-${a.message_id ?? a.timestamp ?? i}`,
        assetId: "APU-Trem-042",
        message: `Anomalia detectada (${(a.probability * 100).toFixed(1)}%)`,
        ts: a.timestamp ? new Date(a.timestamp).getTime() : now - i * 1000,
        tone: "warn",
        probability: a.probability,
      }));

    const demoEvents: FeedEvent[] = DEMO_EVENTS.map((e) => ({
      ...e,
      ts: now - e.ageMin * 60 * 1000,
    }));

    return [...liveEvents, ...demoEvents]
      .sort((a, b) => b.ts - a.ts)
      .slice(0, VISIBLE_FEED_EVENTS);
  }, [alerts]);

  return (
    <Card className="flex w-full flex-col border border-slate-200 bg-white shadow-sm ring-0">
      <CardHeader className="px-5 pb-2 pt-4">
        <CardTitle className="flex items-center justify-between text-sm font-semibold text-slate-900">
          <span className="flex items-center gap-2">
            <Radio className="h-4 w-4 text-slate-500" />
            Eventos Recentes
          </span>
          <span className="text-[10px] font-medium uppercase tracking-wider text-slate-400">
            {events.length} eventos · log
          </span>
        </CardTitle>
      </CardHeader>

      {/* Lista compacta tipo SCADA: alta densidade, severidade pela borda
          esquerda, todos os campos numa única linha quando cabe. */}
      <CardContent className="px-2 pb-2">
        <ScrollArea className="h-[260px] pr-2">
          <ul className="flex flex-col">
            {events.map((evt, index) => {
              const Icon = TONE_ICON[evt.tone];
              const critical = isCriticalEvent(evt);
              const borderTone = critical
                ? CRITICAL_BORDER
                : TONE_BORDER[evt.tone];
              return (
                <li
                  // Composto com index: blinda contra colisões de id quando o
                  // backend dispara eventos em rajada com timestamps idênticos.
                  key={`${evt.id}-${index}`}
                  className={cn(
                    "flex items-center gap-2.5 border-b border-l-4 border-slate-100 px-3 py-1.5 text-xs last:border-b-0 hover:bg-slate-50/60",
                    borderTone,
                  )}
                >
                  {/* Horário monoespaçado — referência temporal precisa.
                      suppressHydrationWarning: o HTML do SSR é renderizado em
                      UTC enquanto o cliente formata no fuso local — divergência
                      esperada. Após a primeira interação no cliente o valor
                      converge para o horário local correto. */}
                  <span
                    suppressHydrationWarning
                    className="shrink-0 font-mono text-[11px] tabular-nums text-slate-500"
                  >
                    {formatClockTime(evt.ts)}
                  </span>

                  {/* Ícone de severidade — minúsculo, sem fundo. */}
                  <Icon
                    className={cn("h-3 w-3 shrink-0", TONE_COLOR[evt.tone])}
                    aria-hidden="true"
                  />

                  {/* Asset ID — fixo em mono para alinhamento vertical na coluna. */}
                  <span className="shrink-0 font-mono text-[11px] font-semibold text-slate-700">
                    {evt.assetId}
                  </span>

                  {/* Mensagem + explicabilidade inline, ambos truncam se necessário. */}
                  <span className="flex min-w-0 flex-1 items-baseline gap-1.5">
                    <span className="truncate text-slate-900">
                      {evt.message}
                    </span>
                    {critical && (
                      <span className="hidden truncate text-[11px] text-slate-500 lg:inline">
                        <span className="text-slate-400">·</span>{" "}
                        {explanationFor(evt.assetId)}
                      </span>
                    )}
                  </span>

                  {/* Tempo relativo à direita — opcional, dispensável em viewports
                      muito estreitos. suppressHydrationWarning porque `fromNow`
                      depende de `Date.now()`, que difere entre o instante do
                      render no servidor e o instante do paint no cliente. */}
                  <span
                    suppressHydrationWarning
                    className="hidden shrink-0 text-[10px] text-slate-400 sm:inline"
                  >
                    {fromNow(evt.ts)}
                  </span>
                </li>
              );
            })}
          </ul>
        </ScrollArea>
      </CardContent>
    </Card>
  );
});

export default EventFeedCard;
