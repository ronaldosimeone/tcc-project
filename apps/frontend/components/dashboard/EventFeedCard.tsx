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
 *
 * RNF-58: decomposto em `components/dashboard/event-feed-card/*` —
 * constants-and-helpers (tipos/tons/mocks/formatação) e FeedEventRow.
 * Nenhuma mudança de comportamento/DOM.
 */

import { memo, useMemo } from "react";
import { Radio } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { useIsClient } from "@/hooks/use-is-client";
import { useAlertWebSocket } from "@/hooks/use-alert-websocket";
import { formatRisk, toRisk } from "@/lib/risk-thresholds";
import {
  DEMO_EVENTS,
  MAX_FEED_EVENTS,
  VISIBLE_FEED_EVENTS,
  type FeedEvent,
} from "./event-feed-card/constants-and-helpers";
import { FeedEventRow } from "./event-feed-card/feed-event-row";

/**
 * `React.memo`: componente sem props — todo o estado (`alerts`) vem de
 * `useAlertWebSocket()` internamente, uma fonte 100% independente do tick
 * SSE de 1Hz que re-renderiza o `FleetDashboard` pai. Sem memo, ainda assim
 * re-renderizava a cada tick SSE (medido com React Profiler — ver
 * frontend_performance_report.md) mesmo sem nenhuma prop para justificar.
 */
const EventFeedCard = memo(function EventFeedCard() {
  const { alerts } = useAlertWebSocket();
  // Horários e "há N min" dependem do relógio e do fuso do navegador: o
  // servidor (UTC) não os renderiza — antes o HTML do SSR ficava na tela
  // com os horários 3 h à frente.
  const isClient = useIsClient();

  const events: FeedEvent[] = useMemo(() => {
    const now = Date.now();

    const liveEvents: FeedEvent[] = alerts
      .slice(0, MAX_FEED_EVENTS)
      .map((a, i) => {
        const risk = toRisk(a.probability, a.decision_threshold);
        return {
          // ID estável por message_id; index só desempata colisões na render.
          id: `live-${a.message_id ?? a.timestamp ?? i}`,
          assetId: "APU-Trem-042",
          message: `Anomalia detectada (risco ${formatRisk(risk)})`,
          ts: a.timestamp ? new Date(a.timestamp).getTime() : now - i * 1000,
          tone: "warn",
          risk,
        };
      });

    const demoEvents: FeedEvent[] = DEMO_EVENTS.map((e) => ({
      ...e,
      ts: now - e.ageMin * 60 * 1000,
    }));

    return [...liveEvents, ...demoEvents]
      .sort((a, b) => b.ts - a.ts)
      .slice(0, VISIBLE_FEED_EVENTS);
  }, [alerts]);

  return (
    <Card className="flex w-full shrink-0 flex-col border border-border bg-card shadow-sm ring-0 short:gap-2 short:py-2">
      <CardHeader className="px-5 pb-2 pt-4 short:pb-1 short:pt-2">
        <CardTitle className="flex items-center justify-between text-sm font-semibold text-foreground">
          <span className="flex items-center gap-2">
            <Radio className="h-4 w-4 text-muted-foreground" />
            Eventos Recentes
          </span>
          <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground/70">
            {events.length} eventos · log
          </span>
        </CardTitle>
      </CardHeader>

      {/* Lista compacta tipo SCADA: alta densidade, severidade pela borda
          esquerda, todos os campos numa única linha quando cabe. */}
      <CardContent className="px-2 pb-2">
        {/* Altura de 5 linhas (~29 px cada): o card não força scroll na
            página; com mais eventos (alertas ao vivo) a lista rola por dentro. */}
        <ScrollArea className="h-[148px] pr-2">
          <ul className="flex flex-col">
            {isClient
              ? events.map((evt, index) => (
                  // Composto com index: blinda contra colisões de id quando o
                  // backend dispara eventos em rajada com timestamps idênticos.
                  <FeedEventRow key={`${evt.id}-${index}`} evt={evt} />
                ))
              : Array.from({ length: 5 }, (_, i) => (
                  <li key={i} className="px-3 py-1.5">
                    <Skeleton className="h-4 w-full" />
                  </li>
                ))}
          </ul>
        </ScrollArea>
      </CardContent>
    </Card>
  );
});

export default EventFeedCard;
