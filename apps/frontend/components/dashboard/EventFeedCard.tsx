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
import { useAlertWebSocket } from "@/hooks/use-alert-websocket";
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
            {events.map((evt, index) => (
              // Composto com index: blinda contra colisões de id quando o
              // backend dispara eventos em rajada com timestamps idênticos.
              <FeedEventRow key={`${evt.id}-${index}`} evt={evt} />
            ))}
          </ul>
        </ScrollArea>
      </CardContent>
    </Card>
  );
});

export default EventFeedCard;
