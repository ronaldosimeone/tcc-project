// ── Linha de evento — RNF-58: extraído de EventFeedCard ─────────────────────

import { cn } from "@/lib/utils";
import {
  CRITICAL_BORDER,
  TONE_BORDER,
  TONE_COLOR,
  TONE_ICON,
  explanationFor,
  formatClockTime,
  fromNow,
  isCriticalEvent,
  type FeedEvent,
} from "./constants-and-helpers";

interface FeedEventRowProps {
  evt: FeedEvent;
}

export function FeedEventRow({ evt }: FeedEventRowProps) {
  const Icon = TONE_ICON[evt.tone];
  const critical = isCriticalEvent(evt);
  const borderTone = critical ? CRITICAL_BORDER : TONE_BORDER[evt.tone];

  return (
    <li
      className={cn(
        "flex items-center gap-2.5 border-b border-l-4 border-border/60 px-3 py-1.5 text-xs last:border-b-0 hover:bg-muted/60",
        borderTone,
      )}
    >
      {/* Horário monoespaçado — referência temporal precisa. Só é
          renderizado no cliente (ver EventFeedCard), no fuso local. */}
      <span className="shrink-0 font-mono text-xs tabular-nums text-muted-foreground">
        {formatClockTime(evt.ts)}
      </span>

      {/* Ícone de severidade — minúsculo, sem fundo. */}
      <Icon
        className={cn("h-3 w-3 shrink-0", TONE_COLOR[evt.tone])}
        aria-hidden="true"
      />

      {/* Asset ID — fixo em mono para alinhamento vertical na coluna. */}
      <span className="shrink-0 font-mono text-xs font-semibold text-foreground/80">
        {evt.assetId}
      </span>

      {/* Mensagem + explicabilidade inline, ambos truncam se necessário. */}
      <span className="flex min-w-0 flex-1 items-baseline gap-1.5">
        <span className="truncate text-foreground">{evt.message}</span>
        {critical && (
          <span className="hidden truncate text-xs text-muted-foreground lg:inline">
            <span className="text-muted-foreground/70">·</span>{" "}
            {explanationFor(evt.assetId)}
          </span>
        )}
      </span>

      {/* Tempo relativo à direita — opcional, dispensável em viewports muito
          estreitos. suppressHydrationWarning porque `fromNow` depende de
          `Date.now()`, calculado só no cliente. */}
      <span className="hidden shrink-0 text-xs text-muted-foreground/70 sm:inline">
        {fromNow(evt.ts)}
      </span>
    </li>
  );
}
