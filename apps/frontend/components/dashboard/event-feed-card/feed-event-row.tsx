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
        "flex items-center gap-2.5 border-b border-l-4 border-slate-100 px-3 py-1.5 text-xs last:border-b-0 hover:bg-slate-50/60",
        borderTone,
      )}
    >
      {/* Horário monoespaçado — referência temporal precisa.
          suppressHydrationWarning: o HTML do SSR é renderizado em UTC
          enquanto o cliente formata no fuso local — divergência esperada.
          Após a primeira interação no cliente o valor converge para o
          horário local correto. */}
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
        <span className="truncate text-slate-900">{evt.message}</span>
        {critical && (
          <span className="hidden truncate text-[11px] text-slate-500 lg:inline">
            <span className="text-slate-400">·</span>{" "}
            {explanationFor(evt.assetId)}
          </span>
        )}
      </span>

      {/* Tempo relativo à direita — opcional, dispensável em viewports muito
          estreitos. suppressHydrationWarning porque `fromNow` depende de
          `Date.now()`, que difere entre o instante do render no servidor e
          o instante do paint no cliente. */}
      <span
        suppressHydrationWarning
        className="hidden shrink-0 text-[10px] text-slate-400 sm:inline"
      >
        {fromNow(evt.ts)}
      </span>
    </li>
  );
}
