// ── Linha de evento — RNF-58: extraído de EventLogTable ─────────────────────

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { HistoryEvent } from "@/lib/history-mock";
import { EVENT_TYPE_CONFIG, SEVERITY_CONFIG } from "./constants";
import { formatTimestamp } from "./helpers";

interface EventRowProps {
  event: HistoryEvent;
  onSelectEvent?: (event: HistoryEvent) => void;
}

export function EventRow({ event, onSelectEvent }: EventRowProps) {
  const sevCfg = SEVERITY_CONFIG[event.severity];
  const typeCfg = EVENT_TYPE_CONFIG[event.type];
  const SevIcon = sevCfg.icon;
  const TypeIcon = typeCfg.icon;

  return (
    <tr
      onClick={() => onSelectEvent?.(event)}
      className={cn(
        "group/row border-b border-slate-100 transition-colors duration-150 last:border-0",
        onSelectEvent
          ? "cursor-pointer hover:bg-slate-50"
          : "hover:bg-muted/20",
      )}
    >
      {/* Timestamp */}
      <td className="py-3 pr-5">
        {onSelectEvent ? (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onSelectEvent(event);
            }}
            aria-label={`${formatTimestamp(
              event.timestamp,
            )}: ver detalhes do evento de ${event.type} em ${
              event.equipment
            }, severidade ${event.severity}`}
            className="whitespace-nowrap rounded-sm font-mono text-xs tabular-nums text-muted-foreground underline-offset-2 outline-none hover:underline focus-visible:ring-3 focus-visible:ring-ring/50"
          >
            {formatTimestamp(event.timestamp)}
          </button>
        ) : (
          <span className="whitespace-nowrap font-mono text-xs tabular-nums text-muted-foreground">
            {formatTimestamp(event.timestamp)}
          </span>
        )}
      </td>

      {/* Equipment */}
      <td className="py-3 pr-5">
        <span className="font-mono text-xs font-medium text-foreground">
          {event.equipment}
        </span>
      </td>

      {/* Type */}
      <td className="py-3 pr-5">
        <div
          className={cn(
            "flex items-center gap-1.5 text-xs font-medium",
            typeCfg.className,
          )}
        >
          <TypeIcon className="h-3.5 w-3.5 shrink-0" />
          {event.type}
        </div>
      </td>

      {/* Severity */}
      <td className="py-3 pr-5">
        <Badge
          variant="outline"
          className={cn("gap-1 text-[11px] font-semibold", sevCfg.className)}
        >
          <SevIcon className="h-3 w-3" />
          {event.severity}
        </Badge>
      </td>

      {/* Duration */}
      <td className="py-3 pr-5">
        <span className="whitespace-nowrap font-mono text-xs tabular-nums text-muted-foreground">
          {event.duration}
        </span>
      </td>

      {/* Description */}
      <td className="max-w-sm py-3">
        <span className="line-clamp-2 text-xs leading-relaxed text-foreground/70 transition-colors group-hover/row:text-foreground/90">
          {event.description}
        </span>
      </td>
    </tr>
  );
}
