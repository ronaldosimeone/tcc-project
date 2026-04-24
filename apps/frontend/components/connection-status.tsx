"use client";

/**
 * ConnectionStatus — RF-17.
 *
 * Dois indicadores de saúde de conexão em tempo real:
 *   • Sensores — canal SSE (/api/stream/sensors)
 *   • Alertas  — canal WebSocket (/ws/alerts)
 *
 * Estados: Online (verde), Reconectando... (âmbar pulsante), Offline (vermelho).
 */

import { cn } from "@/lib/utils";
import type { SSEStatus } from "@/hooks/use-sse";
import type { WsStatus } from "@/hooks/use-alert-websocket";

// ── Tipos internos ────────────────────────────────────────────────────────

type ChannelState = "online" | "reconnecting" | "offline";

function sseToState(s: SSEStatus): ChannelState {
  if (s === "connected") return "online";
  if (s === "reconnecting") return "reconnecting";
  return "offline";
}

function wsToState(s: WsStatus): ChannelState {
  if (s === "open") return "online";
  if (s === "connecting" || s === "reconnecting") return "reconnecting";
  return "offline";
}

const CONFIG: Record<
  ChannelState,
  { dot: string; label: string; text: string }
> = {
  online: {
    dot: "bg-green-500",
    label: "Online",
    text: "text-green-400",
  },
  reconnecting: {
    dot: "bg-amber-500 animate-pulse",
    label: "Reconectando...",
    text: "text-amber-400",
  },
  offline: {
    dot: "bg-red-500",
    label: "Offline",
    text: "text-red-400",
  },
};

// ── Pill individual de canal ──────────────────────────────────────────────

interface ChannelPillProps {
  name: string;
  state: ChannelState;
}

function ChannelPill({ name, state }: ChannelPillProps) {
  const { dot, label, text } = CONFIG[state];
  return (
    <div
      className="flex items-center gap-1.5"
      aria-label={`${name}: ${label}`}
      title={`${name}: ${label}`}
    >
      <span
        className={cn("h-1.5 w-1.5 rounded-full", dot)}
        aria-hidden="true"
      />
      <span className="text-[10px] font-medium text-muted-foreground">
        {name}
      </span>
      <span className={cn("text-[10px] font-semibold tabular-nums", text)}>
        {label}
      </span>
    </div>
  );
}

// ── Componente público ────────────────────────────────────────────────────

export interface ConnectionStatusProps {
  sseStatus: SSEStatus;
  wsStatus: WsStatus;
}

export function ConnectionStatus({
  sseStatus,
  wsStatus,
}: ConnectionStatusProps) {
  return (
    <div
      data-testid="connection-status"
      aria-label="Status das conexões em tempo real"
      className="flex items-center gap-3 rounded-lg border border-border/50 bg-card/40 px-3 py-1.5"
    >
      <ChannelPill name="Sensores" state={sseToState(sseStatus)} />
      <div className="h-3 w-px bg-border/60" aria-hidden="true" />
      <ChannelPill name="Alertas" state={wsToState(wsStatus)} />
    </div>
  );
}
