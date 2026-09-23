"use client";

/**
 * ConnectionStatus — RF-17 / RNF-77.
 *
 * Até três indicadores de saúde em tempo real:
 *   • Sensores — canal SSE (/api/stream/sensors)
 *   • Alertas  — canal WebSocket (/ws/alerts)
 *   • API      — taxa de erro HTTP 5xx (RNF-77), opcional — só aparece
 *                quando `errorRateStatus` é passado (omitido enquanto o
 *                primeiro poll não resolve, ver useErrorRateStatus)
 *
 * Sensores/Alertas: Online (verde), Reconectando... (âmbar pulsante),
 * Offline (vermelho). API reaproveita as MESMAS cores com rótulos próprios
 * (Normal/Atenção/Crítico) — eixo de taxa de erro, não de conectividade.
 */

import { cn } from "@/lib/utils";
import type { SSEStatus } from "@/hooks/use-sse";
import type { WsStatus } from "@/hooks/use-alert-websocket";
import type { ErrorRateStatus } from "@/lib/api-client";

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

/** RNF-77 — reaproveita o MESMO visual verde/âmbar/vermelho já usado para
 * SSE/WS acima: NORMAL≈online, WARNING≈reconnecting, CRITICAL≈offline.
 * Eixo diferente (taxa de erro da API, não conectividade), mas o padrão de
 * 3 estados já existente é exatamente o que o design system pede reusar
 * (RNF-77 Fase 8) — nenhum componente novo necessário para o pill em si. */
function errorRateToState(s: ErrorRateStatus): ChannelState {
  if (s === "NORMAL") return "online";
  if (s === "WARNING") return "reconnecting";
  return "offline";
}

/** Rótulos próprios do eixo de taxa de erro — "Reconectando.../Offline" do
 * CONFIG abaixo descreveriam conectividade, não taxa de erro; a cor/dot
 * continua 100% reaproveitada do CONFIG, só o texto muda via `label`. */
const ERROR_RATE_LABEL: Record<ChannelState, string> = {
  online: "Normal",
  reconnecting: "Atenção",
  offline: "Crítico",
};

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
  /** Sobrescreve o texto do CONFIG (dot/cor continuam vindo de lá) — usado
   * pelo pill de taxa de erro, cujos rótulos são "Normal/Atenção/Crítico". */
  labelOverride?: string;
}

function ChannelPill({ name, state, labelOverride }: ChannelPillProps) {
  const { dot, label: defaultLabel, text } = CONFIG[state];
  const label = labelOverride ?? defaultLabel;
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
  /** RNF-77 — omitido enquanto o primeiro poll de error-rate não resolveu
   * (`null`, ver useErrorRateStatus): o pill não é renderizado até ter um
   * status real, para nunca mostrar um estado inventado/placeholder. */
  errorRateStatus?: ErrorRateStatus | null;
}

export function ConnectionStatus({
  sseStatus,
  wsStatus,
  errorRateStatus,
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
      {errorRateStatus && (
        <>
          <div className="h-3 w-px bg-border/60" aria-hidden="true" />
          <ChannelPill
            name="API"
            state={errorRateToState(errorRateStatus)}
            labelOverride={ERROR_RATE_LABEL[errorRateToState(errorRateStatus)]}
          />
        </>
      )}
    </div>
  );
}
