"use client";

/**
 * AlertToastQueue — RF-16 / RNF-34.
 *
 * Componente de apresentação puro: recebe alerts, status e onAcknowledge
 * como props do SensorMonitor (que detém o useAlertWebSocket singleton).
 *
 * RF-16:  Cada toast exibe tipo, probabilidade e botão 'Reconhecer'.
 * RNF-34: Máximo de 5 toasts simultâneos (garantido pelo hook na camada acima).
 *
 * Posicionamento: fixed bottom-right, sobrepõe todo o conteúdo (z-50).
 * Animação: slide-in-from-right via tw-animate-css.
 * Banner de reconexão: exibido quando status !== "open".
 */

import { AlertTriangle, WifiOff, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { WsAlert, WsStatus } from "@/hooks/use-alert-websocket";

// ── Props ─────────────────────────────────────────────────────────────────

export interface AlertToastQueueProps {
  alerts: WsAlert[];
  status: WsStatus;
  onAcknowledge: (id: string) => void;
}

// ── Banner de status da conexão WS ────────────────────────────────────────

export function ConnectionBanner({ status }: { status: WsStatus }) {
  if (status === "open") return null;

  const isReconnecting = status === "reconnecting" || status === "connecting";

  return (
    <div
      role="status"
      aria-live="polite"
      data-testid="ws-connection-banner"
      className={cn(
        "flex items-center gap-2 rounded-lg border px-3 py-2 text-xs font-medium shadow-xl backdrop-blur-md",
        isReconnecting
          ? "border-amber-400/30 bg-amber-400/15 text-amber-300"
          : "border-red-500/30 bg-red-500/15 text-red-400",
      )}
    >
      <WifiOff className="h-3.5 w-3.5 shrink-0" />
      {isReconnecting ? "Reconectando alertas…" : "Canal de alertas offline"}
    </div>
  );
}

// ── Card individual de alerta ─────────────────────────────────────────────

interface AlertToastProps {
  alert: WsAlert;
  onAcknowledge: (id: string) => void;
}

function AlertToast({ alert, onAcknowledge }: AlertToastProps) {
  const pct = (alert.probability * 100).toFixed(1);
  const isCritical = alert.probability >= 0.65;

  const time = new Date(alert.timestamp).toLocaleTimeString("pt-BR", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  return (
    <div
      role="alert"
      aria-live="assertive"
      aria-atomic="true"
      data-testid="alert-toast"
      data-message-id={alert.message_id}
      className={cn(
        "relative flex w-80 flex-col gap-3 overflow-hidden rounded-xl border p-4 shadow-2xl",
        "animate-in slide-in-from-right-5 fade-in-0 duration-300 ease-out",
        "bg-card/95 backdrop-blur-sm",
        isCritical ? "border-red-500/50" : "border-amber-500/50",
      )}
    >
      {/* Faixa lateral de severidade */}
      <div
        aria-hidden="true"
        className={cn(
          "absolute left-0 top-0 h-full w-1 rounded-l-xl",
          isCritical ? "bg-red-500" : "bg-amber-500",
        )}
      />

      {/* Cabeçalho: tipo + horário */}
      <div className="flex items-start justify-between gap-2 pl-3">
        <div className="flex items-center gap-2">
          <AlertTriangle
            className={cn(
              "h-4 w-4 shrink-0",
              isCritical ? "animate-pulse text-red-400" : "text-amber-400",
            )}
          />
          <span className="text-xs font-bold uppercase tracking-wider text-foreground">
            {isCritical ? "Falha Crítica" : "Degradação Detectada"}
          </span>
        </div>
        <span className="shrink-0 font-mono text-[10px] tabular-nums text-muted-foreground">
          {time}
        </span>
      </div>

      {/* Probabilidade + barra de progresso */}
      <div className="pl-3">
        <div className="mb-1.5 flex items-center justify-between">
          <span className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
            Prob. de Falha
          </span>
          <Badge
            variant="outline"
            className={cn(
              "font-mono text-xs font-bold tabular-nums",
              isCritical
                ? "border-red-500/40 bg-red-500/10 text-red-400"
                : "border-amber-500/40 bg-amber-500/10 text-amber-400",
            )}
          >
            {pct}%
          </Badge>
        </div>
        <div
          className="h-1.5 w-full overflow-hidden rounded-full bg-muted"
          aria-hidden="true"
        >
          <div
            className={cn(
              "h-full rounded-full transition-[width] duration-500",
              isCritical ? "bg-red-500" : "bg-amber-500",
            )}
            style={{ width: `${Math.min(100, alert.probability * 100)}%` }}
          />
        </div>
      </div>

      {/* Botão Reconhecer — RF-16 */}
      <div className="flex justify-end pl-3">
        <Button
          size="sm"
          variant="outline"
          onClick={() => onAcknowledge(alert.message_id)}
          aria-label={`Reconhecer alerta de ${pct}% às ${time}`}
          className={cn(
            "h-7 gap-1.5 px-3 text-[11px] font-semibold transition-colors",
            isCritical
              ? "border-red-500/30 bg-red-500/10 text-red-400 hover:bg-red-500/20 hover:text-red-300"
              : "border-amber-500/30 bg-amber-500/10 text-amber-400 hover:bg-amber-500/20 hover:text-amber-300",
          )}
        >
          <X className="h-3 w-3" />
          Reconhecer
        </Button>
      </div>
    </div>
  );
}

// ── Componente de fila ────────────────────────────────────────────────────

export function AlertToastQueue({
  alerts,
  onAcknowledge,
}: AlertToastQueueProps) {
  const hasAlerts = alerts.length > 0;

  if (!hasAlerts) return null;

  return (
    <div
      className="fixed bottom-6 right-6 z-50 flex flex-col items-end gap-3"
      aria-label="Fila de notificações de alerta"
      data-testid="alert-toast-queue"
    >
      {alerts.map((alert) => (
        <AlertToast
          key={alert.message_id}
          alert={alert}
          onAcknowledge={onAcknowledge}
        />
      ))}
    </div>
  );
}
