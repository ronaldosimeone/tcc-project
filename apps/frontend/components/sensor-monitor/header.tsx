"use client";

// ── Cabeçalho imersivo — RNF-58: extraído de sensor-monitor.tsx ─────────────
//
// CRÍTICO  → fundo vermelho suave + borda vermelha (atenção máxima)
// ALERTA   → fundo âmbar suave + borda âmbar (atenção intermediária)
// NORMAL   → branco neutro
//
// `data-testid="alert-panel"`/`data-risk` e `data-testid="critical-banner"`/
// `role="alert"` (RF-08) preservados EXATAMENTE — ver "fix de integração
// MSW/Playwright do Dashboard": failure_alert.spec.ts/dashboard_flow.spec.ts
// dependem desses atributos neste elemento específico.

import { AlertTriangle, CheckCircle2, WifiOff } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ConnectionStatus } from "@/components/connection-status";
import type { SSEStatus } from "@/hooks/use-sse";
import type { RiskLevel } from "@/hooks/use-sensor-data";
import type { WsStatus } from "@/hooks/use-alert-websocket";
import type { ErrorRateStatus } from "@/lib/api-client";
import { cn } from "@/lib/utils";

export interface SensorMonitorHeaderProps {
  riskLevel: RiskLevel;
  isCriticalState: boolean;
  isAlertState: boolean;
  isLive: boolean;
  isLoading: boolean;
  isOffline: boolean;
  sseStatus: SSEStatus;
  wsStatus: WsStatus;
  /** RNF-77 — `null` enquanto o primeiro poll não resolveu (ver
   * useErrorRateStatus); ConnectionStatus omite o pill nesse caso. */
  errorRateStatus?: ErrorRateStatus | null;
}

export function SensorMonitorHeader({
  riskLevel,
  isCriticalState,
  isAlertState,
  isLive,
  isLoading,
  isOffline,
  sseStatus,
  wsStatus,
  errorRateStatus,
}: SensorMonitorHeaderProps) {
  return (
    <div
      data-testid="alert-panel"
      data-risk={riskLevel}
      className={cn(
        "flex flex-wrap items-center justify-between gap-3 rounded-lg border px-4 py-3 transition-colors",
        isCriticalState
          ? "border-red-500 bg-red-50"
          : isAlertState
          ? "border-amber-500 bg-amber-50"
          : "border-slate-200 bg-white",
      )}
    >
      <div>
        <h1
          className={cn(
            "text-xl font-bold tracking-tight",
            isCriticalState
              ? "text-red-900"
              : isAlertState
              ? "text-amber-900"
              : "text-slate-900",
          )}
        >
          APU-Trem-042
        </h1>
        <p
          className={cn(
            "text-sm",
            isCriticalState
              ? "text-red-700"
              : isAlertState
              ? "text-amber-700"
              : "text-slate-500",
          )}
        >
          Compressor MetroPT-3 ·{" "}
          {isLive
            ? "Streaming em tempo real"
            : sseStatus === "reconnecting"
            ? "Reconectando…"
            : "Conectando…"}
        </p>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <ConnectionStatus
          sseStatus={sseStatus}
          wsStatus={wsStatus}
          errorRateStatus={errorRateStatus}
        />
        {isOffline && (
          <Badge
            variant="outline"
            className="gap-1.5 border-destructive/40 bg-destructive/10 text-destructive"
          >
            <WifiOff className="h-3 w-3" />
            Backend offline
          </Badge>
        )}
        {!isLoading && (
          <Badge
            variant="outline"
            data-testid={isCriticalState ? "critical-banner" : undefined}
            role={isCriticalState ? "alert" : undefined}
            className={cn(
              "gap-1.5 font-semibold",
              isCriticalState
                ? "animate-pulse border-red-500 bg-red-600 text-white"
                : isAlertState
                ? "animate-pulse border-amber-400 bg-amber-100 text-amber-800"
                : "border-emerald-400/40 bg-emerald-50 text-emerald-700",
            )}
          >
            {isCriticalState ? (
              <>
                <span className="relative flex h-2 w-2">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-white opacity-75" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-white" />
                </span>
                FALHA CRÍTICA DETECTADA
              </>
            ) : isAlertState ? (
              <>
                <AlertTriangle className="h-3.5 w-3.5" />
                ALERTA
              </>
            ) : (
              <>
                <CheckCircle2 className="h-3.5 w-3.5" />
                OPERACIONAL
              </>
            )}
          </Badge>
        )}
      </div>
    </div>
  );
}
