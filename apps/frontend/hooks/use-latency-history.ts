"use client";

// ── Hook de histórico de latência — RNF-58: extraído de FleetKPIs ───────────

import { useEffect, useRef, useState } from "react";
import { LATENCY_HISTORY_SIZE } from "@/components/dashboard/fleet-kpis/constants";

/**
 * Telemetria de latência por frame WS. Recebemos um **objeto** (e não um
 * primitivo) propositadamente: cada novo frame WS — mesmo com latência
 * numericamente idêntica ao anterior (ex.: `42 → 42`) — gera uma referência
 * nova, o que faz o `useEffect` disparar. Se passássemos apenas
 * `liveLatency: number`, o React deduplicaria via Object.is e perderíamos
 * leituras consecutivas com o mesmo valor.
 */
export interface LatencyTelemetry {
  /** ID único do frame WS — chave de invalidação do effect. */
  messageId: string;
  /** Latência da inferência em ms. */
  latencyMs: number;
}

interface UseLatencyHistoryResult {
  latencyHistory: number[];
  hasLatency: boolean;
  currentLatencyMs: number | null;
}

/**
 * Mantém uma janela deslizante das últimas `LATENCY_HISTORY_SIZE` leituras de
 * latência. Histórico inicia VAZIO — só passa a reportar valor numérico
 * quando o backend reporta a primeira medição real (evita o "40 ms" fantasma).
 */
export function useLatencyHistory(
  latencyTelemetry: LatencyTelemetry | null,
): UseLatencyHistoryResult {
  const [latencyHistory, setLatencyHistory] = useState<number[]>([]);
  // De-dupe explícito por messageId: garante que cada frame WS gere
  // exactamente uma actualização do histórico, mesmo em StrictMode (que
  // executa effects duas vezes em dev).
  const lastMessageId = useRef<string | null>(null);

  useEffect(() => {
    if (!latencyTelemetry) return;
    if (latencyTelemetry.messageId === lastMessageId.current) return;
    if (!Number.isFinite(latencyTelemetry.latencyMs)) return;

    lastMessageId.current = latencyTelemetry.messageId;
    const sample = Math.max(0, Math.round(latencyTelemetry.latencyMs));
    // Caso canónico de "sincronizar estado com fonte externa" (telemetria
    // WS chega via prop) — sem setState no effect, o histórico nunca
    // atualizaria.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLatencyHistory((prev) => {
      const next = [...prev, sample];
      // Mantém a janela deslizante: descarta o mais antigo se passou do tecto.
      return next.length > LATENCY_HISTORY_SIZE
        ? next.slice(next.length - LATENCY_HISTORY_SIZE)
        : next;
    });
  }, [latencyTelemetry]);

  const hasLatency = latencyHistory.length > 0;
  const currentLatencyMs = hasLatency
    ? latencyHistory[latencyHistory.length - 1]
    : null;

  return { latencyHistory, hasLatency, currentLatencyMs };
}
