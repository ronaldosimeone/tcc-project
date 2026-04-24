"use client";

/**
 * useAlertWebSocket — RF-14 / RF-16 / RNF-30 / RNF-34.
 *
 * RF-14:  Recebe alertas push quando probability > 0.70.
 * RF-16:  Expõe acknowledge(id) para dismissal de alerta pelo usuário.
 * RNF-30: Responde a ping com pong para manter a conexão viva.
 * RNF-34: Fila máxima de QUEUE_MAX (5) alertas. FIFO — o mais antigo é
 *          descartado quando um 6º chega.
 *
 * Reconexão: exponential backoff idêntico ao useSSE (1s a 30s + jitter ±20%).
 * Schema WS: derivado de window.location.protocol para funcionar em HTTP e HTTPS.
 */

import { useCallback, useEffect, useRef, useState } from "react";

// ── Tipos do protocolo ─────────────────────────────────────────────────────

interface WsAlertFrame {
  type: "alert";
  message_id: string;
  probability: number;
  predicted_class: number;
  timestamp: string;
}

interface WsPingFrame {
  type: "ping";
}

type WsServerFrame = WsAlertFrame | WsPingFrame;

// ── Tipos públicos ─────────────────────────────────────────────────────────

export interface WsAlert {
  message_id: string;
  probability: number;
  predicted_class: number;
  timestamp: string;
  receivedAt: number;
}

export type WsStatus =
  | "connecting"
  | "open"
  | "closed"
  | "error"
  | "reconnecting";

export interface UseAlertWebSocketReturn {
  alerts: WsAlert[];
  status: WsStatus;
  acknowledge: (id: string) => void;
  clearAlerts: () => void;
}

// ── Constantes ─────────────────────────────────────────────────────────────

const QUEUE_MAX = 5;
const BACKOFF_BASE_MS = 1_000;
const BACKOFF_MAX_MS = 30_000;

// ── Helpers internos ───────────────────────────────────────────────────────

function buildWsUrl(): string {
  const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${window.location.host}/ws/alerts`;
}

function parseFrame(raw: string): WsServerFrame | null {
  try {
    return JSON.parse(raw) as WsServerFrame;
  } catch {
    return null;
  }
}

// ── Hook ──────────────────────────────────────────────────────────────────

export function useAlertWebSocket(): UseAlertWebSocketReturn {
  const [alerts, setAlerts] = useState<WsAlert[]>([]);
  const [status, setStatus] = useState<WsStatus>("connecting");

  // Estado mutável da sessão — não dispara re-renders
  const session = useRef({
    attempt: 0,
    timer: null as ReturnType<typeof setTimeout> | null,
    ws: null as WebSocket | null,
    destroyed: false,
  });

  const acknowledge = useCallback((id: string) => {
    setAlerts((prev) => prev.filter((a) => a.message_id !== id));
  }, []);

  const clearAlerts = useCallback(() => setAlerts([]), []);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const s = session.current;
    s.destroyed = false;
    s.attempt = 0;

    function connect(): void {
      if (s.destroyed) return;

      setStatus(s.attempt > 0 ? "reconnecting" : "connecting");

      const ws = new WebSocket(buildWsUrl());
      s.ws = ws;

      ws.onopen = () => {
        s.attempt = 0;
        setStatus("open");
      };

      ws.onerror = () => {
        setStatus("error");
      };

      ws.onclose = () => {
        s.ws = null;
        if (s.destroyed) return;

        s.attempt += 1;
        setStatus("reconnecting");

        const backoff = Math.min(
          BACKOFF_BASE_MS * 2 ** (s.attempt - 1),
          BACKOFF_MAX_MS,
        );
        // Jitter ±20% para evitar thundering herd
        const delay = backoff * (0.8 + Math.random() * 0.4);
        s.timer = setTimeout(connect, delay);
      };

      ws.onmessage = ({ data }: MessageEvent<string>) => {
        const frame = parseFrame(data);
        if (!frame) return;

        if (frame.type === "ping") {
          ws.send(JSON.stringify({ type: "pong" }));
          return;
        }

        if (frame.type === "alert") {
          ws.send(
            JSON.stringify({ type: "ack", message_id: frame.message_id }),
          );
          setAlerts((prev) => {
            const next: WsAlert[] = [
              {
                message_id: frame.message_id,
                probability: frame.probability,
                predicted_class: frame.predicted_class,
                timestamp: frame.timestamp,
                receivedAt: Date.now(),
              },
              ...prev,
            ];
            // FIFO: mantém os QUEUE_MAX mais recentes; descarta o(s) mais antigo(s)
            return next.slice(0, QUEUE_MAX);
          });
        }
      };
    }

    connect();

    return () => {
      s.destroyed = true;
      s.ws?.close();
      s.ws = null;
      if (s.timer !== null) {
        clearTimeout(s.timer);
        s.timer = null;
      }
    };
  }, []);

  return { alerts, status, acknowledge, clearAlerts };
}
