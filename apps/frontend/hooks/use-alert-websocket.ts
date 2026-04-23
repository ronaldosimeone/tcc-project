"use client";

/**
 * Hook de alertas em tempo real via WebSocket — RF-14 / RNF-30.
 *
 * RF-14:  Recebe alertas push do backend quando probability > 0.70.
 * RNF-30: Responde ao heartbeat (ping → pong) para manter a conexão viva.
 *
 * Roteamento: ws(s)://<host>/ws/alerts  →  nginx  →  api:8000/ws/alerts
 * O schema (ws/wss) é derivado do protocolo da página para funcionar em
 * HTTP (dev) e HTTPS (prod) sem URL hardcoded.
 */

import { useCallback, useEffect, useRef, useState } from "react";

// ── Tipos do protocolo ────────────────────────────────────────────────────

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

// ── Tipos públicos ────────────────────────────────────────────────────────

export interface WsAlert {
  message_id: string;
  probability: number;
  predicted_class: number;
  timestamp: string;
}

export type WsStatus = "connecting" | "open" | "closed" | "error";

export interface UseAlertWebSocketReturn {
  alerts: WsAlert[];
  status: WsStatus;
  clearAlerts: () => void;
}

// ── Helpers internos ──────────────────────────────────────────────────────

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
  const wsRef = useRef<WebSocket | null>(null);

  const clearAlerts = useCallback(() => setAlerts([]), []);

  useEffect(() => {
    const ws = new WebSocket(buildWsUrl());
    wsRef.current = ws;

    ws.onopen = () => setStatus("open");
    ws.onerror = () => setStatus("error");
    ws.onclose = () => setStatus("closed");

    ws.onmessage = ({ data }: MessageEvent<string>) => {
      const frame = parseFrame(data);
      if (!frame) return;

      if (frame.type === "ping") {
        ws.send(JSON.stringify({ type: "pong" }));
        return;
      }

      if (frame.type === "alert") {
        ws.send(JSON.stringify({ type: "ack", message_id: frame.message_id }));
        setAlerts((prev) =>
          [
            {
              message_id: frame.message_id,
              probability: frame.probability,
              predicted_class: frame.predicted_class,
              timestamp: frame.timestamp,
            },
            ...prev,
          ].slice(0, 50),
        );
      }
    };

    return () => ws.close();
  }, []);

  return { alerts, status, clearAlerts };
}
