"use client";

/**
 * useSSE — motor de conexão SSE com reconexão automática (RF-15).
 *
 * Estratégia de backoff:
 *   delay = min(BASE * 2^(attempt-1), MAX) * jitter(0.8-1.2)
 *   BASE=1s, MAX=30s → sequência: ~0.8s, ~1.6s, ~3.2s, …, ~24s, 30s.
 *
 * onMessage é mantido em ref para evitar reconexões desnecessárias quando
 * a função callback é recriada pelo componente pai.
 */

import { useEffect, useRef, useState } from "react";

export type SSEStatus = "connecting" | "connected" | "reconnecting" | "closed";

export interface UseSSEOptions<T> {
  url: string;
  eventName: string;
  onMessage: (data: T) => void;
  enabled?: boolean;
}

export interface UseSSEResult {
  status: SSEStatus;
  reconnectAttempt: number;
}

const BACKOFF_BASE_MS = 1_000;
const BACKOFF_MAX_MS = 30_000;

export function useSSE<T>({
  url,
  eventName,
  onMessage,
  enabled = true,
}: UseSSEOptions<T>): UseSSEResult {
  const [status, setStatus] = useState<SSEStatus>("connecting");
  const [reconnectAttempt, setReconnectAttempt] = useState(0);

  // Sempre chama a versão mais recente de onMessage sem recriar o EventSource
  const onMessageRef = useRef(onMessage);
  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);

  // Estado mutável da sessão — não dispara re-renders
  const session = useRef({
    attempt: 0,
    timer: null as ReturnType<typeof setTimeout> | null,
    es: null as EventSource | null,
    destroyed: false,
  });

  useEffect(() => {
    if (typeof window === "undefined" || !enabled) {
      setTimeout(() => setStatus("closed"), 0);
      return;
    }

    const s = session.current;
    s.destroyed = false;
    s.attempt = 0;

    function connect(): void {
      if (s.destroyed) return;

      setStatus(s.attempt > 0 ? "reconnecting" : "connecting");

      const es = new EventSource(url);
      s.es = es;

      es.addEventListener(eventName, (e: Event) => {
        const msg = e as MessageEvent<string>;
        try {
          onMessageRef.current(JSON.parse(msg.data) as T);
          if (s.attempt !== 0) {
            s.attempt = 0;
            setReconnectAttempt(0);
          }
          setStatus("connected");
        } catch {
          // ignora JSON malformado
        }
      });

      es.onerror = () => {
        es.close();
        s.es = null;
        if (s.destroyed) return;

        s.attempt += 1;
        setReconnectAttempt(s.attempt);
        setStatus("reconnecting");

        const backoff = Math.min(
          BACKOFF_BASE_MS * 2 ** (s.attempt - 1),
          BACKOFF_MAX_MS,
        );
        // Jitter ±20% para evitar thundering herd
        const delay = backoff * (0.8 + Math.random() * 0.4);
        s.timer = setTimeout(connect, delay);
      };
    }

    connect();

    return () => {
      s.destroyed = true;
      s.es?.close();
      s.es = null;
      if (s.timer !== null) {
        clearTimeout(s.timer);
        s.timer = null;
      }
    };
  }, [url, eventName, enabled]);

  return { status, reconnectAttempt };
}
