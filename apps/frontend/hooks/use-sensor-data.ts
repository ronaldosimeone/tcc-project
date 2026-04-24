"use client";

/**
 * useSensorData — RF-06 / RF-07 / RF-14 / RF-15.
 *
 * Arquitetura de dados:
 *  1. Seed (mount): GET /api/v1/predictions?size=30 — popula histórico inicial.
 *  2. SSE (contínuo): /api/stream/sensors @1Hz — atualiza currentPayload e
 *     janela deslizante de até 60 pontos (sensor data pura, sem predição).
 *  3. Prediction poll (5s): GET /api/v1/predictions?size=1 — atualiza
 *     failure_probability e predicted_class usados no gauge e no riskLevel.
 *  4. WebSocket (RF-14): push imediato quando probabilidade > 0.70.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { PredictPayload, PredictResponse } from "@/lib/api-client";
import { useSSE } from "@/hooks/use-sse";
import type { SSEStatus } from "@/hooks/use-sse";

// ── Constantes exportadas ─────────────────────────────────────────────────
// POLL_INTERVAL_MS representa o intervalo do prediction poll (5 s).
// Mantido como export para compatibilidade com testes existentes.
export const POLL_INTERVAL_MS = 5_000;

const HISTORY_MAX = 60; // janela deslizante: 60 s @ 1 Hz
const SSE_URL = "/api/stream/sensors";

// ── Tipos públicos ────────────────────────────────────────────────────────

export type RiskLevel = "NORMAL" | "ALERTA" | "CRÍTICO";

export interface SensorDataPoint {
  time: string;
  TP2: number;
  TP3: number;
  Motor_current: number;
  Oil_temperature: number;
  failure_probability: number;
  predicted_class: number;
}

export interface SensorDataState {
  history: SensorDataPoint[];
  latest: PredictResponse | null;
  currentPayload: PredictPayload;
  isLoading: boolean;
  isAnomaly: boolean;
  riskLevel: RiskLevel;
  error: Error | null;
  sseStatus: SSEStatus;
  sseReconnectAttempt: number;
}

// ── Utilitários exportados para testes ───────────────────────────────────

export function getRiskLevel(prob: number): RiskLevel {
  if (prob < 0.3) return "NORMAL";
  if (prob < 0.65) return "ALERTA";
  return "CRÍTICO";
}

// ── Tipos internos ────────────────────────────────────────────────────────

interface SensorReading {
  timestamp: string;
  TP2: number;
  TP3: number;
  H1: number;
  DV_pressure: number;
  Reservoirs: number;
  Motor_current: number;
  Oil_temperature: number;
  COMP: number;
  DV_eletric: number;
  Towers: number;
  MPG: number;
  Oil_level: number;
}

interface HistoryItem extends SensorReading {
  failure_probability: number;
  predicted_class: number;
}

interface HistoryPage {
  items: HistoryItem[];
  total: number;
  page: number;
  size: number;
  pages: number;
}

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

type WsFrame = WsAlertFrame | WsPingFrame;

// ── Helpers ───────────────────────────────────────────────────────────────

function toTimeLabel(ts?: string): string {
  const d = ts ? new Date(ts) : new Date();
  return d.toLocaleTimeString("pt-BR", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function sseReadingToPoint(
  r: SensorReading,
  prob: number,
  cls: number,
): SensorDataPoint {
  return {
    time: toTimeLabel(r.timestamp),
    TP2: parseFloat(r.TP2.toFixed(3)),
    TP3: parseFloat(r.TP3.toFixed(3)),
    Motor_current: parseFloat(r.Motor_current.toFixed(3)),
    Oil_temperature: parseFloat(r.Oil_temperature.toFixed(2)),
    failure_probability: prob,
    predicted_class: cls,
  };
}

function historyItemToPoint(item: HistoryItem): SensorDataPoint {
  return {
    time: toTimeLabel(item.timestamp),
    TP2: parseFloat(item.TP2.toFixed(3)),
    TP3: parseFloat(item.TP3.toFixed(3)),
    Motor_current: parseFloat(item.Motor_current.toFixed(3)),
    Oil_temperature: parseFloat(item.Oil_temperature.toFixed(2)),
    failure_probability: parseFloat(item.failure_probability.toFixed(4)),
    predicted_class: item.predicted_class,
  };
}

function readingToPayload(r: SensorReading): PredictPayload {
  return {
    TP2: r.TP2,
    TP3: r.TP3,
    H1: r.H1,
    DV_pressure: r.DV_pressure,
    Reservoirs: r.Reservoirs,
    Motor_current: r.Motor_current,
    Oil_temperature: r.Oil_temperature,
    COMP: r.COMP,
    DV_eletric: r.DV_eletric,
    Towers: r.Towers,
    MPG: r.MPG,
    Oil_level: r.Oil_level,
  };
}

const EMPTY_PAYLOAD: PredictPayload = {
  TP2: 0,
  TP3: 0,
  H1: 0,
  DV_pressure: 0,
  Reservoirs: 0,
  Motor_current: 0,
  Oil_temperature: 0,
  COMP: 0,
  DV_eletric: 0,
  Towers: 0,
  MPG: 0,
  Oil_level: 0,
};

// ── Hook ──────────────────────────────────────────────────────────────────

export function useSensorData(): SensorDataState {
  const [history, setHistory] = useState<SensorDataPoint[]>([]);
  const [latest, setLatest] = useState<PredictResponse | null>(null);
  const [currentPayload, setCurrentPayload] =
    useState<PredictPayload>(EMPTY_PAYLOAD);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  // Refs de predição — atualizados pelo prediction poll e WS sem re-render
  const latestProbRef = useRef(0);
  const latestClassRef = useRef(0);

  // Refs de deduplicação e contexto para WS
  const lastTimestampRef = useRef<string | null>(null);
  const lastSensorRef = useRef<SensorReading | null>(null);

  // ── SSE: leituras em tempo real (1 Hz) ──────────────────────────────────
  const handleSensorReading = useCallback((reading: SensorReading): void => {
    // Deduplicação por timestamp — descarta eventos repetidos
    if (reading.timestamp === lastTimestampRef.current) return;
    lastTimestampRef.current = reading.timestamp;
    lastSensorRef.current = reading;

    const point = sseReadingToPoint(
      reading,
      latestProbRef.current,
      latestClassRef.current,
    );

    setCurrentPayload(readingToPayload(reading));
    setHistory((prev) => {
      const next = [...prev, point];
      return next.length > HISTORY_MAX ? next.slice(-HISTORY_MAX) : next;
    });
    // Limpa error quando SSE começa a entregar dados (recuperação silenciosa)
    setError(null);
    setIsLoading(false);
  }, []);

  const { status: sseStatus, reconnectAttempt: sseReconnectAttempt } =
    useSSE<SensorReading>({
      url: SSE_URL,
      eventName: "sensor_reading",
      onMessage: handleSensorReading,
    });

  // ── Seed: histórico inicial ao montar ─────────────────────────────────────
  useEffect(() => {
    let cancelled = false;

    void (async () => {
      try {
        const res = await fetch("/api/v1/predictions?page=1&size=30", {
          cache: "no-store",
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const page = (await res.json()) as HistoryPage;
        if (cancelled) return;

        if (page.items.length === 0) {
          setIsLoading(false);
          return;
        }

        const newest = page.items[0];
        latestProbRef.current = newest.failure_probability;
        latestClassRef.current = newest.predicted_class;

        setLatest({
          failure_probability: newest.failure_probability,
          predicted_class: newest.predicted_class,
          timestamp: newest.timestamp,
        });

        const points = [...page.items].reverse().map(historyItemToPoint);
        setHistory(points.slice(-HISTORY_MAX));
        setError(null);
        setIsLoading(false);
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e : new Error(String(e)));
          setIsLoading(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  // ── Prediction poll: atualiza failure_probability a cada 5 s ─────────────
  useEffect(() => {
    const poll = async (): Promise<void> => {
      try {
        const res = await fetch("/api/v1/predictions?page=1&size=1", {
          cache: "no-store",
        });
        if (!res.ok) return;

        const page = (await res.json()) as HistoryPage;
        if (page.items.length === 0) return;

        const newest = page.items[0];
        latestProbRef.current = newest.failure_probability;
        latestClassRef.current = newest.predicted_class;

        setLatest({
          failure_probability: newest.failure_probability,
          predicted_class: newest.predicted_class,
          timestamp: newest.timestamp,
        });
        setError(null);
      } catch {
        // erros no poll são silenciosos — o SSE fornece o sinal de conectividade
      }
    };

    const id = setInterval(() => void poll(), POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, []);

  // ── WebSocket: alertas push do backend quando probability > 0.70 (RF-14) ─
  useEffect(() => {
    if (typeof window === "undefined") return;

    const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${scheme}//${window.location.host}/ws/alerts`);

    ws.onmessage = ({ data }: MessageEvent<string>) => {
      let frame: WsFrame;
      try {
        frame = JSON.parse(data) as WsFrame;
      } catch {
        return;
      }

      if (frame.type === "ping") {
        ws.send(JSON.stringify({ type: "pong" }));
        return;
      }

      if (frame.type === "alert") {
        ws.send(JSON.stringify({ type: "ack", message_id: frame.message_id }));

        latestProbRef.current = frame.probability;
        latestClassRef.current = frame.predicted_class;

        setLatest({
          failure_probability: frame.probability,
          predicted_class: frame.predicted_class,
          timestamp: frame.timestamp,
        });
        setIsLoading(false);

        // Usa os últimos valores de sensor reais para o ponto do gráfico
        const sensor = lastSensorRef.current;
        const point: SensorDataPoint = {
          time: toTimeLabel(),
          TP2: parseFloat((sensor?.TP2 ?? 0).toFixed(3)),
          TP3: parseFloat((sensor?.TP3 ?? 0).toFixed(3)),
          Motor_current: parseFloat((sensor?.Motor_current ?? 0).toFixed(3)),
          Oil_temperature: parseFloat(
            (sensor?.Oil_temperature ?? 0).toFixed(2),
          ),
          failure_probability: parseFloat(frame.probability.toFixed(4)),
          predicted_class: frame.predicted_class,
        };

        setHistory((prev) => {
          const next = [...prev, point];
          return next.length > HISTORY_MAX ? next.slice(-HISTORY_MAX) : next;
        });
      }
    };

    return () => ws.close();
  }, []);

  const prob = latest?.failure_probability ?? 0;
  const riskLevel = getRiskLevel(prob);

  return {
    history,
    latest,
    currentPayload,
    isLoading,
    isAnomaly: riskLevel !== "NORMAL",
    riskLevel,
    error,
    sseStatus,
    sseReconnectAttempt,
  };
}
