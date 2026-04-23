"use client";

/**
 * Hook de telemetria — RF-06 / RF-07.
 *
 * Estratégia de dados:
 *  - Polling PASSIVO a cada 5 s: GET /api/v1/predictions?size=1 (RF-06).
 *    Lê o último registro do banco — nunca envia dados sintéticos.
 *  - Primeira chamada busca os últimos 30 registros para seed do gráfico.
 *  - Deduplicação por timestamp: ticks sem novos dados não alteram o estado.
 *  - WebSocket push (RF-14): atualiza imediatamente quando prob > 0.70.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { PredictPayload, PredictResponse } from "@/lib/api-client";

// ── Constantes ────────────────────────────────────────────────────────────

export const POLL_INTERVAL_MS = 5_000;
const HISTORY_MAX = 30;

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
}

// ── Utilitários exportados para testes ───────────────────────────────────

export function getRiskLevel(prob: number): RiskLevel {
  if (prob < 0.3) return "NORMAL";
  if (prob < 0.65) return "ALERTA";
  return "CRÍTICO";
}

// ── Tipos internos (histórico + WS) ──────────────────────────────────────

interface HistoryItem {
  failure_probability: number;
  predicted_class: number;
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

// ── Helpers internos ──────────────────────────────────────────────────────

function toTimeLabel(): string {
  return new Date().toLocaleTimeString("pt-BR", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function itemToPoint(item: HistoryItem): SensorDataPoint {
  return {
    time: new Date(item.timestamp).toLocaleTimeString("pt-BR", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }),
    TP2: parseFloat(item.TP2.toFixed(3)),
    TP3: parseFloat(item.TP3.toFixed(3)),
    Motor_current: parseFloat(item.Motor_current.toFixed(3)),
    Oil_temperature: parseFloat(item.Oil_temperature.toFixed(2)),
    failure_probability: parseFloat(item.failure_probability.toFixed(4)),
    predicted_class: item.predicted_class,
  };
}

function itemToPayload(item: HistoryItem): PredictPayload {
  return {
    TP2: item.TP2,
    TP3: item.TP3,
    H1: item.H1,
    DV_pressure: item.DV_pressure,
    Reservoirs: item.Reservoirs,
    Motor_current: item.Motor_current,
    Oil_temperature: item.Oil_temperature,
    COMP: item.COMP,
    DV_eletric: item.DV_eletric,
    Towers: item.Towers,
    MPG: item.MPG,
    Oil_level: item.Oil_level,
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

  // Refs de controle — não causam re-render
  const isFirstTickRef = useRef(true);
  const lastTimestampRef = useRef<string | null>(null);
  const lastSensorRef = useRef<HistoryItem | null>(null);

  // ── Polling passivo ──────────────────────────────────────────────────────
  // Primeira chamada busca os últimos 30 para seed do gráfico.
  // Chamadas seguintes buscam apenas 1 e ignoram se o timestamp não mudou.
  const tick = useCallback(async (): Promise<void> => {
    const isFirst = isFirstTickRef.current;
    isFirstTickRef.current = false;
    const size = isFirst ? 30 : 1;

    try {
      const res = await fetch(`/api/v1/predictions?page=1&size=${size}`, {
        cache: "no-store",
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const page = (await res.json()) as HistoryPage;
      if (page.items.length === 0) {
        setIsLoading(false);
        return;
      }

      const newest = page.items[0]; // API retorna newest-first

      // Deduplicação: ignora se não há nova predição desde o último tick
      if (!isFirst && newest.timestamp === lastTimestampRef.current) {
        setIsLoading(false);
        return;
      }

      lastTimestampRef.current = newest.timestamp;
      lastSensorRef.current = newest;

      setLatest({
        failure_probability: newest.failure_probability,
        predicted_class: newest.predicted_class,
        timestamp: newest.timestamp,
      });
      setCurrentPayload(itemToPayload(newest));
      setError(null);

      if (isFirst) {
        // Seed: popula com até 30 itens em ordem cronológica
        const points = [...page.items].reverse().map(itemToPoint);
        setHistory(points.slice(-HISTORY_MAX));
      } else {
        // Incremental: adiciona apenas o novo ponto
        const point = itemToPoint(newest);
        setHistory((prev) => {
          const next = [...prev, point];
          return next.length > HISTORY_MAX ? next.slice(-HISTORY_MAX) : next;
        });
      }
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void tick();
    const id = setInterval(() => void tick(), POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [tick]);

  // ── WebSocket: alertas push do backend quando probability > 0.70 (RF-14) ─
  // A URL deriva de window.location.host para funcionar em qualquer ambiente.
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

        setLatest({
          failure_probability: frame.probability,
          predicted_class: frame.predicted_class,
          timestamp: frame.timestamp,
        });
        setError(null);
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
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

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
  };
}
