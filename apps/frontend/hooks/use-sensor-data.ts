"use client";

/**
 * Hook de telemetria — RF-06 / RF-07.
 *
 * Responsabilidades:
 *  - Polling a cada 5 s (RF-06).
 *  - Histórico das últimas HISTORY_MAX leituras para os gráficos.
 *  - Derivação de isAnomaly / riskLevel (RF-07).
 *  - Estados loading / error isolados da UI.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  predict,
  type PredictPayload,
  type PredictResponse,
} from "@/lib/api-client";

// ── Constantes ────────────────────────────────────────────────────────────

export const POLL_INTERVAL_MS = 5_000; // RF-06: polling de 5 s
const HISTORY_MAX = 30;

export const NOMINAL_PAYLOAD: PredictPayload = {
  TP2: 5.5,
  TP3: 9.2,
  H1: 8.8,
  DV_pressure: 2.1,
  Reservoirs: 8.7,
  Motor_current: 4.2,
  Oil_temperature: 68.5,
  COMP: 1,
  DV_eletric: 0,
  Towers: 1,
  MPG: 1,
  Oil_level: 1,
};

// ── Tipos públicos ────────────────────────────────────────────────────────

/** Nível de risco derivado da probabilidade de falha. */
export type RiskLevel = "NORMAL" | "ALERTA" | "CRÍTICO";

/**
 * Um ponto do histórico que alimenta o SensorChart.
 * Contém as 4 séries exigidas pelo RF-06.
 */
export interface SensorDataPoint {
  /** Rótulo de tempo no formato HH:mm:ss */
  time: string;
  /** Pressão a jusante do compressor (bar) */
  TP2: number;
  /** Pressão no painel pneumático (bar) */
  TP3: number;
  /** Corrente do motor (A) */
  Motor_current: number;
  /** Temperatura do óleo (°C) */
  Oil_temperature: number;
  /** Probabilidade de falha retornada pelo modelo */
  failure_probability: number;
  /** Classe predita: 0 = normal · 1 = falha */
  predicted_class: number;
}

export interface SensorDataState {
  /** Histórico de pontos para o gráfico */
  history: SensorDataPoint[];
  /** Resposta mais recente do backend */
  latest: PredictResponse | null;
  /** Payload enviado na última chamada */
  currentPayload: PredictPayload;
  /** Verdadeiro durante a primeira requisição */
  isLoading: boolean;
  /** Verdadeiro quando failure_probability >= 0.3 (RF-07) */
  isAnomaly: boolean;
  /** NORMAL | ALERTA | CRÍTICO */
  riskLevel: RiskLevel;
  /** Último erro de rede/backend (null = sem erro) */
  error: Error | null;
}

// ── Utilitários (exportados para testes unitários) ────────────────────────

export function getRiskLevel(prob: number): RiskLevel {
  if (prob < 0.3) return "NORMAL";
  if (prob < 0.65) return "ALERTA";
  return "CRÍTICO";
}

function jitter(value: number, range: number): number {
  return Math.max(0, value + (Math.random() - 0.5) * range);
}

/** Gera uma leitura com ruído gaussiano a partir da base atual. */
export function simulateNext(base: PredictPayload): PredictPayload {
  return {
    ...base,
    TP2: jitter(base.TP2, 0.35),
    TP3: jitter(base.TP3, 0.25),
    H1: jitter(base.H1, 0.2),
    Motor_current: jitter(base.Motor_current, 0.5),
    Oil_temperature: jitter(base.Oil_temperature, 2.0),
    Reservoirs: jitter(base.Reservoirs, 0.15),
  };
}

function toTimeLabel(): string {
  return new Date().toLocaleTimeString("pt-BR", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

// ── Hook ──────────────────────────────────────────────────────────────────

export function useSensorData(): SensorDataState {
  const baseRef = useRef<PredictPayload>(NOMINAL_PAYLOAD);

  const [history, setHistory] = useState<SensorDataPoint[]>([]);
  const [latest, setLatest] = useState<PredictResponse | null>(null);
  const [currentPayload, setCurrentPayload] =
    useState<PredictPayload>(NOMINAL_PAYLOAD);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const tick = useCallback(async (): Promise<void> => {
    const newPayload = simulateNext(baseRef.current);
    baseRef.current = newPayload;
    setCurrentPayload(newPayload);

    try {
      const result = await predict(newPayload);

      setLatest(result);
      setError(null);

      const point: SensorDataPoint = {
        time: toTimeLabel(),
        TP2: parseFloat(newPayload.TP2.toFixed(3)),
        TP3: parseFloat(newPayload.TP3.toFixed(3)),
        Motor_current: parseFloat(newPayload.Motor_current.toFixed(3)),
        Oil_temperature: parseFloat(newPayload.Oil_temperature.toFixed(2)),
        failure_probability: parseFloat(result.failure_probability.toFixed(4)),
        predicted_class: result.predicted_class,
      };

      setHistory((prev) => {
        const next = [...prev, point];
        return next.length > HISTORY_MAX ? next.slice(-HISTORY_MAX) : next;
      });
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
