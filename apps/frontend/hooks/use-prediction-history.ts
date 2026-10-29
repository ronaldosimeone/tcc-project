"use client";

/**
 * Hook de histórico de predições — RF-08 / RNF-14.
 *
 * RNF-14: Persiste o histórico de eventos no localStorage (últimos 50).
 * RF-08:  Cada entrada registra o riskLevel para disparar o banner crítico.
 *
 * Estratégia anti-hydration-mismatch (Next.js SSR):
 * localStorage só é acessado dentro de useEffect — nunca durante SSR.
 * O estado inicial é sempre { history: [], isMounted: false }, garantindo
 * que servidor e cliente renderizem o mesmo HTML no primeiro paint.
 * Após a montagem no cliente, isMounted passa para true e o histórico
 * salvo é recarregado de forma transparente.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import dayjs from "dayjs";
import type { PredictResponse } from "@/lib/api-client";
import { getRiskLevel, type RiskLevel } from "@/hooks/use-sensor-data";
import { toRisk } from "@/lib/risk-thresholds";
import { useActiveThreshold } from "@/hooks/use-active-threshold";

// ── Constantes ────────────────────────────────────────────────────────────

export const PREDICTION_HISTORY_STORAGE_KEY = "predictiq:prediction-history";
export const PREDICTION_HISTORY_MAX = 50;

// ── Tipos públicos ────────────────────────────────────────────────────────

export interface PredictionHistoryEntry {
  /** Identificador único do evento */
  id: string;
  /** Timestamp ISO 8601 da predição */
  timestamp: string;
  /** Probabilidade de falha sanitizada [0, 1] */
  failure_probability: number;
  /** Classe predita: 0 = normal · 1 = falha */
  predicted_class: number;
  /** Índice de risco relativo ao limiar do modelo (0–1). Ausente em
   *  entradas antigas do localStorage — nesse caso vale a probabilidade. */
  risk?: number;
  /** Nível de risco derivado do índice */
  riskLevel: RiskLevel;
}

interface UsePredictionHistoryReturn {
  /** Histórico em ordem anti-cronológica — mais recente primeiro */
  history: PredictionHistoryEntry[];
  /** false durante SSR e antes da hidratação do cliente */
  isMounted: boolean;
  /** Apaga o histórico da memória e do localStorage */
  clearHistory: () => void;
}

// ── Helpers internos ──────────────────────────────────────────────────────

/**
 * Sanitiza um valor de probabilidade para [0, 1].
 * Retorna null para NaN, Infinity e valores não-numéricos.
 * Valores negativos são fixados em 0; valores > 1 são fixados em 1.
 */
export function sanitizeProbability(raw: unknown): number | null {
  if (typeof raw !== "number" || !Number.isFinite(raw)) return null;
  return Math.min(1, Math.max(0, raw));
}

function isValidEntry(value: unknown): value is PredictionHistoryEntry {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v["id"] === "string" &&
    typeof v["timestamp"] === "string" &&
    typeof v["failure_probability"] === "number" &&
    typeof v["predicted_class"] === "number" &&
    (v["riskLevel"] === "NORMAL" ||
      v["riskLevel"] === "ALERTA" ||
      v["riskLevel"] === "CRÍTICO")
  );
}

function readFromStorage(): PredictionHistoryEntry[] {
  // Proteção extra para o SSR do Next.js
  if (typeof window === "undefined") return [];

  try {
    const raw = localStorage.getItem(PREDICTION_HISTORY_STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isValidEntry);
  } catch {
    // JSON corrompido ou localStorage indisponível
    return [];
  }
}

function writeToStorage(entries: PredictionHistoryEntry[]): void {
  // Proteção extra para o SSR do Next.js
  if (typeof window === "undefined") return;

  try {
    localStorage.setItem(
      PREDICTION_HISTORY_STORAGE_KEY,
      JSON.stringify(entries),
    );
  } catch {
    // Quota excedida — continua apenas em memória
  }
}

// ── Hook ──────────────────────────────────────────────────────────────────

/** Instante em segundos inteiros (chave de deduplicação). */
const toSecond = (timestamp: string): number => dayjs(timestamp).unix();

export function usePredictionHistory(
  latest: PredictResponse | null,
): UsePredictionHistoryReturn {
  const [isMounted, setIsMounted] = useState(false);
  const decisionThreshold = useActiveThreshold();
  const [history, setHistory] = useState<PredictionHistoryEntry[]>([]);

  // Último segundo registrado. A mesma predição chega pelo poll (timestamp
  // da predição) e pelo WebSocket em CRÍTICO (timestamp do alerta, ms
  // depois): uma entrada por segundo, nunca uma mais antiga que a última.
  const lastSecondRef = useRef<number | null>(null);

  // Efeito 1 — hidratação: lê o localStorage apenas no cliente
  useEffect(() => {
    const stored = readFromStorage();

    // Tudo encapsulado no setTimeout para evitar cascading renders no ESLint
    setTimeout(() => {
      if (stored.length > 0) {
        setHistory(stored);
        lastSecondRef.current = stored[0]
          ? toSecond(stored[0].timestamp)
          : null;
      }
      setIsMounted(true);
    }, 0);
  }, []); // executa uma única vez na montagem

  // Efeito 2 — persiste nova entrada quando latest muda (após hidratação)
  useEffect(() => {
    if (!isMounted || latest === null) return;
    const second = toSecond(latest.timestamp);
    if (lastSecondRef.current !== null && second <= lastSecondRef.current) {
      return;
    }

    // Descarta entradas com probabilidade inválida (NaN, Infinity, null, etc.)
    const prob = sanitizeProbability(latest.failure_probability);
    if (prob === null) return;

    lastSecondRef.current = second;

    const entry: PredictionHistoryEntry = {
      id: `${latest.timestamp}-${Math.random().toString(36).slice(2, 7)}`,
      timestamp: latest.timestamp,
      failure_probability: prob,
      predicted_class: latest.predicted_class,
      risk: toRisk(prob, decisionThreshold),
      riskLevel: getRiskLevel(prob, decisionThreshold),
    };

    setTimeout(() => {
      setHistory((prev) => {
        const next = [entry, ...prev].slice(0, PREDICTION_HISTORY_MAX);
        writeToStorage(next);
        return next;
      });
    }, 0);
  }, [isMounted, latest, decisionThreshold]);

  const clearHistory = useCallback((): void => {
    setHistory([]);
    lastSecondRef.current = null;
    try {
      localStorage.removeItem(PREDICTION_HISTORY_STORAGE_KEY);
    } catch {
      // Ignorar erros de remoção
    }
  }, []);

  return { history, isMounted, clearHistory };
}
