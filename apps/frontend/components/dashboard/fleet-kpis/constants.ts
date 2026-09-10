// ── Constantes/séries mock — RNF-58: extraído de FleetKPIs ──────────────────

import type { RiskLevel } from "@/lib/risk-thresholds";

export type SparkTone = "ok" | "warn" | "danger" | "neutral";

export const SPARK_COLOR: Record<SparkTone, string> = {
  ok: "#10b981",
  warn: "#f59e0b",
  danger: "#f43f5e",
  neutral: "#64748b",
};

export const TONE_BG: Record<SparkTone, string> = {
  neutral: "bg-slate-100 text-slate-600",
  ok: "bg-emerald-50 text-emerald-600",
  warn: "bg-amber-50 text-amber-600",
  danger: "bg-rose-50 text-rose-600",
};

/** Tamanho da janela deslizante da sparkline de latência. */
export const LATENCY_HISTORY_SIZE = 24;

// ── Séries mock para as sparklines (24 pontos = 1 ponto/hora) ───────────────

export const SPARK_HEALTH = [
  91, 92, 92, 93, 91, 90, 91, 92, 93, 94, 94, 93, 94, 95, 94, 93, 92, 93, 94,
  94, 95, 95, 94, 94,
];

export const SPARK_ANOMALY = [
  12, 14, 11, 13, 18, 20, 24, 22, 27, 31, 28, 33, 36, 34, 38, 41, 39, 42, 44,
  43, 45, 44, 46, 46,
];

// Cor do bloco Andon por nível — pulse só em CRÍTICO para evitar ruído visual.
export const ANDON_COLOR: Record<RiskLevel, string> = {
  NORMAL: "bg-emerald-500",
  ALERTA: "bg-amber-500",
  CRÍTICO: "bg-red-500 animate-pulse",
};
