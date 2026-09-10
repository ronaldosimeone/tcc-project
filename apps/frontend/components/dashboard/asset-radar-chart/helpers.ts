// ── Escalas/helpers — RNF-58: extraído de AssetRadarChart ───────────────────

// Physical scale ceilings. Each value is the absolute upper operational
// limit of the sensor. Normalization: (raw / max) × 100 → percentage of
// capacity [0–100]. This ensures TP2 (max 12 bar) and Oil_temperature
// (max 80 °C) share the same 0–100 axis without one dominating the polygon.
export const MAX_SCALE = {
  TP2: 12, // bar
  TP3: 12, // bar
  H1: 12, // bar
  Motor_current: 10, // A
  Oil_temperature: 80, // °C
  Reservoirs: 12, // bar
} as const;

/** Map a raw sensor reading to a 0–100 percentage of its physical ceiling. */
export function pct(value: number, max: number): number {
  return parseFloat(Math.min(100, Math.max(0, (value / max) * 100)).toFixed(1));
}

// Optimal/healthy reference — approximate median values from healthy
// (label=0) windows in the MetroPT-3 training dataset — pressurised,
// steady-state operation. Used as the green "Ótimo" area; divergence from
// this shape signals trouble.
export const OPTIMAL_PCT = {
  TP2: pct(10.1, MAX_SCALE.TP2), // 84.2 %
  TP3: pct(10.1, MAX_SCALE.TP3), // 84.2 %
  H1: pct(8.5, MAX_SCALE.H1), // 70.8 %
  Motor_current: pct(3.8, MAX_SCALE.Motor_current), // 38.0 %
  Oil_temperature: pct(64, MAX_SCALE.Oil_temperature), // 80.0 %
  Reservoirs: pct(7.0, MAX_SCALE.Reservoirs), // 58.3 %
} as const;

// ── Anomaly score helpers ─────────────────────────────────────────────────

export type AnomalyLevel = "NORMAL" | "ALERTA" | "CRÍTICO";

export function toAnomalyLevel(score: number): AnomalyLevel {
  if (score < 0.3) return "NORMAL";
  if (score < 0.65) return "ALERTA";
  return "CRÍTICO";
}

export const ANOMALY_STYLE: Record<AnomalyLevel, string> = {
  NORMAL: "border-green-500/30 bg-green-500/10 text-green-400",
  ALERTA: "border-amber-500/30 bg-amber-500/10 text-amber-400",
  CRÍTICO: "border-red-500/30   bg-red-500/10   text-red-400",
};

// ── Chart style constants ─────────────────────────────────────────────────
// NOTA: `var(--x)` direto, sem `hsl(...)` — os tokens do tema já são
// `oklch(...)` completos; `hsl(oklch(...))` é CSS inválido e renderiza
// transparente (mesmo bug corrigido no @theme inline de app/globals.css).

export const GRID_STROKE = "var(--border)";
export const ANGLE_TICK = { fontSize: 11, fill: "var(--muted-foreground)" };
