/**
 * Limiares oficiais de risco de falha — fonte única de verdade.
 *
 * Faixas (probabilidade [0–1]):
 *   NORMAL   : 0.00 ≤ p < 0.35
 *   ALERTA   : 0.35 ≤ p < 0.65
 *   CRÍTICO  : 0.65 ≤ p ≤ 1.00
 *
 * Toda a UI (cards, badges, cores, gating de explicabilidade, dedupe de
 * toasts) DEVE importar `RISK_THRESHOLDS` daqui. Não duplicar magic numbers.
 */

export type RiskLevel = "NORMAL" | "ALERTA" | "CRÍTICO";

export const RISK_THRESHOLDS = {
  /** Probabilidade mínima para sair de NORMAL e entrar em ALERTA. */
  ALERT: 0.35,
  /** Probabilidade mínima para entrar em CRÍTICO. */
  CRITICAL: 0.65,
} as const;

/** Classifica uma probabilidade [0–1] nas 3 faixas oficiais. */
export function getRiskLevel(prob: number): RiskLevel {
  if (prob < RISK_THRESHOLDS.ALERT) return "NORMAL";
  if (prob < RISK_THRESHOLDS.CRITICAL) return "ALERTA";
  return "CRÍTICO";
}

/** `true` se a probabilidade está no patamar CRÍTICO (≥ 65%). */
export function isCriticalProb(prob: number): boolean {
  return prob >= RISK_THRESHOLDS.CRITICAL;
}

/** `true` se a probabilidade está no patamar ALERTA (35–64.99%). */
export function isAlertProb(prob: number): boolean {
  return prob >= RISK_THRESHOLDS.ALERT && prob < RISK_THRESHOLDS.CRITICAL;
}
