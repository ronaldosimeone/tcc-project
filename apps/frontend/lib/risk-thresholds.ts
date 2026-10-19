/**
 * Limiares oficiais de risco de falha — fonte única de verdade.
 *
 * Cada modelo é calibrado com o PRÓPRIO limiar de decisão (card do modelo,
 * escolhido sem vazamento: MLP 0,08 … XGBoost 0,905). O risco é, portanto,
 * relativo a esse limiar:
 *   CRÍTICO  : p ≥ limiar                 (o modelo decide "falha")
 *   ALERTA   : p ≥ limiar · ALERT_RATIO   (mesma proporção 0,35/0,65)
 *   NORMAL   : abaixo disso
 *
 * Sem limiar conhecido (modelo não carregado, payload antigo) valem as
 * faixas fixas: NORMAL < 0,35 ≤ ALERTA < 0,65 ≤ CRÍTICO.
 *
 * Toda a UI (cards, badges, cores, gating de explicabilidade, dedupe de
 * toasts) DEVE importar daqui. Não duplicar magic numbers.
 */

export type RiskLevel = "NORMAL" | "ALERTA" | "CRÍTICO";

export const RISK_THRESHOLDS = {
  /** Probabilidade mínima para sair de NORMAL e entrar em ALERTA. */
  ALERT: 0.35,
  /** Probabilidade mínima para entrar em CRÍTICO. */
  CRITICAL: 0.65,
} as const;

/** Fração do limiar do modelo a partir da qual o risco é ALERTA. */
export const ALERT_RATIO: number =
  RISK_THRESHOLDS.ALERT / RISK_THRESHOLDS.CRITICAL;

const bounds = (
  decisionThreshold?: number | null,
): { alert: number; critical: number } =>
  decisionThreshold == null
    ? { alert: RISK_THRESHOLDS.ALERT, critical: RISK_THRESHOLDS.CRITICAL }
    : {
        alert: decisionThreshold * ALERT_RATIO,
        critical: decisionThreshold,
      };

/** Classifica uma probabilidade [0–1] relativa ao limiar do modelo. */
export function getRiskLevel(
  prob: number,
  decisionThreshold?: number | null,
): RiskLevel {
  const { alert, critical } = bounds(decisionThreshold);
  if (prob < alert) return "NORMAL";
  if (prob < critical) return "ALERTA";
  return "CRÍTICO";
}

/** `true` se a probabilidade está no patamar CRÍTICO. */
export function isCriticalProb(
  prob: number,
  decisionThreshold?: number | null,
): boolean {
  return getRiskLevel(prob, decisionThreshold) === "CRÍTICO";
}

/** `true` se a probabilidade está no patamar ALERTA. */
export function isAlertProb(
  prob: number,
  decisionThreshold?: number | null,
): boolean {
  return getRiskLevel(prob, decisionThreshold) === "ALERTA";
}
