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

/**
 * Escala de risco relativa ao limiar do modelo — mesma de
 * `apps/backend/src/core/risk.py` (aviso crítico, RF-25):
 *   risco = 0,65 · p / limiar                        (p < limiar)
 *   risco = 0,65 + 0,35 · (p − limiar)/(1 − limiar)   (p ≥ limiar)
 * 0,65 = o modelo decide "falha" (CRÍTICO no painel); 1 = probabilidade 1.
 */
export const CRITICAL_RISK: number = RISK_THRESHOLDS.CRITICAL;

const clampThr = (thr: number): number =>
  Math.min(Math.max(thr, 1e-9), 1 - 1e-9);

/** Probabilidade → risco relativo ao limiar do modelo. */
export const riskIndex = (prob: number, decisionThreshold: number): number => {
  const p = Math.min(Math.max(prob, 0), 1);
  const thr = clampThr(decisionThreshold);
  if (p < thr) return (CRITICAL_RISK * p) / thr;
  return CRITICAL_RISK + ((1 - CRITICAL_RISK) * (p - thr)) / (1 - thr);
};

/**
 * Índice de risco exibido na UI (0–1). Com limiar conhecido é `riskIndex`;
 * sem ele, a própria probabilidade — coerente com as faixas fixas.
 * Classificar o valor devolvido: `getRiskLevel(risk)` (sem limiar).
 */
export const toRisk = (
  prob: number,
  decisionThreshold?: number | null,
): number =>
  decisionThreshold == null
    ? Math.min(Math.max(prob, 0), 1)
    : riskIndex(prob, decisionThreshold);

/** `0.653` → `"65.3%"`. */
export const formatRisk = (risk: number): string =>
  `${(Math.min(Math.max(risk, 0), 1) * 100).toFixed(1)}%`;

/** Leituras na mediana exibida — a mesma persistência k = 6 do alarme (RN-03). */
export const SMOOTHING_WINDOW = 6;

/**
 * Mediana das `SMOOTHING_WINDOW` probabilidades mais recentes (mais nova
 * primeiro). Só para exibição: o ciclo carga/alívio do compressor faz a
 * probabilidade de cada leitura oscilar; o alarme já exige k leituras.
 */
export const smoothProbability = (recent: readonly number[]): number => {
  const xs = recent
    .slice(0, SMOOTHING_WINDOW)
    .filter(Number.isFinite)
    .sort((a, b) => a - b);
  if (xs.length === 0) return 0;
  const mid = Math.floor(xs.length / 2);
  return xs.length % 2 ? xs[mid] : (xs[mid - 1] + xs[mid]) / 2;
};

/** Inverso de `riskIndex`: a probabilidade em que o modelo atinge `risk`. */
export const probabilityForRisk = (
  risk: number,
  decisionThreshold: number,
): number => {
  const r = Math.min(Math.max(risk, 0), 1);
  const thr = clampThr(decisionThreshold);
  if (r < CRITICAL_RISK) return (r / CRITICAL_RISK) * thr;
  return thr + ((r - CRITICAL_RISK) / (1 - CRITICAL_RISK)) * (1 - thr);
};
