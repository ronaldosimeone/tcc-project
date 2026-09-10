// ── Mock data — RNF-58: extraído de FleetHealthTable.tsx ────────────────────

import { RISK_THRESHOLDS } from "@/lib/risk-thresholds";
import type { RiskLevel } from "@/hooks/use-sensor-data";

export interface MockAsset {
  id: string;
  riskLevel: RiskLevel;
  /** Saúde sintética [0–100] derivada do mock. */
  health: number;
  /** Variação de saúde nas últimas 24h em pontos percentuais (+/−). */
  trendPp: number;
  prob: number;
  lastSeen: string;
}

// Invariante mantida: prob === (100 - health) / 100. Saúde + Risco somam 100.
export const MOCK_ASSETS: MockAsset[] = [
  {
    id: "APU-Trem-015",
    riskLevel: "NORMAL",
    health: 95,
    trendPp: 1.2,
    prob: 0.05,
    lastSeen: "2 min",
  },
  {
    id: "APU-Trem-023",
    riskLevel: "ALERTA",
    health: 62,
    trendPp: -4.5,
    prob: 0.38,
    lastSeen: "1 min",
  },
  {
    id: "APU-Trem-031",
    riskLevel: "NORMAL",
    health: 96,
    trendPp: 0.3,
    prob: 0.04,
    lastSeen: "3 min",
  },
  {
    id: "APU-Trem-055",
    riskLevel: "NORMAL",
    health: 88,
    trendPp: -1.1,
    prob: 0.12,
    lastSeen: "4 min",
  },
];

/**
 * Mapeia saúde [0–100] no nível de risco oficial.
 *
 * Invariante: `prob = (100 - health) / 100`, portanto:
 *   - health ≥ 65  ↔ prob < 0.35 → NORMAL
 *   - 35 ≤ h < 65  ↔ 0.35 ≤ prob < 0.65 → ALERTA
 *   - health < 35  ↔ prob ≥ 0.65 → CRÍTICO
 * As fronteiras 65/35 são derivadas de RISK_THRESHOLDS — não duplicar.
 */
export function healthToRiskLevel(health: number): RiskLevel {
  const probEquivalent = (100 - health) / 100;
  if (probEquivalent < RISK_THRESHOLDS.ALERT) return "NORMAL";
  if (probEquivalent < RISK_THRESHOLDS.CRITICAL) return "ALERTA";
  return "CRÍTICO";
}

/** Cor da barra de progresso por nível de risco. */
export function healthBarClass(health: number): string {
  const level = healthToRiskLevel(health);
  if (level === "NORMAL") return "[&>*]:bg-emerald-500";
  if (level === "ALERTA") return "[&>*]:bg-amber-500";
  return "[&>*]:bg-red-500";
}

/** Cor do texto do percentual por nível de risco. */
export function healthTextClass(health: number): string {
  const level = healthToRiskLevel(health);
  if (level === "NORMAL") return "text-emerald-600";
  if (level === "ALERTA") return "text-amber-700";
  return "text-red-700";
}
