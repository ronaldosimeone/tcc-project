/**
 * Risco relativo ao limiar calibrado do modelo ativo.
 *
 * Achado ao vivo: com mlp_v2 (limiar 0,08) e bilstm_v2 (0,27) o modelo
 * decidia "falha" com P = 0,2–0,5 e a tela mostrava NORMAL, porque as faixas
 * eram fixas (0,35 / 0,65) para todos os modelos.
 */
import { describe, expect, it } from "vitest";

import {
  ALERT_RATIO,
  getRiskLevel,
  isAlertProb,
  isCriticalProb,
} from "@/lib/risk-thresholds";

describe("risco relativo ao limiar do modelo", () => {
  it("CRÍTICO exatamente quando o modelo decide falha (P ≥ limiar)", () => {
    expect(getRiskLevel(0.08, 0.08)).toBe("CRÍTICO");
    expect(getRiskLevel(0.3, 0.08)).toBe("CRÍTICO");
    expect(isCriticalProb(0.3, 0.27)).toBe(true);
  });

  it("modelo de limiar alto não fica crítico abaixo do próprio limiar", () => {
    expect(getRiskLevel(0.75, 0.905)).toBe("ALERTA");
    expect(isCriticalProb(0.75, 0.905)).toBe(false);
  });

  it("ALERTA começa na mesma proporção das faixas oficiais", () => {
    expect(ALERT_RATIO).toBeCloseTo(0.35 / 0.65, 6);
    expect(getRiskLevel(0.27 * ALERT_RATIO, 0.27)).toBe("ALERTA");
    expect(getRiskLevel(0.27 * ALERT_RATIO - 0.001, 0.27)).toBe("NORMAL");
    expect(isAlertProb(0.2, 0.27)).toBe(true);
  });

  it("sem limiar conhecido mantém as faixas fixas", () => {
    expect(getRiskLevel(0.3)).toBe("NORMAL");
    expect(getRiskLevel(0.5, null)).toBe("ALERTA");
    expect(getRiskLevel(0.7, undefined)).toBe("CRÍTICO");
  });
});

describe("escala de risco do alerta crítico", () => {
  it("65% é exatamente o limiar do modelo e a inversa fecha", async () => {
    const { riskIndex, probabilityForRisk, CRITICAL_RISK } = await import(
      "@/lib/risk-thresholds"
    );
    expect(CRITICAL_RISK).toBeCloseTo(0.65, 6);
    expect(riskIndex(0.06, 0.06)).toBeCloseTo(0.65, 6);
    for (const thr of [0.06, 0.1, 0.694, 0.984]) {
      for (const r of [0.5, 0.65, 0.85, 0.95]) {
        expect(riskIndex(probabilityForRisk(r, thr), thr)).toBeCloseTo(r, 6);
      }
    }
    // RF v3 (limiar 0,694): 85% ≈ P 0,869 — próximo da regra antiga de 85%
    expect(probabilityForRisk(0.85, 0.694)).toBeCloseTo(0.869, 3);
  });
});
