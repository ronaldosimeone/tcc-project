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
