/**
 * Testes de renderização para SensorChart — RF-06 / RF-07.
 *
 * Textos reais renderizados pelo componente (verificados no fonte):
 *   - Legenda:            "TP2 (bar)", "TP3 (bar)", "Corrente (A)", "Temperatura (°C)"
 *   - Label pressão:      "Pressão — TP2 & TP3 (bar)"   (via &mdash; + &amp;)
 *   - Label elétrico:     "Corrente (A) & Temperatura (°C)" (via &amp;) — NÃO "elétrico"
 *
 * Queries ambíguas:
 *   /TP2/i bate em DOIS elementos: legenda "TP2 (bar)" + label "Pressão — TP2 & TP3 (bar)".
 *   Solução: usar container.textContent (string plana, sem erro de múltiplos elementos)
 *   ou getAllByText quando a intenção é só confirmar existência.
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SensorChart } from "@/components/sensor-chart";
import type { SensorDataPoint } from "@/hooks/use-sensor-data";

// ── Polyfill — Recharts usa ResizeObserver ────────────────────────────────

class MockResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = MockResizeObserver;

// ── Fixtures ──────────────────────────────────────────────────────────────

function makePoint(override: Partial<SensorDataPoint> = {}): SensorDataPoint {
  return {
    time: "12:00:00",
    TP2: 5.52,
    TP3: 9.18,
    Motor_current: 4.1,
    Oil_temperature: 68.3,
    failure_probability: 0.06,
    predicted_class: 0,
    ...override,
  };
}

function makeHistory(n: number): SensorDataPoint[] {
  return Array.from({ length: n }, (_, i) =>
    makePoint({ time: `12:00:${String(i).padStart(2, "0")}` }),
  );
}

// ── Testes ────────────────────────────────────────────────────────────────

describe("SensorChart", () => {
  // ── Estado de loading ───────────────────────────────────────────────────

  it("renderiza sem lançar erros com histórico vazio", () => {
    expect(() =>
      render(<SensorChart history={[]} isAnomaly={false} riskLevel="NORMAL" />),
    ).not.toThrow();
  });

  it("exibe a mensagem 'Coletando dados' quando history está vazio", () => {
    render(<SensorChart history={[]} isAnomaly={false} riskLevel="NORMAL" />);
    expect(screen.getByRole("status")).toHaveTextContent(/coletando dados/i);
  });

  it("exibe 'Coletando dados' com apenas 1 ponto (gráfico precisa de ≥ 2)", () => {
    render(
      <SensorChart
        history={[makePoint()]}
        isAnomaly={false}
        riskLevel="NORMAL"
      />,
    );
    expect(screen.getByRole("status")).toHaveTextContent(/coletando dados/i);
  });

  // ── Renderização com dados ──────────────────────────────────────────────

  it("não exibe 'Coletando dados' quando há pontos suficientes", () => {
    render(
      <SensorChart
        history={makeHistory(5)}
        isAnomaly={false}
        riskLevel="NORMAL"
      />,
    );
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("renderiza o título 'Telemetria de Sensores'", () => {
    render(
      <SensorChart
        history={makeHistory(5)}
        isAnomaly={false}
        riskLevel="NORMAL"
      />,
    );
    expect(screen.getByText(/telemetria de sensores/i)).toBeInTheDocument();
  });

  it("exibe os labels das 4 séries na legenda (RF-06)", () => {
    // "TP2" aparece em DOIS elementos (legenda + label do sub-chart de pressão).
    // Usar container.textContent evita o erro "multiple elements found".
    const { container } = render(
      <SensorChart
        history={makeHistory(5)}
        isAnomaly={false}
        riskLevel="NORMAL"
      />,
    );
    const text = container.textContent ?? "";
    expect(text).toContain("TP2 (bar)");
    expect(text).toContain("TP3 (bar)");
    expect(text).toContain("Corrente (A)");
    expect(text).toContain("Temperatura (°C)");
  });

  it("exibe os labels dos sub-gráficos de pressão e elétrico", () => {
    // Label pressão:  "Pressão — TP2 & TP3 (bar)"
    // Label elétrico: "Corrente (A) & Temperatura (°C)"  ← NÃO é "elétrico"
    const { container } = render(
      <SensorChart
        history={makeHistory(5)}
        isAnomaly={false}
        riskLevel="NORMAL"
      />,
    );
    const text = container.textContent ?? "";
    expect(text).toMatch(/pressão/i);
    // O componente renderiza "Corrente (A) & Temperatura (°C)" para o chart elétrico
    expect(text).toMatch(/corrente \(A\).*temperatura/i);
  });

  // ── RF-07 — Destaque visual de anomalia ────────────────────────────────

  it("NÃO exibe ícone de alerta quando riskLevel é NORMAL", () => {
    render(
      <SensorChart
        history={makeHistory(5)}
        isAnomaly={false}
        riskLevel="NORMAL"
      />,
    );
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("exibe ícone de alerta quando isAnomaly=true (RF-07)", () => {
    render(
      <SensorChart
        history={makeHistory(5)}
        isAnomaly={true}
        riskLevel="ALERTA"
      />,
    );
    expect(screen.getByRole("alert")).toBeInTheDocument();
  });

  it("ícone de alerta exibe o texto 'ALERTA' (RF-07)", () => {
    render(
      <SensorChart
        history={makeHistory(5)}
        isAnomaly={true}
        riskLevel="ALERTA"
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("ALERTA");
  });

  it("ícone de alerta exibe o texto 'CRÍTICO' (RF-07)", () => {
    render(
      <SensorChart
        history={makeHistory(5)}
        isAnomaly={true}
        riskLevel="CRÍTICO"
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("CRÍTICO");
  });

  it("card tem data-anomaly='true' quando isAnomaly=true (RF-07)", () => {
    const { container } = render(
      <SensorChart
        history={makeHistory(5)}
        isAnomaly={true}
        riskLevel="ALERTA"
      />,
    );
    expect(container.querySelector("[data-anomaly='true']")).not.toBeNull();
  });

  it("card tem data-risk com o nível correto (RF-07)", () => {
    const { container } = render(
      <SensorChart
        history={makeHistory(5)}
        isAnomaly={true}
        riskLevel="CRÍTICO"
      />,
    );
    expect(container.querySelector("[data-risk='CRÍTICO']")).not.toBeNull();
  });

  // ── className ───────────────────────────────────────────────────────────

  it("aceita className customizada sem lançar erros", () => {
    expect(() =>
      render(
        <SensorChart
          history={makeHistory(3)}
          isAnomaly={false}
          riskLevel="NORMAL"
          className="lg:col-span-3"
        />,
      ),
    ).not.toThrow();
  });
});
