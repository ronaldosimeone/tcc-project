/**
 * Testes de `ConnectionStatus` — RF-17 / RNF-77.
 *
 * Foco desta suíte: o pill de taxa de erro (RNF-77) é opcional, reaproveita
 * o layout já coberto por outros testes (connection-resilience.test.tsx
 * cobre sensores/alertas via o Dashboard inteiro) — aqui testado isolado,
 * sem montar o SensorMonitor. Comportamento e acessibilidade (via
 * aria-label/title), não snapshot visual (RNF-77 Fase 12).
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ConnectionStatus } from "@/components/connection-status";

describe("ConnectionStatus — pill de taxa de erro (RNF-77)", () => {
  it("não renderiza o pill 'API' quando errorRateStatus é omitido", () => {
    render(<ConnectionStatus sseStatus="connected" wsStatus="open" />);
    expect(screen.queryByText("API")).not.toBeInTheDocument();
  });

  it("não renderiza o pill 'API' quando errorRateStatus é null (poll ainda não resolveu)", () => {
    render(
      <ConnectionStatus
        sseStatus="connected"
        wsStatus="open"
        errorRateStatus={null}
      />,
    );
    expect(screen.queryByText("API")).not.toBeInTheDocument();
  });

  it("NORMAL — renderiza o pill 'API' com rótulo 'Normal'", () => {
    render(
      <ConnectionStatus
        sseStatus="connected"
        wsStatus="open"
        errorRateStatus="NORMAL"
      />,
    );
    expect(screen.getByText("API")).toBeInTheDocument();
    expect(screen.getByText("Normal")).toBeInTheDocument();
  });

  it("WARNING — renderiza o pill 'API' com rótulo 'Atenção'", () => {
    render(
      <ConnectionStatus
        sseStatus="connected"
        wsStatus="open"
        errorRateStatus="WARNING"
      />,
    );
    expect(screen.getByText("Atenção")).toBeInTheDocument();
  });

  it("CRITICAL — renderiza o pill 'API' com rótulo 'Crítico'", () => {
    render(
      <ConnectionStatus
        sseStatus="connected"
        wsStatus="open"
        errorRateStatus="CRITICAL"
      />,
    );
    expect(screen.getByText("Crítico")).toBeInTheDocument();
  });

  it("recuperação: de CRITICAL para NORMAL, o rótulo atualiza (não fica preso no pior estado)", () => {
    const { rerender } = render(
      <ConnectionStatus
        sseStatus="connected"
        wsStatus="open"
        errorRateStatus="CRITICAL"
      />,
    );
    expect(screen.getByText("Crítico")).toBeInTheDocument();

    rerender(
      <ConnectionStatus
        sseStatus="connected"
        wsStatus="open"
        errorRateStatus="NORMAL"
      />,
    );
    expect(screen.queryByText("Crítico")).not.toBeInTheDocument();
    expect(screen.getByText("Normal")).toBeInTheDocument();
  });

  it("acessibilidade: pill 'API' expõe aria-label com nome e rótulo legível", () => {
    render(
      <ConnectionStatus
        sseStatus="connected"
        wsStatus="open"
        errorRateStatus="CRITICAL"
      />,
    );
    expect(screen.getByLabelText("API: Crítico")).toBeInTheDocument();
  });

  it("não interfere nos pills existentes de Sensores/Alertas", () => {
    render(
      <ConnectionStatus
        sseStatus="reconnecting"
        wsStatus="open"
        errorRateStatus="NORMAL"
      />,
    );
    expect(
      screen.getByLabelText("Sensores: Reconectando..."),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Alertas: Online")).toBeInTheDocument();
  });
});
