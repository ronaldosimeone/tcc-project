/**
 * Testes do FleetKPIs — RNF-59.
 *
 * Componente 100% controlado por props (sem fetch/hook de rede) — os 4
 * cards (Saúde Global, Ativos em Alerta, Anomalia Máxima, Latência) são
 * testados diretamente, incluindo a janela deslizante de latência
 * (`useLatencyHistory`) via re-renders com novos `messageId`.
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import FleetKPIs from "@/components/dashboard/FleetKPIs";

describe("FleetKPIs", () => {
  it("exibe skeletons nos 4 cards quando isLoading=true", () => {
    const { container } = render(
      <FleetKPIs
        liveProbability={0}
        effectiveRiskLevel="NORMAL"
        latencyTelemetry={null}
        isLoading
      />,
    );
    expect(
      container.querySelectorAll("[aria-hidden='true']").length,
    ).toBeGreaterThan(0);
    expect(
      screen.queryByText(/aguardando primeira inferência/i),
    ).not.toBeInTheDocument();
  });

  it("estado NORMAL: mostra a matriz Andon com o bloco ao vivo verde e o rótulo correto", () => {
    render(
      <FleetKPIs
        liveProbability={0.12}
        effectiveRiskLevel="NORMAL"
        latencyTelemetry={null}
        isLoading={false}
      />,
    );
    expect(screen.getByText("Saúde Global da Frota")).toBeInTheDocument();
    expect(screen.getByText("Ativos em Alerta")).toBeInTheDocument();
    expect(screen.getByText("12.0")).toBeInTheDocument(); // anomalia máxima = 12.0%
    expect(screen.getByLabelText("APU-Trem-042: NORMAL")).toBeInTheDocument();
    // 1 ativo simulado (APU-Trem-023) já vem em ALERTA no mock fixo — total 1/5.
    expect(screen.getByText("1")).toBeInTheDocument();
  });

  it("estado CRÍTICO: o bloco ao vivo da matriz Andon reflete o risco", () => {
    render(
      <FleetKPIs
        liveProbability={0.91}
        effectiveRiskLevel="CRÍTICO"
        latencyTelemetry={null}
        isLoading={false}
      />,
    );
    expect(screen.getByLabelText("APU-Trem-042: CRÍTICO")).toBeInTheDocument();
    expect(screen.getByText("91.0")).toBeInTheDocument();
    // 2 ativos em alerta agora: APU-Trem-042 (crítico) + APU-Trem-023 (mock).
    expect(screen.getByText("2")).toBeInTheDocument();
  });

  it("sem telemetria de latência: mostra 'aguardando primeira inferência'", () => {
    render(
      <FleetKPIs
        liveProbability={0.2}
        effectiveRiskLevel="NORMAL"
        latencyTelemetry={null}
        isLoading={false}
      />,
    );
    expect(
      screen.getByText("aguardando primeira inferência…"),
    ).toBeInTheDocument();
  });

  it("acumula o histórico de latência a cada novo messageId e mostra a leitura mais recente", () => {
    const { rerender } = render(
      <FleetKPIs
        liveProbability={0.2}
        effectiveRiskLevel="NORMAL"
        latencyTelemetry={{ messageId: "msg-1", latencyMs: 42 }}
        isLoading={false}
      />,
    );
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByText("ao vivo · 1/24 amostras")).toBeInTheDocument();

    rerender(
      <FleetKPIs
        liveProbability={0.2}
        effectiveRiskLevel="NORMAL"
        latencyTelemetry={{ messageId: "msg-2", latencyMs: 58 }}
        isLoading={false}
      />,
    );
    expect(screen.getByText("58")).toBeInTheDocument();
    expect(screen.getByText("ao vivo · 2/24 amostras")).toBeInTheDocument();

    // Re-render com o MESMO messageId não deve duplicar a amostra (de-dupe).
    rerender(
      <FleetKPIs
        liveProbability={0.2}
        effectiveRiskLevel="NORMAL"
        latencyTelemetry={{ messageId: "msg-2", latencyMs: 58 }}
        isLoading={false}
      />,
    );
    expect(screen.getByText("ao vivo · 2/24 amostras")).toBeInTheDocument();
  });

  it("ignora leituras de latência não-finitas (NaN/Infinity)", () => {
    render(
      <FleetKPIs
        liveProbability={0.2}
        effectiveRiskLevel="NORMAL"
        latencyTelemetry={{ messageId: "msg-bad", latencyMs: NaN }}
        isLoading={false}
      />,
    );
    expect(
      screen.getByText("aguardando primeira inferência…"),
    ).toBeInTheDocument();
  });
});
