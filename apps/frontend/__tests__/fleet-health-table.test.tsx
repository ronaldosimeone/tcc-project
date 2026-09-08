/**
 * Testes de regressão — FleetHealthTable (RNF-39).
 *
 * Cobertura:
 *   - Renderização correta das 5 linhas (1 live + 4 mock) com status/saúde.
 *   - Atualização de status quando `effectiveRiskLevel`/`effectiveProb`
 *     realmente mudam (garante que o `React.memo` não "congelou" o
 *     componente).
 *   - Interação de seleção (`onSelect`) preservada.
 *   - Regressão do memo: re-render com as MESMAS props não deve re-executar
 *     o corpo do componente (`cn()` não é chamado de novo) — prova
 *     determinística de que a otimização funciona, sem depender de timing.
 */

import { render, screen, fireEvent, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import FleetHealthTable from "@/components/dashboard/FleetHealthTable";

vi.mock("@/lib/utils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/utils")>();
  return { ...actual, cn: vi.fn(actual.cn) };
});

import { cn } from "@/lib/utils";

const baseProps = {
  effectiveRiskLevel: "NORMAL" as const,
  effectiveProb: 0.05,
  isLoading: false,
  selectedId: "APU-Trem-042",
  onSelect: vi.fn(),
};

describe("FleetHealthTable", () => {
  it("renderiza a linha LIVE e as 4 linhas simuladas", () => {
    render(<FleetHealthTable {...baseProps} />);
    expect(screen.getByText("APU-Trem-042")).toBeInTheDocument();
    expect(screen.getByText("APU-Trem-015")).toBeInTheDocument();
    expect(screen.getByText("APU-Trem-023")).toBeInTheDocument();
    expect(screen.getByText("APU-Trem-031")).toBeInTheDocument();
    expect(screen.getByText("APU-Trem-055")).toBeInTheDocument();
    // 5 badges de risco (1 live via prop + 4 mock)
    expect(
      screen.getAllByText(/NORMAL|ALERTA|CRÍTICO/).length,
    ).toBeGreaterThanOrEqual(5);
  });

  it("atualiza o status/saúde da linha LIVE quando effectiveRiskLevel muda (regressão RF-19)", () => {
    const { rerender } = render(
      <FleetHealthTable
        {...baseProps}
        effectiveRiskLevel="NORMAL"
        effectiveProb={0.05}
      />,
    );
    const liveRow = () => screen.getByText("APU-Trem-042").closest("tr")!;
    expect(within(liveRow()).getByText("95%")).toBeInTheDocument(); // (1-0.05)*100

    rerender(
      <FleetHealthTable
        {...baseProps}
        effectiveRiskLevel="CRÍTICO"
        effectiveProb={0.8}
      />,
    );
    expect(within(liveRow()).getByText("20%")).toBeInTheDocument(); // (1-0.8)*100
    expect(within(liveRow()).getByText("CRÍTICO")).toBeInTheDocument();
  });

  it("chama onSelect com o id correto ao clicar numa linha", () => {
    const onSelect = vi.fn();
    render(<FleetHealthTable {...baseProps} onSelect={onSelect} />);

    fireEvent.click(screen.getByText("APU-Trem-042").closest("tr")!);
    expect(onSelect).toHaveBeenCalledWith("APU-Trem-042");
  });

  it("mostra o skeleton em vez da tabela quando isLoading=true", () => {
    render(<FleetHealthTable {...baseProps} isLoading />);
    expect(screen.queryByText("APU-Trem-042")).not.toBeInTheDocument();
  });

  it("React.memo: re-render com as MESMAS props não re-executa o corpo do componente", () => {
    const mockedCn = cn as unknown as ReturnType<typeof vi.fn>;
    mockedCn.mockClear();

    const { rerender } = render(<FleetHealthTable {...baseProps} />);
    const callsAfterMount = mockedCn.mock.calls.length;
    expect(callsAfterMount).toBeGreaterThan(0); // sanity: cn() é chamado no mount

    // Mesmas props (novo objeto, mas valores idênticos) — memo deve bloquear.
    rerender(<FleetHealthTable {...baseProps} />);
    expect(mockedCn.mock.calls.length).toBe(callsAfterMount);
  });

  it("React.memo: re-render com props DIFERENTES re-executa normalmente", () => {
    const mockedCn = cn as unknown as ReturnType<typeof vi.fn>;
    mockedCn.mockClear();

    const { rerender } = render(<FleetHealthTable {...baseProps} />);
    const callsAfterMount = mockedCn.mock.calls.length;

    rerender(
      <FleetHealthTable
        {...baseProps}
        effectiveRiskLevel="CRÍTICO"
        effectiveProb={0.9}
      />,
    );
    expect(mockedCn.mock.calls.length).toBeGreaterThan(callsAfterMount);
  });
});
