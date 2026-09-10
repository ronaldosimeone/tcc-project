/**
 * Testes do AssetTable — RNF-59.
 *
 * Componente 100% controlado por props, sem rede — cobre loading skeleton,
 * a linha ao vivo (LIVE) vs. as linhas mockadas, seleção de linha e o link
 * de telemetria.
 */
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import AssetTable from "@/components/dashboard/AssetTable";

describe("AssetTable", () => {
  it("exibe o skeleton de carregamento quando isLoading=true", () => {
    const { container } = render(
      <AssetTable
        effectiveRiskLevel="NORMAL"
        effectiveProb={0.1}
        tp2={8.1}
        oilTemp={72}
        isLoading
        selectedId="APU-Trem-042"
        onSelect={vi.fn()}
      />,
    );
    expect(container.querySelector(".animate-pulse")).not.toBeNull();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("renderiza a linha ao vivo com o risco/prob/TP2/temp efetivos e as linhas mockadas", () => {
    render(
      <AssetTable
        effectiveRiskLevel="CRÍTICO"
        effectiveProb={0.91}
        tp2={6.2}
        oilTemp={95.5}
        isLoading={false}
        selectedId="APU-Trem-042"
        onSelect={vi.fn()}
      />,
    );
    expect(screen.getByText("APU-Trem-042")).toBeInTheDocument();
    expect(screen.getByText("LIVE")).toBeInTheDocument();
    expect(screen.getByText("91.0%")).toBeInTheDocument();
    expect(screen.getByText("6.20")).toBeInTheDocument();
    expect(screen.getByText("95.5")).toBeInTheDocument();
    expect(screen.getByText("CRÍTICO")).toBeInTheDocument();
    // Ativos mockados também aparecem, marcados como simulados.
    expect(screen.getByText("APU-Trem-015")).toBeInTheDocument();
    expect(screen.getAllByRole("link", { name: /telemetria/i })).toHaveLength(
      1,
    );
  });

  it("chama onSelect com o id correto ao clicar numa linha (live e mockada)", () => {
    const onSelect = vi.fn();
    render(
      <AssetTable
        effectiveRiskLevel="NORMAL"
        effectiveProb={0.1}
        tp2={8.1}
        oilTemp={72}
        isLoading={false}
        selectedId="APU-Trem-042"
        onSelect={onSelect}
      />,
    );

    fireEvent.click(screen.getByText("APU-Trem-015").closest("tr")!);
    expect(onSelect).toHaveBeenCalledWith("APU-Trem-015");

    fireEvent.click(screen.getByText("APU-Trem-042").closest("tr")!);
    expect(onSelect).toHaveBeenCalledWith("APU-Trem-042");
  });

  it("clicar no link de telemetria não dispara onSelect (stopPropagation)", () => {
    const onSelect = vi.fn();
    render(
      <AssetTable
        effectiveRiskLevel="NORMAL"
        effectiveProb={0.1}
        tp2={8.1}
        oilTemp={72}
        isLoading={false}
        selectedId="APU-Trem-042"
        onSelect={onSelect}
      />,
    );

    fireEvent.click(screen.getByRole("link", { name: /telemetria/i }));
    expect(onSelect).not.toHaveBeenCalled();
  });
});
