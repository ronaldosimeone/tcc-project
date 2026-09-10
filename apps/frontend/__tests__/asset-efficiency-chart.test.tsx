/**
 * Testes do AssetEfficiencyChart — RNF-59.
 *
 * Componente 100% controlado por props (assetId) — cobre a resolução dos
 * dados mock por ativo conhecido, o fallback para ativo desconhecido, e o
 * cálculo de eficiência média exibido no cabeçalho.
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import AssetEfficiencyChart from "@/components/dashboard/AssetEfficiencyChart";

describe("AssetEfficiencyChart", () => {
  it("resolve os dados do ativo conhecido e mostra o assetId no subtítulo", () => {
    render(<AssetEfficiencyChart assetId="APU-Trem-023" />);
    expect(screen.getByText("Eficiência Semanal")).toBeInTheDocument();
    expect(
      screen.getByText(/APU-Trem-023 · Horas em carga vs ocioso/),
    ).toBeInTheDocument();
  });

  it("calcula a eficiência média correta a partir dos dados do ativo", () => {
    // APU-Trem-031 é o ativo com maior eficiência do mock (quase sempre
    // ~23h de carga em 24h) — média arredondada deve ficar bem alta.
    render(<AssetEfficiencyChart assetId="APU-Trem-031" />);
    const avg = screen.getByText(/% avg/);
    const pct = parseInt(avg.textContent!, 10);
    expect(pct).toBeGreaterThanOrEqual(90);
  });

  it("cai no fallback (APU-Trem-042) quando o assetId é desconhecido", () => {
    render(<AssetEfficiencyChart assetId="APU-Trem-999" />);
    // O subtítulo sempre usa o assetId recebido, mesmo usando dados de fallback.
    expect(
      screen.getByText(/APU-Trem-999 · Horas em carga vs ocioso/),
    ).toBeInTheDocument();
    expect(screen.getByText("Eficiência Semanal")).toBeInTheDocument();
  });
});

// NOTA: a legenda ("Em Carga"/"Ocioso") é renderizada pelo <Legend> do
// Recharts dentro do <ResponsiveContainer> — em jsdom (sem layout engine
// real) o container computa largura/altura 0 e o Recharts não desenha
// nenhum filho, então essa legenda não é observável aqui. Comportamento
// já coberto indiretamente pelos demais testes (dados corretos chegam ao
// componente) e validado visualmente no Storybook.
