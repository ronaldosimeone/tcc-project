/**
 * Testes de regressão — ModelStatusCard (RNF-39).
 *
 * Cobertura funcional (o "prova de memo bailout" para este componente fica
 * a cargo da medição real com React Profiler documentada em
 * `frontend_performance_report.md` — aqui cobrimos correção de conteúdo,
 * não timing):
 *   - Renderiza o total e a distribuição (saudáveis/atenção/crítico).
 *   - Atualiza a distribuição exibida quando a prop `distribution` muda de
 *     valor (garante que o `React.memo` não "congelou" o componente).
 *   - Mostra o modelo ativo assim que `listModels()` resolve.
 */

import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import ModelStatusCard from "@/components/dashboard/ModelStatusCard";

vi.mock("@/lib/api-client", () => ({
  listModels: vi.fn(),
}));

import { listModels } from "@/lib/api-client";

const mockedListModels = listModels as unknown as ReturnType<typeof vi.fn>;

beforeEach(() => {
  mockedListModels.mockResolvedValue({
    active_model: "random_forest_v2",
    models: [],
  });
});

describe("ModelStatusCard", () => {
  it("renderiza o total e a distribuição de saúde", () => {
    render(
      <ModelStatusCard
        distribution={{ healthy: 4, warning: 1, critical: 0 }}
      />,
    );
    expect(screen.getByText("5")).toBeInTheDocument(); // total
    const rows = screen.getAllByRole("listitem");
    expect(
      rows.find((r) => r.textContent?.includes("Saudáveis"))?.textContent,
    ).toContain("4");
    expect(
      rows.find((r) => r.textContent?.includes("Atenção"))?.textContent,
    ).toContain("1");
  });

  it("atualiza a distribuição exibida quando a prop muda de valor (regressão memo)", () => {
    const { rerender } = render(
      <ModelStatusCard
        distribution={{ healthy: 4, warning: 1, critical: 0 }}
      />,
    );
    expect(screen.getByText("5")).toBeInTheDocument();

    rerender(
      <ModelStatusCard
        distribution={{ healthy: 2, warning: 1, critical: 2 }}
      />,
    );
    expect(screen.getByText("5")).toBeInTheDocument(); // total ainda 5, mas a distribuição mudou
    // 2 críticos agora — o valor "2" aparece na linha "Crítico"
    const rows = screen.getAllByRole("listitem");
    const criticalRow = rows.find((r) => r.textContent?.includes("Crítico"));
    expect(criticalRow?.textContent).toContain("2");
  });

  it("mostra o modelo ativo assim que listModels() resolve", async () => {
    render(
      <ModelStatusCard
        distribution={{ healthy: 5, warning: 0, critical: 0 }}
      />,
    );
    await waitFor(() => {
      expect(screen.getByText(/RANDOM_FOREST_V2/i)).toBeInTheDocument();
    });
  });

  it("mostra 'indisponível' quando listModels() falha", async () => {
    mockedListModels.mockRejectedValue(new Error("network error"));
    render(
      <ModelStatusCard
        distribution={{ healthy: 5, warning: 0, critical: 0 }}
      />,
    );
    await waitFor(() => {
      expect(screen.getByText("indisponível")).toBeInTheDocument();
    });
  });
});
