/**
 * Teste de regressão — Sidebar + code splitting do SimulationPanel (RNF-40).
 *
 * `SimulationPanel` passou a ser carregado via `next/dynamic` (só quando o
 * usuário abre o painel) em vez de estar sempre no bundle inicial do
 * Dashboard. Este teste garante que a interação ainda funciona: clicar em
 * "Simulação" continua abrindo o painel, apenas com um passo de import
 * assíncrono no meio (coberto com `findBy*`/`waitFor`).
 */

import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import Sidebar from "@/components/sidebar";

vi.mock("next/navigation", () => ({
  usePathname: () => "/",
}));

vi.mock("@/lib/api-client", () => ({
  listModels: vi
    .fn()
    .mockResolvedValue({ active_model: "random_forest_v2", models: [] }),
  getSimulatorMode: vi.fn().mockResolvedValue({ mode: "NORMAL" }),
  setSimulatorMode: vi.fn(),
  swapActiveModel: vi.fn(),
}));

describe("Sidebar", () => {
  it("abre o SimulationPanel (code-split) ao clicar em 'Simulação'", async () => {
    render(<Sidebar />);

    expect(screen.queryByText("Painel de Simulação")).not.toBeInTheDocument();

    fireEvent.click(screen.getByText("Simulação"));

    // Carregamento assíncrono (next/dynamic) — precisa de waitFor. Timeout
    // generoso: a suíte completa roda vários arquivos jsdom em paralelo, e
    // o import dinâmico + portal do Sheet podem legitimamente demorar mais
    // que o default (1000ms) sob essa carga.
    await waitFor(
      () => {
        expect(screen.getByText("Painel de Simulação")).toBeInTheDocument();
      },
      { timeout: 5000 },
    );
  });

  it("renderiza os itens de navegação principais", () => {
    render(<Sidebar />);
    expect(screen.getByText("Dashboard")).toBeInTheDocument();
    expect(screen.getByText("Sensores")).toBeInTheDocument();
    expect(screen.getByText("Histórico")).toBeInTheDocument();
  });
});
