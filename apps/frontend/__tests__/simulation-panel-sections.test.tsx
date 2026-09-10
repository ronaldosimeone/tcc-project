/**
 * Testes de ModelSection e ScenarioSection — RF-11/RNF-29 — RNF-59.
 *
 * `@/lib/api-client` é mockado (mesmo padrão de alert-settings-form.test.tsx)
 * — nenhuma chamada de rede real. `lib/model-name.ts` é usado de verdade
 * (não mockado) — exercita `filterAndFormatModels`/`formatModelName` reais.
 */
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

// jsdom não implementa hasPointerCapture/scrollIntoView, usados pelo Radix
// Select ao abrir/navegar a listbox — sem esses stubs, o clique no trigger
// não abre o popper em ambiente de teste.
if (!Element.prototype.hasPointerCapture) {
  Element.prototype.hasPointerCapture = () => false;
}
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {};
}

vi.mock("@/lib/api-client", () => ({
  listModels: vi.fn(),
  swapActiveModel: vi.fn(),
  getSimulatorMode: vi.fn(),
  setSimulatorMode: vi.fn(),
}));

import {
  getSimulatorMode,
  listModels,
  setSimulatorMode,
  swapActiveModel,
  type ModelsListResponse,
  type SimulatorModeResponse,
} from "@/lib/api-client";
import { ModelSection } from "@/components/simulation-panel/model-section";
import { ScenarioSection } from "@/components/simulation-panel/scenario-section";

const mockedListModels = listModels as unknown as ReturnType<typeof vi.fn>;
const mockedSwap = swapActiveModel as unknown as ReturnType<typeof vi.fn>;
const mockedGetMode = getSimulatorMode as unknown as ReturnType<typeof vi.fn>;
const mockedSetMode = setSimulatorMode as unknown as ReturnType<typeof vi.fn>;

const MODELS_RESPONSE: ModelsListResponse = {
  active_model: "random_forest_v2",
  models: [
    { name: "random_forest_v2", active: true, artefact_ready: true },
    { name: "xgboost_v1", active: false, artefact_ready: true },
    { name: "isolation_forest_v1", active: false, artefact_ready: false },
  ],
};

describe("ModelSection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("carrega e exibe o modelo activo formatado, listando só os artefatos prontos", async () => {
    mockedListModels.mockResolvedValue(MODELS_RESPONSE);
    render(<ModelSection />);

    await waitFor(() => {
      expect(screen.getByText(/activo · /)).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("combobox"));
    // isolation_forest_v1 tem artefact_ready=false — não vira option.
    expect(
      screen.getByRole("option", { name: "Random Forest" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "XGBoost" })).toBeInTheDocument();
    expect(
      screen.queryByRole("option", { name: /isolation/i }),
    ).not.toBeInTheDocument();
  });

  it("troca de modelo com optimistic update e recarrega a lista em caso de sucesso", async () => {
    mockedListModels.mockResolvedValue(MODELS_RESPONSE);
    mockedSwap.mockResolvedValue({
      previous_model: "random_forest_v2",
      active_model: "xgboost_v1",
      message: "ok",
    });
    render(<ModelSection />);

    await waitFor(() => {
      expect(screen.getByText(/activo · /)).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("combobox"));
    fireEvent.click(screen.getByRole("option", { name: "XGBoost" }));

    expect(screen.getByText(/trocando para xgboost/i)).toBeInTheDocument();
    expect(mockedSwap).toHaveBeenCalledWith("xgboost_v1");

    await waitFor(() => {
      expect(mockedListModels).toHaveBeenCalledTimes(2); // reload após o swap
    });
  });

  it("reverte a seleção e mostra erro quando o swap falha", async () => {
    mockedListModels.mockResolvedValue(MODELS_RESPONSE);
    mockedSwap.mockRejectedValue(new Error("Falha ao trocar de modelo"));
    render(<ModelSection />);

    await waitFor(() => {
      expect(screen.getByText(/activo · /)).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("combobox"));
    fireEvent.click(screen.getByRole("option", { name: "XGBoost" }));

    await waitFor(() => {
      expect(screen.getByText("Falha ao trocar de modelo")).toBeInTheDocument();
    });
  });

  it("exibe mensagem de erro quando o carregamento inicial falha", async () => {
    mockedListModels.mockRejectedValue(new Error("Falha ao carregar modelos"));
    render(<ModelSection />);

    await waitFor(() => {
      expect(screen.getByText("Falha ao carregar modelos")).toBeInTheDocument();
    });
  });
});

describe("ScenarioSection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const NORMAL: SimulatorModeResponse = { mode: "NORMAL", message: "ok" };

  it("carrega o cenário atual e marca o radio correspondente", async () => {
    mockedGetMode.mockResolvedValue(NORMAL);
    render(<ScenarioSection />);

    await waitFor(() => {
      expect(screen.getByText("NORMAL")).toBeInTheDocument();
    });
    const normalRadio = screen.getByRole("radio", { name: /normal/i });
    expect(normalRadio).toHaveAttribute("data-state", "checked");
  });

  it("troca de cenário ao clicar num radio diferente", async () => {
    mockedGetMode.mockResolvedValue(NORMAL);
    mockedSetMode.mockResolvedValue({
      mode: "FAILURE",
      message: "ok",
    } as SimulatorModeResponse);
    render(<ScenarioSection />);

    await waitFor(() => {
      expect(screen.getByRole("radio", { name: /normal/i })).toHaveAttribute(
        "data-state",
        "checked",
      );
    });

    fireEvent.click(document.getElementById("scenario-FAILURE")!);

    await waitFor(() => {
      expect(mockedSetMode).toHaveBeenCalledWith("FAILURE");
    });
    await waitFor(() => {
      expect(screen.getByText("FAILURE")).toBeInTheDocument();
    });
  });

  it("exibe erro quando a troca de cenário falha", async () => {
    mockedGetMode.mockResolvedValue(NORMAL);
    mockedSetMode.mockRejectedValue(new Error("Falha ao trocar cenário"));
    render(<ScenarioSection />);

    await waitFor(() => {
      expect(screen.getByRole("radio", { name: /normal/i })).toHaveAttribute(
        "data-state",
        "checked",
      );
    });

    fireEvent.click(screen.getByRole("radio", { name: /degradação/i }));

    await waitFor(() => {
      expect(screen.getByText("Falha ao trocar cenário")).toBeInTheDocument();
    });
  });

  it("exibe erro quando a leitura inicial do cenário falha", async () => {
    mockedGetMode.mockRejectedValue(new Error("Falha ao ler cenário"));
    render(<ScenarioSection />);

    await waitFor(() => {
      expect(screen.getByText("Falha ao ler cenário")).toBeInTheDocument();
    });
  });
});
