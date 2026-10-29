/**
 * Seção "Falhas reais" do painel de simulação.
 *
 * 1. Cada cenário mostra o selo certo: vista no treino (demonstração) ×
 *    inédita (teste honesto).
 * 2. Cenário sem dados no servidor fica desabilitado.
 * 3. Reproduzir envia PUT /simulator/scenario a partir do início da falha.
 */
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { RealFailureSection } from "@/components/simulation-panel/real-failure-section";

const SCENARIOS = [
  {
    id: "oil_leak_2022_05",
    label: "Vazamento de óleo — mai/2022",
    failure_type: "oil_leak",
    dataset: "MetroPT 2022",
    start: "2022-05-30T12:00:00",
    end: "2022-06-02T06:18:00",
    seen_in_training: true,
    available: true,
    note: "nota a",
  },
  {
    id: "oil_leak_mt2_07",
    label: "Vazamento de óleo — jul/2022 (MetroPT2)",
    failure_type: "oil_leak",
    dataset: "MetroPT2",
    start: "2022-07-11T10:10:18",
    end: "2022-07-14T10:22:08",
    seen_in_training: false,
    available: false,
    note: "nota b",
  },
];

const IDLE = {
  mode: "NORMAL",
  scenario: null,
  step: null,
  total: null,
  phase: null,
};

const json = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

const fetchMock = vi.fn();

beforeEach(() => {
  vi.stubEnv("NEXT_PUBLIC_API_URL", "/api");
  vi.stubGlobal("fetch", fetchMock);
  fetchMock.mockImplementation((url: string, init?: RequestInit) => {
    if (url.endsWith("/simulator/scenarios"))
      return Promise.resolve(json(SCENARIOS));
    if (url.endsWith("/simulator/scenario") && init?.method === "PUT") {
      return Promise.resolve(
        json({
          ...IDLE,
          mode: "FAILURE",
          scenario: SCENARIOS[0],
          step: 0,
          total: 10,
        }),
      );
    }
    return Promise.resolve(json(IDLE));
  });
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  fetchMock.mockReset();
});

describe("RealFailureSection", () => {
  it("mostra o selo de honestidade de cada falha", async () => {
    render(<RealFailureSection />);
    expect(
      await screen.findByText("Vista no treino · demonstração"),
    ).toBeInTheDocument();
    expect(screen.getByText("Inédita · teste honesto")).toBeInTheDocument();
  });

  it("desabilita cenário sem dados no servidor", async () => {
    render(<RealFailureSection />);
    const button = await screen.findByRole("button", {
      name: "Reproduzir Vazamento de óleo — jul/2022 (MetroPT2)",
    });
    expect(button).toBeDisabled();
  });

  it("reproduz a partir do início da falha", async () => {
    render(<RealFailureSection />);
    fireEvent.click(
      await screen.findByRole("button", {
        name: "Reproduzir Vazamento de óleo — mai/2022",
      }),
    );
    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(
          ([url, init]) =>
            String(url).endsWith("/simulator/scenario") &&
            (init as RequestInit | undefined)?.method === "PUT",
        ),
      ).toBe(true),
    );
    const put = fetchMock.mock.calls.find(
      ([, init]) => (init as RequestInit | undefined)?.method === "PUT",
    );
    expect(JSON.parse(String((put?.[1] as RequestInit).body))).toEqual({
      id: "oil_leak_2022_05",
      lead_minutes: 0,
    });
  });

  it("marca Normal quando nenhuma falha está em reprodução", async () => {
    render(<RealFailureSection />);
    const normal = await screen.findByRole("button", { name: /normal/i });
    await waitFor(() => expect(normal).toHaveAttribute("aria-pressed", "true"));
  });

  it("Parar volta ao modo normal", async () => {
    const playing = {
      mode: "FAILURE",
      scenario: SCENARIOS[0],
      step: 3,
      total: 10,
      phase: "failure",
    };
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url.endsWith("/simulator/scenarios"))
        return Promise.resolve(json(SCENARIOS));
      if (url.endsWith("/simulator/mode"))
        return Promise.resolve(json({ mode: "NORMAL", message: "ok" }));
      return Promise.resolve(json(init?.method === "PUT" ? IDLE : playing));
    });
    render(<RealFailureSection />);
    fireEvent.click(await screen.findByRole("button", { name: /parar/i }));
    await waitFor(() => {
      const put = fetchMock.mock.calls.find(([url]) =>
        String(url).endsWith("/simulator/mode"),
      );
      expect(JSON.parse(String((put?.[1] as RequestInit).body))).toEqual({
        mode: "NORMAL",
      });
    });
  });
});
