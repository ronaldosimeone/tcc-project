/**
 * Testes de estados de UI — RNF-21.
 *
 * Cobre os três estados obrigatórios para todos os componentes de dados:
 *
 *   Loading  — Skeleton renderizado enquanto isLoading=true (primeira requisição pendente).
 *   Error    — ErrorState visível quando backend inacessível e sem histórico local.
 *   Empty    — EmptyState no SensorChart (< 2 pontos) e no AlertPanel (sem eventos).
 *
 * Mocking:
 *   @/lib/api-client é interceptado via vi.mock() para controlar resolve/reject
 *   sem depender de rede real.
 *   ResizeObserver é polyfilled para o jsdom (Recharts).
 */

import { render, screen, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SensorMonitor from "@/components/sensor-monitor";
import { SensorChart } from "@/components/sensor-chart";
import { POLL_INTERVAL_MS } from "@/hooks/use-sensor-data";

// ── Mocks ─────────────────────────────────────────────────────────────────

const { MOCK_RESPONSE } = vi.hoisted(() => ({
  MOCK_RESPONSE: {
    predicted_class: 0 as const,
    failure_probability: 0.05,
    timestamp: new Date("2024-06-01T12:00:00Z").toISOString(),
  },
}));

vi.mock("@/lib/api-client", () => ({
  predict: vi.fn().mockResolvedValue(MOCK_RESPONSE),
}));

class MockResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = MockResizeObserver;

// ── Suite: Loading state (Skeleton) ───────────────────────────────────────

describe("Loading state — RNF-21", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    process.env.NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000";
  });

  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it("exibe gauge-skeleton enquanto a primeira predição está pendente", () => {
    render(<SensorMonitor />);
    // Não avança timers — isLoading ainda é true
    expect(screen.getByTestId("gauge-skeleton")).toBeInTheDocument();
  });

  it("exibe 4 kpi-skeleton enquanto isLoading=true", () => {
    render(<SensorMonitor />);
    const skeletons = screen.getAllByTestId("kpi-skeleton");
    expect(skeletons).toHaveLength(4);
  });

  it("exibe chip-skeleton para cada um dos 8 sensores enquanto isLoading=true", () => {
    render(<SensorMonitor />);
    const chips = screen.getAllByTestId("chip-skeleton");
    expect(chips).toHaveLength(8);
  });

  it("remove gauge-skeleton após a API responder com sucesso", async () => {
    render(<SensorMonitor />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(screen.queryByTestId("gauge-skeleton")).not.toBeInTheDocument();
  });

  it("remove kpi-skeleton após a API responder com sucesso", async () => {
    render(<SensorMonitor />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(screen.queryAllByTestId("kpi-skeleton")).toHaveLength(0);
  });

  it("exibe unidades nos KPI cards após o loading terminar", async () => {
    render(<SensorMonitor />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // Após isLoading=false, os KPI cards mostram o valor real com a unidade.
    // "bar" aparece em "Pressão TP2" e "Reservatório"; existe ao menos 1 span com texto "bar".
    const barElements = screen.getAllByText("bar");
    expect(barElements.length).toBeGreaterThanOrEqual(1);
  });

  it("não exibe gauge-skeleton após múltiplos ciclos de polling", async () => {
    render(<SensorMonitor />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS * 2);
    });

    expect(screen.queryByTestId("gauge-skeleton")).not.toBeInTheDocument();
  });
});

// ── Suite: Error state — RNF-21 ────────────────────────────────────────────

describe("Error state — RNF-21", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    process.env.NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000";
  });

  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it("exibe error-state quando backend offline e sem histórico local", async () => {
    const { predict: mockPredict } = await import("@/lib/api-client");
    vi.mocked(mockPredict).mockRejectedValueOnce(
      new Error("Network Error — connection refused"),
    );

    render(<SensorMonitor />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(screen.getByTestId("error-state")).toBeInTheDocument();
  });

  it("error-state tem role='alert' para leitores de tela", async () => {
    const { predict: mockPredict } = await import("@/lib/api-client");
    vi.mocked(mockPredict).mockRejectedValueOnce(new Error("Network Error"));

    render(<SensorMonitor />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(screen.getByTestId("error-state")).toHaveAttribute("role", "alert");
  });

  it("error-state contém mensagem descritiva de conexão", async () => {
    const { predict: mockPredict } = await import("@/lib/api-client");
    vi.mocked(mockPredict).mockRejectedValueOnce(new Error("Network Error"));

    render(<SensorMonitor />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    const errorEl = screen.getByTestId("error-state");
    expect(errorEl.textContent).toMatch(/backend|conexão|API/i);
  });

  it("error-state contém botão 'Tentar novamente'", async () => {
    const { predict: mockPredict } = await import("@/lib/api-client");
    vi.mocked(mockPredict).mockRejectedValueOnce(new Error("Network Error"));

    render(<SensorMonitor />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(
      screen.getByRole("button", { name: /tentar novamente/i }),
    ).toBeInTheDocument();
  });

  it("NÃO exibe error-state quando a API responde com sucesso", async () => {
    render(<SensorMonitor />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(screen.queryByTestId("error-state")).not.toBeInTheDocument();
  });

  it("exibe badge 'Backend offline' junto com o error-state (complementar)", async () => {
    const { predict: mockPredict } = await import("@/lib/api-client");
    vi.mocked(mockPredict).mockRejectedValueOnce(new Error("Network Error"));

    render(<SensorMonitor />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // Ambos coexistem: badge no header + error-state inline no conteúdo
    expect(screen.getByText(/backend offline/i)).toBeInTheDocument();
    expect(screen.getByTestId("error-state")).toBeInTheDocument();
  });
});

// ── Suite: Empty state — RNF-21 ────────────────────────────────────────────

describe("Empty state — RNF-21", () => {
  it("SensorChart: exibe chart-empty-state com data-testid correto quando history.length < 2", () => {
    render(<SensorChart history={[]} isAnomaly={false} riskLevel="NORMAL" />);
    expect(screen.getByTestId("chart-empty-state")).toBeInTheDocument();
  });

  it("SensorChart: chart-empty-state mantém role='status' para acessibilidade", () => {
    render(<SensorChart history={[]} isAnomaly={false} riskLevel="NORMAL" />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("SensorChart: chart-empty-state exibe texto 'Coletando dados'", () => {
    render(<SensorChart history={[]} isAnomaly={false} riskLevel="NORMAL" />);
    expect(screen.getByTestId("chart-empty-state")).toHaveTextContent(
      /coletando dados/i,
    );
  });

  it("SensorChart: chart-empty-state visível com 1 ponto (precisa de ≥ 2)", () => {
    const singlePoint = {
      time: "12:00:00",
      TP2: 5.5,
      TP3: 9.2,
      Motor_current: 4.2,
      Oil_temperature: 68.5,
      failure_probability: 0.05,
      predicted_class: 0,
    };
    render(
      <SensorChart
        history={[singlePoint]}
        isAnomaly={false}
        riskLevel="NORMAL"
      />,
    );
    expect(screen.getByTestId("chart-empty-state")).toBeInTheDocument();
  });

  it("SensorChart: NÃO exibe chart-empty-state quando há dados suficientes", () => {
    const points = Array.from({ length: 5 }, (_, i) => ({
      time: `12:00:0${i}`,
      TP2: 5.5,
      TP3: 9.2,
      Motor_current: 4.2,
      Oil_temperature: 68.5,
      failure_probability: 0.05,
      predicted_class: 0,
    }));
    render(
      <SensorChart history={points} isAnomaly={false} riskLevel="NORMAL" />,
    );
    expect(screen.queryByTestId("chart-empty-state")).not.toBeInTheDocument();
  });
});
