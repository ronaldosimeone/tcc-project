import { render, screen, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { PredictResponse } from "@/lib/api-client";
import SensorMonitor from "@/components/sensor-monitor";

// ── Mocks ────────────────────────────────────────────────────────────────────

const MOCK_RESPONSE: PredictResponse = {
  predicted_class: 0,
  failure_probability: 0.08,
  timestamp: new Date("2024-06-01T12:00:00Z").toISOString(),
};

vi.mock("@/lib/api-client", () => ({
  predict: vi.fn().mockResolvedValue(MOCK_RESPONSE),
}));

// Recharts usa ResizeObserver — polyfill necessário em jsdom
class MockResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = MockResizeObserver;

// ── Suite ────────────────────────────────────────────────────────────────────

describe("SensorMonitor", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    process.env.NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000";
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it("renderiza o título principal do painel", () => {
    render(<SensorMonitor />);
    expect(
      screen.getByRole("heading", { name: /monitoramento em tempo real/i }),
    ).toBeInTheDocument();
  });

  it("exibe os quatro KPI cards com labels corretos", () => {
    render(<SensorMonitor />);
    expect(screen.getByText(/pressão tp2/i)).toBeInTheDocument();
    expect(screen.getByText(/temperatura óleo/i)).toBeInTheDocument();
    expect(screen.getByText(/corrente motor/i)).toBeInTheDocument();
    expect(screen.getByText(/reservatório/i)).toBeInTheDocument();
  });

  it("exibe a seção do Painel de Sensores com todos os labels", () => {
    render(<SensorMonitor />);
    const sensorLabels = ["TP3", "H1", "DV Pressure", "COMP", "Towers", "MPG"];
    for (const label of sensorLabels) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
  });

  it("exibe o status NORMAL após a primeira chamada à API ter sido resolvida", async () => {
    render(<SensorMonitor />);
    // Avança timers para disparar o primeiro tick e resolver a Promise
    await act(async () => {
      await vi.runAllTimersAsync();
    });
    expect(screen.getByText("NORMAL")).toBeInTheDocument();
  });

  it("chama predict() a cada intervalo de polling", async () => {
    const { predict: mockPredict } = await import("@/lib/api-client");

    render(<SensorMonitor />);

    // Tick inicial
    await act(async () => {
      await vi.runAllTimersAsync();
    });

    // Avança mais 3 intervalos
    for (let i = 0; i < 3; i++) {
      await act(async () => {
        vi.advanceTimersByTime(3_000);
        await vi.runAllTimersAsync();
      });
    }

    // Deve ter chamado pelo menos 4 vezes (1 inicial + 3 por intervalo)
    expect(mockPredict).toHaveBeenCalledTimes(4);
  });

  it("exibe badge 'Backend offline' quando predict() lança erro", async () => {
    const { predict: mockPredict } = await import("@/lib/api-client");
    vi.mocked(mockPredict).mockRejectedValueOnce(new Error("Network Error"));

    render(<SensorMonitor />);
    await act(async () => {
      await vi.runAllTimersAsync();
    });

    expect(screen.getByText(/backend offline/i)).toBeInTheDocument();
  });
});
