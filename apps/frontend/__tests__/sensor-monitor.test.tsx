import { render, screen, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SensorMonitor from "@/components/sensor-monitor";
import { POLL_INTERVAL_MS } from "@/hooks/use-sensor-data";

// ── Mocks ─────────────────────────────────────────────────────────────────
//
// vi.mock() é hoistado pelo Vitest ANTES de todas as declarações de variável.
// Por isso MOCK_RESPONSE deve ser criado com vi.hoisted() — garante que o
// valor esteja disponível quando o factory do mock for executado.

const { MOCK_RESPONSE } = vi.hoisted(() => ({
  MOCK_RESPONSE: {
    predicted_class: 0 as const,
    failure_probability: 0.08,
    timestamp: new Date("2024-06-01T12:00:00Z").toISOString(),
  },
}));

vi.mock("@/lib/api-client", () => ({
  predict: vi.fn().mockResolvedValue(MOCK_RESPONSE),
}));

// Recharts usa ResizeObserver internamente — polyfill necessário em jsdom
class MockResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = MockResizeObserver;

// ── Suite ──────────────────────────────────────────────────────────────────

describe("SensorMonitor", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    process.env.NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000";
  });

  afterEach(() => {
    vi.clearAllTimers(); // cancela setInterval antes de restaurar os timers
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

  it("exibe a seção do Painel de Sensores com os labels corretos", () => {
    render(<SensorMonitor />);
    // Usa os labels exatos que sensor-monitor.tsx renderiza nos SensorChips
    const sensorLabels = ["TP3", "H1", "DV Press.", "COMP", "Towers"];
    for (const label of sensorLabels) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
  });

  it("exibe o status NORMAL após a primeira chamada à API ser resolvida", async () => {
    render(<SensorMonitor />);

    // advanceTimersByTimeAsync(0) avança 0ms mas aguarda todas as Promises
    // pendentes — resolve o void tick() inicial sem disparar o setInterval
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(screen.getAllByText("NORMAL")[0]).toBeInTheDocument();
  });

  it("chama predict() a cada intervalo de polling", async () => {
    const { predict: mockPredict } = await import("@/lib/api-client");
    render(<SensorMonitor />);

    // resolve o tick inicial
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // avança 3 intervalos usando o POLL_INTERVAL_MS real do hook (5 s)
    for (let i = 0; i < 3; i++) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS);
      });
    }

    // 1 chamada inicial + 3 por intervalo
    expect(mockPredict).toHaveBeenCalledTimes(4);
  });

  it("exibe badge 'Backend offline' quando predict() lança erro", async () => {
    const { predict: mockPredict } = await import("@/lib/api-client");
    vi.mocked(mockPredict).mockRejectedValueOnce(new Error("Network Error"));

    render(<SensorMonitor />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(screen.getByText(/backend offline/i)).toBeInTheDocument();
  });
});
