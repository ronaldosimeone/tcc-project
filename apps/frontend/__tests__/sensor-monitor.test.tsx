import { render, screen, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SensorMonitor from "@/components/sensor-monitor";
import { POLL_INTERVAL_MS } from "@/hooks/use-sensor-data";

// ── Dados de teste ─────────────────────────────────────────────────────────
//
// O hook usa GET /api/v1/predictions (polling passivo), não mais predict().
// O mock retorna uma HistoryPage com um único HistoryItem.

const MOCK_ITEM = {
  failure_probability: 0.08,
  predicted_class: 0,
  timestamp: "2024-06-01T12:00:00.000Z",
  TP2: 5.5,
  TP3: 9.2,
  H1: 7.1,
  DV_pressure: 0.1,
  Reservoirs: 8.8,
  Motor_current: 4.2,
  Oil_temperature: 68.5,
  COMP: 1,
  DV_eletric: 0,
  Towers: 1,
  MPG: 0,
  Oil_level: 1,
};

const MOCK_PAGE = { items: [MOCK_ITEM], total: 1, page: 1, size: 1, pages: 1 };

// ── Polyfills de ambiente jsdom ────────────────────────────────────────────

// Recharts usa ResizeObserver internamente
class MockResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = MockResizeObserver;

// O hook abre WebSocket no useEffect; o mock evita conexões reais e warnings de act()
class MockWebSocket {
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  send = vi.fn();
  close = vi.fn();
}
globalThis.WebSocket = MockWebSocket as unknown as typeof WebSocket;

// ── Suite ──────────────────────────────────────────────────────────────────

describe("SensorMonitor", () => {
  let mockFetch: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.useFakeTimers();
    process.env.NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000";
    mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => MOCK_PAGE,
    });
    vi.stubGlobal("fetch", mockFetch);
  });

  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
    vi.unstubAllGlobals();
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

    // advanceTimersByTimeAsync(0) avança 0 ms mas aguarda todas as Promises
    // pendentes — resolve o void tick() inicial sem disparar o setInterval
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(screen.getAllByText("NORMAL")[0]).toBeInTheDocument();
  });

  it("chama fetch GET /predictions a cada intervalo de polling", async () => {
    render(<SensorMonitor />);

    // Resolve o tick inicial (size=30)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // Avança 3 intervalos usando o POLL_INTERVAL_MS real do hook (5 s)
    for (let i = 0; i < 3; i++) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS);
      });
    }

    // 1 chamada inicial + 3 por intervalo
    expect(mockFetch).toHaveBeenCalledTimes(4);
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/v1/predictions"),
      expect.objectContaining({ cache: "no-store" }),
    );
  });

  it("exibe badge 'Backend offline' quando fetch lança erro", async () => {
    mockFetch.mockRejectedValueOnce(new Error("Network Error"));

    render(<SensorMonitor />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(screen.getByText(/backend offline/i)).toBeInTheDocument();
  });
});
