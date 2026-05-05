/**
 * Testes de resiliência de conexão — RF-17 / RNF-35.
 *
 * Valida as transições de estado do modo degradado:
 *   Online → Queda de SSE → banner de degradação (dados preservados)
 *             → Recuperação → banner desaparece
 *
 * Strategy:
 *   TrackableEventSource registra instâncias e permite simulação de erros,
 *   possibilitando controlar o ciclo de reconexão do useSSE.
 *   Math.random é spionado (jitter = 0) para delays determinísticos (800ms).
 *
 * NOTA: Estes testes cobrem o modal de degradação (data-testid="degraded-mode-banner"),
 *   não o error-state (que exige histórico vazio). Histórico é pré-carregado via fetch.
 */

import { render, screen, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SensorMonitor from "@/components/sensor-monitor";

// ── Dados de teste ─────────────────────────────────────────────────────────

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

const MOCK_PAGE_WITH_DATA = {
  items: [MOCK_ITEM],
  total: 1,
  page: 1,
  size: 1,
  pages: 1,
};

// ── TrackableEventSource ───────────────────────────────────────────────────

class TrackableEventSource {
  static instances: TrackableEventSource[] = [];

  url: string;
  closed = false;
  onerror: ((e: Event) => void) | null = null;

  private readonly _listeners = new Map<string, Array<(e: Event) => void>>();

  constructor(url: string) {
    this.url = url;
    TrackableEventSource.instances.push(this);
  }

  addEventListener(type: string, listener: (e: Event) => void): void {
    if (!this._listeners.has(type)) this._listeners.set(type, []);
    this._listeners.get(type)!.push(listener);
  }

  removeEventListener(): void {}

  close(): void {
    this.closed = true;
  }

  simulateError(): void {
    this.closed = true;
    this.onerror?.(new Event("error"));
  }

  dispatchNamedEvent(type: string, data: string): void {
    const event = new MessageEvent(type, { data });
    this._listeners.get(type)?.forEach((l) => l(event));
  }

  static reset(): void {
    TrackableEventSource.instances = [];
  }

  static get latest(): TrackableEventSource {
    return TrackableEventSource.instances[
      TrackableEventSource.instances.length - 1
    ]!;
  }
}

// ── MockWebSocket (mesmo do sensor-monitor.test.tsx) ──────────────────────

class MockWebSocket {
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onopen: (() => void) | null = null;
  send = vi.fn();
  close = vi.fn();
}

// ── ResizeObserver polyfill ────────────────────────────────────────────────

class MockResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = MockResizeObserver;

// ── Suite ──────────────────────────────────────────────────────────────────

describe("RF-17 / RNF-35: Resiliência de Conexão e Modo Degradado", () => {
  let mockFetch: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.useFakeTimers();
    vi.spyOn(Math, "random").mockReturnValue(0); // jitter = 0 → delay = backoff * 0.8
    TrackableEventSource.reset();
    vi.stubGlobal("EventSource", TrackableEventSource);
    vi.stubGlobal("WebSocket", MockWebSocket);

    mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => MOCK_PAGE_WITH_DATA,
    });
    vi.stubGlobal("fetch", mockFetch);
    process.env.NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000";
  });

  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    vi.clearAllMocks();
  });

  // ── ConnectionStatus — RF-17 ──────────────────────────────────────────────

  it("RF-17: renderiza o painel de status de conexão", async () => {
    render(<SensorMonitor />);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(screen.getByTestId("connection-status")).toBeInTheDocument();
  });

  it("RF-17: exibe os canais SSE e WS no painel de status", async () => {
    render(<SensorMonitor />);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    const statusEl = screen.getByTestId("connection-status");
    expect(statusEl.textContent).toMatch(/sensores/i);
    expect(statusEl.textContent).toMatch(/alertas/i);
  });

  // ── RNF-35: Modo Degradado ────────────────────────────────────────────────

  it("RNF-35: NÃO exibe banner degradado quando SSE está conectando (initial load)", async () => {
    // Seed fetch nunca resolve → sem histórico → banner não pode aparecer
    // (banner exige status="reconnecting"|"error" + history.length > 0)
    mockFetch.mockImplementationOnce(() => new Promise(() => {}));

    render(<SensorMonitor />);
    // SSE em "connecting" → sem histórico ainda → sem banner
    expect(
      screen.queryByTestId("degraded-mode-banner"),
    ).not.toBeInTheDocument();
  });

  it("RNF-35: exibe banner degradado quando SSE perde conexão com dados em cache", async () => {
    render(<SensorMonitor />);

    // 1. Seed fetch resolve → histórico populado
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // 2. SSE cai → status = "reconnecting"
    await act(async () => {
      TrackableEventSource.instances[0]!.simulateError();
    });

    // Banner deve aparecer: temos histórico E SSE está reconnecting
    expect(screen.getByTestId("degraded-mode-banner")).toBeInTheDocument();
  });

  it("RNF-35: banner contém texto de modo degradado", async () => {
    render(<SensorMonitor />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    await act(async () => {
      TrackableEventSource.instances[0]!.simulateError();
    });

    const banner = screen.getByTestId("degraded-mode-banner");
    expect(banner.textContent).toMatch(/modo degradado/i);
    expect(banner.textContent).toMatch(/últimos dados conhecidos/i);
  });

  it("RNF-35: banner tem role='status' para acessibilidade", async () => {
    render(<SensorMonitor />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    await act(async () => {
      TrackableEventSource.instances[0]!.simulateError();
    });

    expect(screen.getByTestId("degraded-mode-banner")).toHaveAttribute(
      "role",
      "status",
    );
  });

  it("RNF-35: dados (KPI cards) permanecem visíveis após queda do SSE", async () => {
    render(<SensorMonitor />);

    // Resolve seed fetch → dados carregados
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // KPI cards estão visíveis
    expect(screen.queryByTestId("gauge-skeleton")).not.toBeInTheDocument();

    // SSE cai
    await act(async () => {
      TrackableEventSource.instances[0]!.simulateError();
    });

    // Dados ainda presentes — error-state NÃO deve aparecer
    expect(screen.queryByTestId("error-state")).not.toBeInTheDocument();
    // KPI cards ainda presentes
    expect(screen.getByText(/pressão tp2/i)).toBeInTheDocument();
  });

  it("RNF-35: badge de risco permanece visível em modo degradado", async () => {
    render(<SensorMonitor />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // Risco visível antes da queda
    expect(screen.getAllByText("NORMAL")[0]).toBeInTheDocument();

    await act(async () => {
      TrackableEventSource.instances[0]!.simulateError();
    });

    // Risco ainda visível após queda — dados em cache
    expect(screen.getAllByText("NORMAL")[0]).toBeInTheDocument();
  });

  it("RNF-35: banner desaparece quando SSE reconecta com sucesso", async () => {
    render(<SensorMonitor />);

    // Seed fetch
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // SSE cai → banner aparece
    await act(async () => {
      TrackableEventSource.instances[0]!.simulateError();
    });

    expect(screen.getByTestId("degraded-mode-banner")).toBeInTheDocument();

    // Backoff attempt 1: 1000ms * 0.8 = 800ms → nova instância do EventSource criada
    await act(async () => {
      await vi.advanceTimersByTimeAsync(800);
    });

    // Nova instância criada; simula evento de dados recebidos → status = "connected"
    await act(async () => {
      TrackableEventSource.latest.dispatchNamedEvent(
        "sensor_reading",
        JSON.stringify({
          timestamp: "2024-06-01T12:00:01.000Z",
          TP2: 5.6,
          TP3: 9.1,
          H1: 7.2,
          DV_pressure: 0.1,
          Reservoirs: 8.9,
          Motor_current: 4.3,
          Oil_temperature: 68.6,
          COMP: 1,
          DV_eletric: 0,
          Towers: 1,
          MPG: 0,
          Oil_level: 1,
        }),
      );
    });

    // Banner deve ter desaparecido (status = "connected")
    expect(
      screen.queryByTestId("degraded-mode-banner"),
    ).not.toBeInTheDocument();
  });

  it("RNF-35: NÃO exibe banner quando seed fetch falha (ErrorState aparece em seu lugar)", async () => {
    mockFetch.mockRejectedValueOnce(new Error("Network Error"));

    render(<SensorMonitor />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // Hard offline: sem histórico → ErrorState (não banner degradado)
    expect(screen.getByTestId("error-state")).toBeInTheDocument();
    expect(
      screen.queryByTestId("degraded-mode-banner"),
    ).not.toBeInTheDocument();
  });
});
