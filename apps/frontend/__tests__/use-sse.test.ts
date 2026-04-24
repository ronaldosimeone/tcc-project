/**
 * Testes unitários para useSSE — RF-15 (reconexão com exponential backoff).
 *
 * MockEventSource simula a API do browser EventSource para testar:
 *  - criação de conexão
 *  - parsing de mensagens
 *  - transições de status
 *  - reconexão com backoff exponencial
 *  - cleanup ao desmontar
 *
 * Math.random é spionado para retornar 0, tornando o jitter determinístico:
 *   delay = backoff * (0.8 + 0 * 0.4) = backoff * 0.8
 */

import { renderHook, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useSSE } from "@/hooks/use-sse";
import type { SSEStatus } from "@/hooks/use-sse";

// ── MockEventSource ───────────────────────────────────────────────────────

class MockEventSource {
  static instances: MockEventSource[] = [];

  url: string;
  closed = false;
  onerror: ((e: Event) => void) | null = null;

  private readonly _listeners = new Map<string, Array<(e: Event) => void>>();

  constructor(url: string) {
    this.url = url;
    MockEventSource.instances.push(this);
  }

  addEventListener(type: string, listener: (e: Event) => void): void {
    if (!this._listeners.has(type)) this._listeners.set(type, []);
    this._listeners.get(type)!.push(listener);
  }

  removeEventListener(type: string, listener: (e: Event) => void): void {
    const list = this._listeners.get(type);
    if (!list) return;
    const idx = list.indexOf(listener);
    if (idx !== -1) list.splice(idx, 1);
  }

  dispatchNamedEvent(type: string, data: string): void {
    const event = new MessageEvent(type, { data });
    this._listeners.get(type)?.forEach((l) => l(event));
  }

  simulateError(): void {
    this.closed = true;
    this.onerror?.(new Event("error"));
  }

  close(): void {
    this.closed = true;
  }

  static reset(): void {
    MockEventSource.instances = [];
  }
}

// ── Suite ─────────────────────────────────────────────────────────────────

describe("useSSE", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    // Jitter determinístico: delay = backoff * 0.8
    vi.spyOn(Math, "random").mockReturnValue(0);
    MockEventSource.reset();
    vi.stubGlobal("EventSource", MockEventSource);
  });

  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  // ── Conexão ──────────────────────────────────────────────────────────────

  it("cria EventSource na URL correta ao montar", () => {
    renderHook(() =>
      useSSE({
        url: "/api/stream/sensors",
        eventName: "sensor_reading",
        onMessage: vi.fn(),
      }),
    );

    expect(MockEventSource.instances).toHaveLength(1);
    expect(MockEventSource.instances[0]!.url).toBe("/api/stream/sensors");
  });

  it("inicia com status 'connecting'", () => {
    const { result } = renderHook(() =>
      useSSE({
        url: "/api/stream/sensors",
        eventName: "sensor_reading",
        onMessage: vi.fn(),
      }),
    );

    expect(result.current.status).toBe<SSEStatus>("connecting");
    expect(result.current.reconnectAttempt).toBe(0);
  });

  // ── Recebimento de mensagens ──────────────────────────────────────────────

  it("chama onMessage com JSON parseado ao receber evento", async () => {
    const onMessage = vi.fn();

    renderHook(() =>
      useSSE({
        url: "/api/stream/sensors",
        eventName: "sensor_reading",
        onMessage,
      }),
    );

    await act(async () => {
      MockEventSource.instances[0]!.dispatchNamedEvent(
        "sensor_reading",
        JSON.stringify({ TP2: 5.5, timestamp: "2024-06-01T12:00:00Z" }),
      );
    });

    expect(onMessage).toHaveBeenCalledOnce();
    expect(onMessage).toHaveBeenCalledWith({
      TP2: 5.5,
      timestamp: "2024-06-01T12:00:00Z",
    });
  });

  it("transiciona para 'connected' após o primeiro evento", async () => {
    const { result } = renderHook(() =>
      useSSE({
        url: "/api/stream/sensors",
        eventName: "sensor_reading",
        onMessage: vi.fn(),
      }),
    );

    await act(async () => {
      MockEventSource.instances[0]!.dispatchNamedEvent(
        "sensor_reading",
        JSON.stringify({ TP2: 5.5 }),
      );
    });

    expect(result.current.status).toBe<SSEStatus>("connected");
  });

  it("ignora JSON malformado sem lançar exceção", async () => {
    const onMessage = vi.fn();

    renderHook(() =>
      useSSE({
        url: "/api/stream/sensors",
        eventName: "sensor_reading",
        onMessage,
      }),
    );

    await act(async () => {
      MockEventSource.instances[0]!.dispatchNamedEvent(
        "sensor_reading",
        "isto-nao-e-json",
      );
    });

    expect(onMessage).not.toHaveBeenCalled();
  });

  // ── Cleanup ───────────────────────────────────────────────────────────────

  it("fecha o EventSource ao desmontar", () => {
    const { unmount } = renderHook(() =>
      useSSE({
        url: "/api/stream/sensors",
        eventName: "sensor_reading",
        onMessage: vi.fn(),
      }),
    );

    const es = MockEventSource.instances[0]!;
    unmount();

    expect(es.closed).toBe(true);
  });

  // ── Reconexão com backoff (RF-15) ─────────────────────────────────────────

  it("transiciona para 'reconnecting' e incrementa reconnectAttempt após erro", async () => {
    const { result } = renderHook(() =>
      useSSE({
        url: "/api/stream/sensors",
        eventName: "sensor_reading",
        onMessage: vi.fn(),
      }),
    );

    await act(async () => {
      MockEventSource.instances[0]!.simulateError();
    });

    expect(result.current.status).toBe<SSEStatus>("reconnecting");
    expect(result.current.reconnectAttempt).toBe(1);
  });

  it("cria nova conexão após o delay de backoff do attempt 1 (800ms)", async () => {
    renderHook(() =>
      useSSE({
        url: "/api/stream/sensors",
        eventName: "sensor_reading",
        onMessage: vi.fn(),
      }),
    );

    // Attempt 1: backoff = 1000ms * 0.8 = 800ms
    await act(async () => {
      MockEventSource.instances[0]!.simulateError();
    });

    expect(MockEventSource.instances).toHaveLength(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(800);
    });

    expect(MockEventSource.instances).toHaveLength(2);
  });

  it("dobra o backoff a cada tentativa (exponential backoff)", async () => {
    renderHook(() =>
      useSSE({
        url: "/api/stream/sensors",
        eventName: "sensor_reading",
        onMessage: vi.fn(),
      }),
    );

    // Erro 1: backoff = 800ms → reconecta
    await act(async () => {
      MockEventSource.instances[0]!.simulateError();
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(800);
    });
    expect(MockEventSource.instances).toHaveLength(2);

    // Erro 2: backoff = 1600ms (2000ms * 0.8)
    await act(async () => {
      MockEventSource.instances[1]!.simulateError();
    });

    // Após 800ms ainda NÃO reconectou
    await act(async () => {
      await vi.advanceTimersByTimeAsync(800);
    });
    expect(MockEventSource.instances).toHaveLength(2);

    // Após mais 800ms (total 1600ms) reconecta
    await act(async () => {
      await vi.advanceTimersByTimeAsync(800);
    });
    expect(MockEventSource.instances).toHaveLength(3);
  });

  it("não cria nova conexão se desmontado durante o backoff", async () => {
    const { unmount } = renderHook(() =>
      useSSE({
        url: "/api/stream/sensors",
        eventName: "sensor_reading",
        onMessage: vi.fn(),
      }),
    );

    // Dispara erro (agenda reconexão em 800ms)
    await act(async () => {
      MockEventSource.instances[0]!.simulateError();
    });

    // Desmonta antes do timer disparar
    unmount();

    // Avança além do delay — não deve criar nova conexão
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });

    expect(MockEventSource.instances).toHaveLength(1);
  });

  it("reseta reconnectAttempt para 0 após reconexão bem-sucedida", async () => {
    const { result } = renderHook(() =>
      useSSE({
        url: "/api/stream/sensors",
        eventName: "sensor_reading",
        onMessage: vi.fn(),
      }),
    );

    // Erro → reconexão
    await act(async () => {
      MockEventSource.instances[0]!.simulateError();
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(800);
    });

    expect(result.current.reconnectAttempt).toBe(1);

    // Nova conexão recebe evento → reseta
    await act(async () => {
      MockEventSource.instances[1]!.dispatchNamedEvent(
        "sensor_reading",
        JSON.stringify({ TP2: 5.5 }),
      );
    });

    expect(result.current.reconnectAttempt).toBe(0);
    expect(result.current.status).toBe<SSEStatus>("connected");
  });
});
