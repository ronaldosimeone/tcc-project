/**
 * Testes unitários para useSensorData — RF-06 / RF-07 (SSE + prediction poll).
 *
 * Estratégia após refatoração para SSE:
 *  - EventSource mockado globalmente (simula /api/stream/sensors)
 *  - WebSocket mockado globalmente (evita conexões reais)
 *  - fetch mockado para seed (size=30) e prediction poll (size=1)
 *  - Eventos SSE simulados via MockEventSource.dispatchNamedEvent
 *
 * Por que advanceTimersByTimeAsync:
 *   O hook usa setInterval para o prediction poll. advanceTimersByTimeAsync(ms)
 *   avança exatamente `ms` ms, dispara os callbacks devidos e aguarda Promises
 *   pendentes — sem risco de loop infinito.
 */

import { renderHook, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { PredictResponse } from "@/lib/api-client";
import {
  getRiskLevel,
  POLL_INTERVAL_MS,
  useSensorData,
} from "@/hooks/use-sensor-data";

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

  removeEventListener(): void {}

  dispatchNamedEvent(type: string, data: string): void {
    const event = new MessageEvent(type, { data });
    this._listeners.get(type)?.forEach((l) => l(event));
  }

  close(): void {
    this.closed = true;
  }

  static reset(): void {
    MockEventSource.instances = [];
  }
}

// ── Fixtures ──────────────────────────────────────────────────────────────

const BASE_READING = {
  timestamp: "2024-06-01T12:00:00.000Z",
  TP2: 5.5,
  TP3: 9.2,
  H1: 8.8,
  DV_pressure: 2.1,
  Reservoirs: 8.7,
  Motor_current: 4.2,
  Oil_temperature: 68.5,
  COMP: 1.0,
  DV_eletric: 0.0,
  Towers: 1.0,
  MPG: 1.0,
  Oil_level: 1.0,
};

const BASE_ITEM = {
  ...BASE_READING,
  failure_probability: 0.06,
  predicted_class: 0,
};

function makePageResponse(item = BASE_ITEM) {
  return {
    ok: true,
    json: () =>
      Promise.resolve({
        items: [item],
        total: 1,
        page: 1,
        size: 1,
        pages: 1,
      }),
  } as unknown as Response;
}

// ── getRiskLevel (função pura — sem timers) ───────────────────────────────

describe("getRiskLevel", () => {
  it("retorna NORMAL para prob < 0.3", () => {
    expect(getRiskLevel(0)).toBe("NORMAL");
    expect(getRiskLevel(0.1)).toBe("NORMAL");
    expect(getRiskLevel(0.299)).toBe("NORMAL");
  });

  it("retorna ALERTA para 0.3 <= prob < 0.65", () => {
    expect(getRiskLevel(0.3)).toBe("ALERTA");
    expect(getRiskLevel(0.5)).toBe("ALERTA");
    expect(getRiskLevel(0.649)).toBe("ALERTA");
  });

  it("retorna CRÍTICO para prob >= 0.65", () => {
    expect(getRiskLevel(0.65)).toBe("CRÍTICO");
    expect(getRiskLevel(0.9)).toBe("CRÍTICO");
    expect(getRiskLevel(1)).toBe("CRÍTICO");
  });
});

// ── useSensorData ─────────────────────────────────────────────────────────

describe("useSensorData", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.useFakeTimers();
    vi.spyOn(Math, "random").mockReturnValue(0); // jitter determinístico

    MockEventSource.reset();
    vi.stubGlobal("EventSource", MockEventSource);
    vi.stubGlobal(
      "WebSocket",
      class {
        onmessage = null;
        onerror = null;
        send = vi.fn();
        close = vi.fn();
      },
    );

    // fetch retorna timestamps únicos por chamada (evita dedup no prediction poll)
    let callCount = 0;
    fetchMock = vi.fn().mockImplementation(() => {
      callCount++;
      return Promise.resolve(
        makePageResponse({
          ...BASE_ITEM,
          timestamp: new Date(
            Date.UTC(2024, 5, 1, 12, 0, callCount),
          ).toISOString(),
        }),
      );
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  // ── Estado inicial ──────────────────────────────────────────────────────

  it("começa com isLoading=true, history vazio e error null", () => {
    // Seed fetch fica pendente → estado inicial preservado e sem setState
    // posterior que dispararia warning de act() após o assert.
    fetchMock.mockImplementationOnce(() => new Promise(() => {}));

    const { result } = renderHook(() => useSensorData());

    expect(result.current.isLoading).toBe(true);
    expect(result.current.history).toHaveLength(0);
    expect(result.current.error).toBeNull();
    expect(result.current.latest).toBeNull();
  });

  it("começa com isAnomaly=false e riskLevel NORMAL", () => {
    fetchMock.mockImplementationOnce(() => new Promise(() => {}));

    const { result } = renderHook(() => useSensorData());

    expect(result.current.isAnomaly).toBe(false);
    expect(result.current.riskLevel).toBe("NORMAL");
  });

  // ── Seed (fetch ao montar) ──────────────────────────────────────────────

  it("isLoading passa para false após o seed fetch", async () => {
    const { result } = renderHook(() => useSensorData());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.isLoading).toBe(false);
  });

  it("popula histórico com itens do seed fetch", async () => {
    const { result } = renderHook(() => useSensorData());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.history).toHaveLength(1);
  });

  it("ponto do histórico contém as 4 séries de sensor (RF-06)", async () => {
    const { result } = renderHook(() => useSensorData());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    const point = result.current.history[0];
    expect(point).toBeDefined();
    expect(typeof point!.TP2).toBe("number");
    expect(typeof point!.TP3).toBe("number");
    expect(typeof point!.Motor_current).toBe("number");
    expect(typeof point!.Oil_temperature).toBe("number");
    expect(typeof point!.failure_probability).toBe("number");
  });

  it("latest reflete o item mais recente do seed", async () => {
    fetchMock.mockImplementationOnce(() =>
      Promise.resolve(makePageResponse(BASE_ITEM)),
    );

    const { result } = renderHook(() => useSensorData());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.latest).toEqual<PredictResponse>({
      failure_probability: BASE_ITEM.failure_probability,
      predicted_class: BASE_ITEM.predicted_class,
      timestamp: BASE_ITEM.timestamp,
    });
  });

  // ── SSE: eventos em tempo real ──────────────────────────────────────────

  it("adiciona ponto ao histórico a cada evento SSE", async () => {
    const { result } = renderHook(() => useSensorData());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    const beforeCount = result.current.history.length;
    const es = MockEventSource.instances[0]!;

    await act(async () => {
      es.dispatchNamedEvent(
        "sensor_reading",
        JSON.stringify({
          ...BASE_READING,
          timestamp: "2024-06-01T12:00:10.000Z",
        }),
      );
    });

    expect(result.current.history).toHaveLength(beforeCount + 1);
  });

  it("atualiza currentPayload com dados do evento SSE", async () => {
    const { result } = renderHook(() => useSensorData());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    const es = MockEventSource.instances[0]!;

    await act(async () => {
      es.dispatchNamedEvent(
        "sensor_reading",
        JSON.stringify({
          ...BASE_READING,
          timestamp: "2024-06-01T12:00:10.000Z",
          TP2: 7.77,
        }),
      );
    });

    expect(result.current.currentPayload.TP2).toBe(7.77);
  });

  it("não acumula duplicatas (dedup por timestamp no SSE)", async () => {
    const { result } = renderHook(() => useSensorData());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    const beforeCount = result.current.history.length;
    const es = MockEventSource.instances[0]!;
    const sameTs = "2024-06-01T12:00:10.000Z";

    await act(async () => {
      es.dispatchNamedEvent(
        "sensor_reading",
        JSON.stringify({ ...BASE_READING, timestamp: sameTs }),
      );
      es.dispatchNamedEvent(
        "sensor_reading",
        JSON.stringify({ ...BASE_READING, timestamp: sameTs }),
      );
    });

    // Apenas 1 ponto novo — duplicata ignorada
    expect(result.current.history).toHaveLength(beforeCount + 1);
  });

  it("mantém no máximo 60 pontos na janela deslizante (RNF-33)", async () => {
    const { result } = renderHook(() => useSensorData());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    const es = MockEventSource.instances[0]!;

    // Emite 65 eventos com timestamps únicos
    await act(async () => {
      for (let i = 1; i <= 65; i++) {
        const ts = new Date(Date.UTC(2024, 5, 1, 12, 1, i)).toISOString();
        es.dispatchNamedEvent(
          "sensor_reading",
          JSON.stringify({ ...BASE_READING, timestamp: ts }),
        );
      }
    });

    expect(result.current.history.length).toBeLessThanOrEqual(60);
  });

  it("isLoading false ao receber primeiro evento SSE (mesmo sem seed)", async () => {
    fetchMock.mockRejectedValueOnce(new Error("fail")); // seed falha

    const { result } = renderHook(() => useSensorData());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // Seed falhou — error está setado, isLoading false
    expect(result.current.isLoading).toBe(false);

    // SSE entrega dado → limpa error
    const es = MockEventSource.instances[0]!;
    await act(async () => {
      es.dispatchNamedEvent(
        "sensor_reading",
        JSON.stringify({
          ...BASE_READING,
          timestamp: "2024-06-01T12:00:05.000Z",
        }),
      );
    });

    expect(result.current.error).toBeNull();
  });

  // ── Prediction poll ─────────────────────────────────────────────────────

  it("atualiza latest após POLL_INTERVAL_MS (prediction poll)", async () => {
    const { result } = renderHook(() => useSensorData());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // Configura o próximo fetch (prediction poll) para retornar CRÍTICO
    fetchMock.mockImplementationOnce(() =>
      Promise.resolve(
        makePageResponse({
          ...BASE_ITEM,
          failure_probability: 0.9,
          predicted_class: 1,
          timestamp: "2024-06-01T12:00:10.000Z",
        }),
      ),
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS);
    });

    expect(result.current.latest?.failure_probability).toBe(0.9);
    expect(result.current.riskLevel).toBe("CRÍTICO");
  });

  it("chama fetch novamente após POLL_INTERVAL_MS", async () => {
    renderHook(() => useSensorData());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    const callsAfterSeed = fetchMock.mock.calls.length;

    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS);
    });

    expect(fetchMock.mock.calls.length).toBeGreaterThan(callsAfterSeed);
  });

  // ── Estado de erro ──────────────────────────────────────────────────────

  it("define error quando seed fetch falha e não há histórico", async () => {
    fetchMock.mockRejectedValueOnce(new Error("Network Error"));

    const { result } = renderHook(() => useSensorData());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.error).toBeInstanceOf(Error);
    expect(result.current.error?.message).toBe("Network Error");
  });

  it("isLoading passa para false mesmo quando seed fetch falha", async () => {
    fetchMock.mockRejectedValueOnce(new Error("Timeout"));

    const { result } = renderHook(() => useSensorData());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.isLoading).toBe(false);
  });

  it("define error quando seed fetch retorna status não-ok", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: false,
      status: 503,
    } as unknown as Response);

    const { result } = renderHook(() => useSensorData());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.error).toBeInstanceOf(Error);
  });

  it("limpa error ao receber evento SSE após falha do seed", async () => {
    fetchMock.mockRejectedValueOnce(new Error("Network Error"));

    const { result } = renderHook(() => useSensorData());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(result.current.error).not.toBeNull();

    const es = MockEventSource.instances[0]!;
    await act(async () => {
      es.dispatchNamedEvent(
        "sensor_reading",
        JSON.stringify({
          ...BASE_READING,
          timestamp: "2024-06-01T12:00:05.000Z",
        }),
      );
    });

    expect(result.current.error).toBeNull();
  });

  // ── Anomalia (RF-07) ────────────────────────────────────────────────────

  it("isAnomaly=true quando failure_probability >= 0.3 (via seed)", async () => {
    fetchMock.mockImplementationOnce(() =>
      Promise.resolve(
        makePageResponse({
          ...BASE_ITEM,
          failure_probability: 0.5,
          predicted_class: 1,
        }),
      ),
    );

    const { result } = renderHook(() => useSensorData());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.isAnomaly).toBe(true);
    expect(result.current.riskLevel).toBe("ALERTA");
  });

  it("riskLevel=CRÍTICO quando failure_probability >= 0.65", async () => {
    fetchMock.mockImplementationOnce(() =>
      Promise.resolve(
        makePageResponse({
          ...BASE_ITEM,
          failure_probability: 0.8,
          predicted_class: 1,
        }),
      ),
    );

    const { result } = renderHook(() => useSensorData());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.riskLevel).toBe("CRÍTICO");
  });

  it("failure_probability no ponto do histórico é um float válido", async () => {
    fetchMock.mockImplementationOnce(() =>
      Promise.resolve(
        makePageResponse({
          ...BASE_ITEM,
          failure_probability: 0.123456789,
        }),
      ),
    );

    const { result } = renderHook(() => useSensorData());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    const prob = result.current.history[0]?.failure_probability;
    expect(typeof prob).toBe("number");
    expect(Number.isFinite(prob)).toBe(true);
  });

  // ── Cleanup ─────────────────────────────────────────────────────────────

  it("fecha o EventSource ao desmontar", async () => {
    const { unmount } = renderHook(() => useSensorData());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    const es = MockEventSource.instances[0]!;
    unmount();

    expect(es.closed).toBe(true);
  });

  it("para o prediction poll ao desmontar", async () => {
    const { unmount } = renderHook(() => useSensorData());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    const callsAtUnmount = fetchMock.mock.calls.length;
    unmount();

    await act(async () => {
      vi.advanceTimersByTime(POLL_INTERVAL_MS * 3);
    });

    expect(fetchMock.mock.calls.length).toBe(callsAtUnmount);
  });
});
