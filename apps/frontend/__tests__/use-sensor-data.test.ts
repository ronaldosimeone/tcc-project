/**
 * Testes unitários para useSensorData — RF-06 / RF-07.
 *
 * Estratégia após refatoração para polling passivo:
 *  - `predict` não é mais chamado; o hook usa `fetch` global.
 *  - `fetch` é mockado via vi.stubGlobal para cada suite.
 *  - A deduplicação por timestamp exige respostas com timestamps distintos
 *    nos testes que avançam múltiplos ticks.
 *
 * Por que advanceTimersByTimeAsync e não run*TimersAsync:
 *   O hook usa setInterval. vi.runAllTimersAsync() re-dispara o intervalo
 *   até o limite de 10 000 timers. vi.advanceTimersByTimeAsync(ms) avança
 *   exatamente `ms` ms, dispara só os callbacks devidos e aguarda Promises
 *   geradas — sem risco de loop infinito.
 */

import { renderHook, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { PredictResponse } from "@/lib/api-client";
import {
  getRiskLevel,
  POLL_INTERVAL_MS,
  useSensorData,
} from "@/hooks/use-sensor-data";

// ── Fixtures ──────────────────────────────────────────────────────────────

const BASE_ITEM = {
  id: 1,
  failure_probability: 0.06,
  predicted_class: 0,
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

const SUCCESS_RESPONSE: PredictResponse = {
  predicted_class: BASE_ITEM.predicted_class,
  failure_probability: BASE_ITEM.failure_probability,
  timestamp: BASE_ITEM.timestamp,
};

function makePageResponse(item: typeof BASE_ITEM) {
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
  // fetchMock retorna timestamps únicos por chamada para evitar deduplicação
  let callCount: number;
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.useFakeTimers();
    callCount = 0;

    fetchMock = vi.fn().mockImplementation(() => {
      callCount++;
      const ts = new Date(Date.UTC(2024, 5, 1, 12, 0, callCount)).toISOString();
      return Promise.resolve(makePageResponse({ ...BASE_ITEM, timestamp: ts }));
    });

    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  // ── Estado inicial ──────────────────────────────────────────────────────

  it("começa com isLoading=true, history vazio e error null", () => {
    const { result } = renderHook(() => useSensorData());

    expect(result.current.isLoading).toBe(true);
    expect(result.current.history).toHaveLength(0);
    expect(result.current.error).toBeNull();
    expect(result.current.latest).toBeNull();
  });

  it("começa com isAnomaly=false e riskLevel NORMAL", () => {
    const { result } = renderHook(() => useSensorData());

    expect(result.current.isAnomaly).toBe(false);
    expect(result.current.riskLevel).toBe("NORMAL");
  });

  // ── Após o primeiro tick (seed — size=30) ──────────────────────────────

  it("isLoading passa para false após o primeiro tick", async () => {
    const { result } = renderHook(() => useSensorData());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.isLoading).toBe(false);
  });

  it("adiciona 1 ponto ao histórico após o primeiro tick (seed)", async () => {
    const { result } = renderHook(() => useSensorData());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.history).toHaveLength(1);
  });

  it("ponto do histórico contém as 4 séries exigidas por RF-06", async () => {
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

  it("latest reflete a resposta do banco após o primeiro tick", async () => {
    // Override: primeiro tick usa timestamp do BASE_ITEM para comparar
    fetchMock.mockImplementationOnce(() =>
      Promise.resolve(makePageResponse(BASE_ITEM)),
    );

    const { result } = renderHook(() => useSensorData());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.latest).toEqual(SUCCESS_RESPONSE);
  });

  // ── Polling passivo (RF-06 — 5 s) ─────────────────────────────────────

  it("chama fetch novamente após POLL_INTERVAL_MS", async () => {
    renderHook(() => useSensorData());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    const callsAfterFirst = fetchMock.mock.calls.length;

    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS);
    });

    expect(fetchMock.mock.calls.length).toBeGreaterThan(callsAfterFirst);
  });

  it("acumula pontos no histórico a cada tick (RF-06)", async () => {
    const { result } = renderHook(() => useSensorData());

    // tick inicial (seed)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // 3 ticks via setInterval — cada um com timestamp único (dedup não bloqueia)
    for (let i = 0; i < 3; i++) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS);
      });
    }

    // 1 seed + 3 incrementais = 4 pontos
    expect(result.current.history.length).toBe(4);
  });

  it("não acumula duplicatas quando o timestamp não muda", async () => {
    // Todos os ticks retornam o mesmo timestamp
    const fixedTs = "2024-06-01T12:00:00.000Z";
    fetchMock.mockImplementation(() =>
      Promise.resolve(makePageResponse({ ...BASE_ITEM, timestamp: fixedTs })),
    );

    const { result } = renderHook(() => useSensorData());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS);
    });

    // Apenas o seed (isFirst=true ignora dedup) adiciona 1 ponto
    expect(result.current.history.length).toBe(1);
  });

  // ── Estado de erro ──────────────────────────────────────────────────────

  it("define error quando fetch() lança exceção", async () => {
    fetchMock.mockRejectedValueOnce(new Error("Network Error"));

    const { result } = renderHook(() => useSensorData());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.error).toBeInstanceOf(Error);
    expect(result.current.error?.message).toBe("Network Error");
  });

  it("isLoading passa para false mesmo quando fetch() falha", async () => {
    fetchMock.mockRejectedValueOnce(new Error("Timeout"));

    const { result } = renderHook(() => useSensorData());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.isLoading).toBe(false);
  });

  it("define error quando fetch() retorna status não-ok", async () => {
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

  it("limpa o error em um tick bem-sucedido após falha", async () => {
    fetchMock.mockRejectedValueOnce(new Error("Temporário"));

    const { result } = renderHook(() => useSensorData());

    // tick com erro
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(result.current.error).not.toBeNull();

    // próximo tick — sucesso (fetchMock tem timestamp único para não deduplicar)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS);
    });

    expect(result.current.error).toBeNull();
  });

  // ── Anomalia (RF-07) ────────────────────────────────────────────────────

  it("isAnomaly=true quando failure_probability >= 0.3", async () => {
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

  // ── failure_probability como float (formato esperado pelo backend) ──────

  it("failure_probability no ponto do histórico é um float válido", async () => {
    fetchMock.mockImplementationOnce(() =>
      Promise.resolve(
        makePageResponse({ ...BASE_ITEM, failure_probability: 0.123456789 }),
      ),
    );

    const { result } = renderHook(() => useSensorData());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    const prob = result.current.history[0]?.failure_probability;
    expect(typeof prob).toBe("number");
    expect(Number.isFinite(prob)).toBe(true);
    // itemToPoint usa toFixed(4) → máximo 4 casas decimais
    expect(prob).toBe(parseFloat((0.123456789).toFixed(4)));
  });

  // ── Cleanup ─────────────────────────────────────────────────────────────

  it("para o polling quando o componente é desmontado", async () => {
    const { unmount } = renderHook(() => useSensorData());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    const callsAtUnmount = fetchMock.mock.calls.length;
    unmount();

    vi.advanceTimersByTime(POLL_INTERVAL_MS * 3);

    expect(fetchMock.mock.calls.length).toBe(callsAtUnmount);
  });
});
