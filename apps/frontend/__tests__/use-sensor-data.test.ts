/**
 * Testes unitários para useSensorData — RF-06 / RF-07.
 *
 * Por que advanceTimersByTimeAsync e não run*TimersAsync:
 *
 *   O hook usa setInterval. Com vi.runAllTimersAsync() o intervalo
 *   fica re-disparando até bater no limite de 10 000 timers (loop
 *   infinito). Com vi.runOnlyPendingTimersAsync() o loop também pode
 *   ocorrer se o timer recém agendado for considerado "pending".
 *
 *   vi.advanceTimersByTimeAsync(ms) é a API correta: avança o relógio
 *   exatamente `ms` milissegundos, dispara apenas os callbacks que
 *   deveriam ocorrer nesse intervalo, e aguarda todas as Promises
 *   geradas — sem risco de loop infinito.
 *
 *   Para o tick inicial (void tick() direto no useEffect, não via timer),
 *   usamos advanceTimersByTimeAsync(0): não dispara nenhum timer, mas
 *   aguarda as Promises pendentes na fila de microtasks.
 */

import { renderHook, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { PredictResponse } from "@/lib/api-client";
import {
  getRiskLevel,
  POLL_INTERVAL_MS,
  useSensorData,
} from "@/hooks/use-sensor-data";

// ── Mocks ─────────────────────────────────────────────────────────────────

const SUCCESS_RESPONSE: PredictResponse = {
  predicted_class: 0,
  failure_probability: 0.06,
  timestamp: new Date("2024-06-01T12:00:00Z").toISOString(),
};

vi.mock("@/lib/api-client", () => ({
  predict: vi.fn(),
}));

async function getMockPredict() {
  const mod = await import("@/lib/api-client");
  return vi.mocked(mod.predict);
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
  beforeEach(async () => {
    vi.useFakeTimers();
    process.env.NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000";
    // Configura o mock padrão para todos os testes do bloco
    const mockPredict = await getMockPredict();
    mockPredict.mockResolvedValue(SUCCESS_RESPONSE);
  });

  afterEach(() => {
    vi.clearAllTimers(); // cancela o setInterval antes de restaurar os timers
    vi.useRealTimers();
    vi.clearAllMocks();
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

  // ── Após o primeiro tick (direto — não via timer) ───────────────────────

  it("isLoading passa para false após o primeiro tick", async () => {
    const { result } = renderHook(() => useSensorData());

    // avança 0 ms: nenhum timer dispara, mas as Promises pendentes (tick inicial)
    // são aguardadas e os setState subsequentes são processados pelo act()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.isLoading).toBe(false);
  });

  it("adiciona 1 ponto ao histórico após o primeiro tick", async () => {
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

  it("latest reflete a resposta do backend após o primeiro tick", async () => {
    const { result } = renderHook(() => useSensorData());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.latest).toEqual(SUCCESS_RESPONSE);
  });

  // ── Polling (RF-06 — 5 s) ──────────────────────────────────────────────

  it("chama predict novamente após POLL_INTERVAL_MS", async () => {
    const mockPredict = await getMockPredict();
    renderHook(() => useSensorData());

    // resolve o tick inicial
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    const callsAfterFirst = mockPredict.mock.calls.length;

    // dispara exatamente 1 tick do setInterval
    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS);
    });

    expect(mockPredict.mock.calls.length).toBeGreaterThan(callsAfterFirst);
  });

  it("acumula pontos no histórico a cada tick (RF-06)", async () => {
    const { result } = renderHook(() => useSensorData());

    // tick inicial
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // 3 ticks via setInterval
    for (let i = 0; i < 3; i++) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS);
      });
    }

    // 1 inicial + 3 via timer = 4 pontos
    expect(result.current.history.length).toBe(4);
  });

  // ── Estado de erro ──────────────────────────────────────────────────────

  it("define error quando predict() lança exceção", async () => {
    const mockPredict = await getMockPredict();
    mockPredict.mockRejectedValueOnce(new Error("Network Error"));

    const { result } = renderHook(() => useSensorData());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.error).toBeInstanceOf(Error);
    expect(result.current.error?.message).toBe("Network Error");
  });

  it("isLoading passa para false mesmo quando predict() falha", async () => {
    const mockPredict = await getMockPredict();
    mockPredict.mockRejectedValueOnce(new Error("Timeout"));

    const { result } = renderHook(() => useSensorData());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.isLoading).toBe(false);
  });

  it("limpa o error em um tick bem-sucedido após falha", async () => {
    const mockPredict = await getMockPredict();
    mockPredict.mockRejectedValueOnce(new Error("Temporário"));

    const { result } = renderHook(() => useSensorData());

    // tick com erro
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(result.current.error).not.toBeNull();

    // próximo tick — sucesso (mockResolvedValue já está configurado no beforeEach)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS);
    });

    expect(result.current.error).toBeNull();
  });

  // ── Anomalia (RF-07) ────────────────────────────────────────────────────

  it("isAnomaly=true quando failure_probability >= 0.3", async () => {
    const mockPredict = await getMockPredict();
    mockPredict.mockResolvedValueOnce({
      ...SUCCESS_RESPONSE,
      failure_probability: 0.5,
      predicted_class: 1,
    });

    const { result } = renderHook(() => useSensorData());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.isAnomaly).toBe(true);
    expect(result.current.riskLevel).toBe("ALERTA");
  });

  it("riskLevel=CRÍTICO quando failure_probability >= 0.65", async () => {
    const mockPredict = await getMockPredict();
    mockPredict.mockResolvedValueOnce({
      ...SUCCESS_RESPONSE,
      failure_probability: 0.8,
      predicted_class: 1,
    });

    const { result } = renderHook(() => useSensorData());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.riskLevel).toBe("CRÍTICO");
  });

  // ── Cleanup ─────────────────────────────────────────────────────────────

  it("para o polling quando o componente é desmontado", async () => {
    const mockPredict = await getMockPredict();

    const { unmount } = renderHook(() => useSensorData());

    // tick inicial
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    const callsAtUnmount = mockPredict.mock.calls.length;

    // unmount() chama o cleanup do useEffect → clearInterval
    unmount();

    // avança 3 intervalos — o polling deve estar cancelado
    vi.advanceTimersByTime(POLL_INTERVAL_MS * 3);

    expect(mockPredict.mock.calls.length).toBe(callsAtUnmount);
  });
});
