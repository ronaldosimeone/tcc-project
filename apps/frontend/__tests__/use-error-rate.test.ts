/**
 * Testes de `useErrorRateStatus` (RNF-77).
 *
 * Mesmo padrão de fake timers já usado em sensor-monitor.test.tsx para o
 * prediction poll — aqui isolado no hook, sem montar o Dashboard inteiro.
 */

import { renderHook, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useErrorRateStatus } from "@/hooks/use-error-rate";

const OK_RESPONSE = (status: string) => ({
  ok: true,
  json: async () => ({
    status,
    error_rate: 0.0,
    window: "5m",
    threshold_warning: 0.01,
    threshold_critical: 0.05,
    prometheus_reachable: true,
  }),
});

describe("useErrorRateStatus", () => {
  let mockFetch: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.useFakeTimers();
    process.env.NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000";
  });

  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it("começa como null antes do primeiro poll resolver", () => {
    mockFetch = vi.fn().mockResolvedValue(OK_RESPONSE("NORMAL"));
    vi.stubGlobal("fetch", mockFetch);

    const { result } = renderHook(() => useErrorRateStatus());
    expect(result.current).toBeNull();
  });

  it("resolve para o status devolvido pela API no primeiro poll", async () => {
    mockFetch = vi.fn().mockResolvedValue(OK_RESPONSE("WARNING"));
    vi.stubGlobal("fetch", mockFetch);

    const { result } = renderHook(() => useErrorRateStatus());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current).toBe("WARNING");
  });

  it("chama GET /observability/error-rate", async () => {
    mockFetch = vi.fn().mockResolvedValue(OK_RESPONSE("NORMAL"));
    vi.stubGlobal("fetch", mockFetch);

    renderHook(() => useErrorRateStatus());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/observability/error-rate"),
      expect.objectContaining({ cache: "no-store" }),
    );
  });

  it("faz poll novamente após o intervalo e reflete uma mudança de status (CRITICAL)", async () => {
    mockFetch = vi
      .fn()
      .mockResolvedValueOnce(OK_RESPONSE("NORMAL"))
      .mockResolvedValueOnce(OK_RESPONSE("CRITICAL"));
    vi.stubGlobal("fetch", mockFetch);

    const { result } = renderHook(() => useErrorRateStatus());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(result.current).toBe("NORMAL");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(15_000);
    });
    expect(result.current).toBe("CRITICAL");
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it("mantém o último status conhecido quando o fetch falha (não trava/derruba)", async () => {
    mockFetch = vi
      .fn()
      .mockResolvedValueOnce(OK_RESPONSE("NORMAL"))
      .mockRejectedValueOnce(new Error("network down"));
    vi.stubGlobal("fetch", mockFetch);

    const { result } = renderHook(() => useErrorRateStatus());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(result.current).toBe("NORMAL");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(15_000);
    });
    expect(result.current).toBe("NORMAL");
  });

  it("para de fazer poll após o unmount", async () => {
    mockFetch = vi.fn().mockResolvedValue(OK_RESPONSE("NORMAL"));
    vi.stubGlobal("fetch", mockFetch);

    const { unmount } = renderHook(() => useErrorRateStatus());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    const callsBeforeUnmount = mockFetch.mock.calls.length;

    unmount();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });

    expect(mockFetch).toHaveBeenCalledTimes(callsBeforeUnmount);
  });
});
