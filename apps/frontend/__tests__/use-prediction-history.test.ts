/**
 * Testes unitários para usePredictionHistory — RF-08 / RNF-14.
 *
 * Cobertura:
 * - Estado inicial (isMounted=false, history=[])
 * - Hidratação: leitura do localStorage após montagem
 * - Adição de entradas ao histórico
 * - Persistência no localStorage
 * - Limite de PREDICTION_HISTORY_MAX entradas
 * - clearHistory
 * - RF-08: riskLevel correto por threshold
 * - QA: valores inválidos de failure_probability (NaN, Infinity, null, neg., > 1)
 * - QA: localStorage corrompido (JSON inválido, não-array)
 */

import { renderHook, act } from "@testing-library/react";
import { describe, expect, it, beforeEach, afterEach, vi } from "vitest";
import type { PredictResponse } from "@/lib/api-client";
import {
  usePredictionHistory,
  sanitizeProbability,
  PREDICTION_HISTORY_STORAGE_KEY,
  PREDICTION_HISTORY_MAX,
} from "@/hooks/use-prediction-history";

// ── Fixtures ──────────────────────────────────────────────────────────────

function makeResponse(
  overrides: Partial<PredictResponse> = {},
): PredictResponse {
  return {
    predicted_class: 0,
    failure_probability: 0.1,
    timestamp: new Date().toISOString(),
    ...overrides,
  };
}

// ── Lifecycle Global ──────────────────────────────────────────────────────

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  localStorage.clear();
});

// ── sanitizeProbability (função pura) ─────────────────────────────────────

describe("sanitizeProbability", () => {
  it("retorna o valor para números finitos no intervalo [0,1]", () => {
    expect(sanitizeProbability(0)).toBe(0);
    expect(sanitizeProbability(0.5)).toBe(0.5);
    expect(sanitizeProbability(1)).toBe(1);
  });

  it("fixa negativos em 0", () => {
    expect(sanitizeProbability(-0.5)).toBe(0);
    expect(sanitizeProbability(-100)).toBe(0);
  });

  it("fixa valores > 1 em 1", () => {
    expect(sanitizeProbability(1.5)).toBe(1);
    expect(sanitizeProbability(99)).toBe(1);
  });

  it("retorna null para NaN", () => {
    expect(sanitizeProbability(NaN)).toBeNull();
  });

  it("retorna null para Infinity", () => {
    expect(sanitizeProbability(Infinity)).toBeNull();
    expect(sanitizeProbability(-Infinity)).toBeNull();
  });

  it("retorna null para null (runtime)", () => {
    expect(sanitizeProbability(null)).toBeNull();
  });

  it("retorna null para undefined (runtime)", () => {
    expect(sanitizeProbability(undefined)).toBeNull();
  });

  it("retorna null para string numérica (runtime)", () => {
    expect(sanitizeProbability("0.5")).toBeNull();
  });
});

// ── usePredictionHistory ──────────────────────────────────────────────────

describe("usePredictionHistory", () => {
  // 1. Liga os fake timers e limpa o cache ANTES de cada teste
  beforeEach(() => {
    vi.useFakeTimers();
    localStorage.clear();
  });

  // 2. Desliga os fake timers APÓS cada teste
  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
  });

  // 3. Helper universal para avançar o tempo do setTimeout do nosso Hook
  async function flushEffects() {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
  }

  // ── Estado inicial ──────────────────────────────────────────────────────

  it("history começa vazio quando localStorage está vazio", async () => {
    const { result } = renderHook(() => usePredictionHistory(null));
    await flushEffects();
    expect(result.current.history).toHaveLength(0);
  });

  // ── Hidratação ──────────────────────────────────────────────────────────

  it("isMounted passa para true após a montagem", async () => {
    const { result } = renderHook(() => usePredictionHistory(null));
    await flushEffects();
    expect(result.current.isMounted).toBe(true);
  });

  it("carrega histórico do localStorage na montagem", async () => {
    const stored = [
      {
        id: "ts1-abc",
        timestamp: "2024-01-01T10:00:00.000Z",
        failure_probability: 0.4,
        predicted_class: 1,
        riskLevel: "ALERTA",
      },
    ];
    localStorage.setItem(
      PREDICTION_HISTORY_STORAGE_KEY,
      JSON.stringify(stored),
    );

    const { result } = renderHook(() => usePredictionHistory(null));
    await flushEffects();

    expect(result.current.history).toHaveLength(1);
    expect(result.current.history[0]?.riskLevel).toBe("ALERTA");
  });

  it("ignora localStorage corrompido (JSON inválido) e inicia vazio", async () => {
    localStorage.setItem(PREDICTION_HISTORY_STORAGE_KEY, "{ broken json }}}");

    const { result } = renderHook(() => usePredictionHistory(null));
    await flushEffects();

    expect(result.current.history).toHaveLength(0);
    expect(result.current.isMounted).toBe(true);
  });

  it("ignora localStorage com dado não-array e inicia vazio", async () => {
    localStorage.setItem(
      PREDICTION_HISTORY_STORAGE_KEY,
      JSON.stringify({ not: "an array" }),
    );

    const { result } = renderHook(() => usePredictionHistory(null));
    await flushEffects();

    expect(result.current.history).toHaveLength(0);
  });

  it("ignora entradas inválidas dentro do array do localStorage", async () => {
    const mixed = [
      {
        id: "ok1",
        timestamp: "2024-01-01T10:00:00Z",
        failure_probability: 0.1,
        predicted_class: 0,
        riskLevel: "NORMAL",
      },
      { id: 123, timestamp: "bad" }, // inválida — id não é string
    ];
    localStorage.setItem(PREDICTION_HISTORY_STORAGE_KEY, JSON.stringify(mixed));

    const { result } = renderHook(() => usePredictionHistory(null));
    await flushEffects();

    expect(result.current.history).toHaveLength(1);
  });

  // ── Adição de entradas ──────────────────────────────────────────────────

  it("adiciona entrada ao histórico quando latest é fornecido", async () => {
    const latest = makeResponse({ failure_probability: 0.1 });
    const { result } = renderHook(() => usePredictionHistory(latest));
    await flushEffects();

    expect(result.current.history).toHaveLength(1);
  });

  it("nova entrada fica na posição 0 (mais recente primeiro)", async () => {
    const ts1 = "2024-01-01T10:00:00.000Z";
    const ts2 = "2024-01-01T10:00:05.000Z";

    const { result, rerender } = renderHook(
      ({ latest }: { latest: PredictResponse | null }) =>
        usePredictionHistory(latest),
      { initialProps: { latest: makeResponse({ timestamp: ts1 }) } },
    );
    await flushEffects();

    rerender({ latest: makeResponse({ timestamp: ts2 }) });
    await flushEffects();

    expect(result.current.history[0]?.timestamp).toBe(ts2);
    expect(result.current.history[1]?.timestamp).toBe(ts1);
  });

  it("não adiciona entrada duplicada com o mesmo timestamp", async () => {
    const latest = makeResponse({ timestamp: "2024-01-01T10:00:00.000Z" });

    const { result, rerender } = renderHook(
      ({ l }: { l: PredictResponse }) => usePredictionHistory(l),
      { initialProps: { l: latest } },
    );
    await flushEffects();

    // Re-renderiza com o mesmo objeto (mesmo timestamp)
    rerender({ l: { ...latest } });
    await flushEffects();

    expect(result.current.history).toHaveLength(1);
  });

  it("não adiciona entrada quando latest é null", async () => {
    const { result } = renderHook(() => usePredictionHistory(null));
    await flushEffects();

    expect(result.current.history).toHaveLength(0);
  });

  // ── Limite de histórico ─────────────────────────────────────────────────

  it(`mantém no máximo ${PREDICTION_HISTORY_MAX} entradas`, async () => {
    // Pré-carrega localStorage com 50 entradas
    const full = Array.from({ length: PREDICTION_HISTORY_MAX }, (_, i) => ({
      id: `id-${i}`,
      timestamp: `2024-01-01T00:00:${String(i).padStart(2, "0")}.000Z`,
      failure_probability: 0.1,
      predicted_class: 0,
      riskLevel: "NORMAL",
    }));
    localStorage.setItem(PREDICTION_HISTORY_STORAGE_KEY, JSON.stringify(full));

    const newLatest = makeResponse({ timestamp: "2024-01-01T01:00:00.000Z" });
    const { result } = renderHook(() => usePredictionHistory(newLatest));
    await flushEffects();

    expect(result.current.history).toHaveLength(PREDICTION_HISTORY_MAX);
  });

  // ── Persistência no localStorage ────────────────────────────────────────

  it("persiste a nova entrada no localStorage", async () => {
    const latest = makeResponse({
      failure_probability: 0.5,
      timestamp: "2024-01-01T10:00:00.000Z",
    });

    renderHook(() => usePredictionHistory(latest));
    await flushEffects();

    const stored = localStorage.getItem(PREDICTION_HISTORY_STORAGE_KEY);
    expect(stored).not.toBeNull();
    const parsed: unknown = JSON.parse(stored!);
    expect(Array.isArray(parsed)).toBe(true);
    expect((parsed as unknown[]).length).toBeGreaterThan(0);
  });

  // ── clearHistory ────────────────────────────────────────────────────────

  it("clearHistory esvazia o histórico em memória", async () => {
    const latest = makeResponse({ failure_probability: 0.5 });
    const { result } = renderHook(() => usePredictionHistory(latest));
    await flushEffects();

    expect(result.current.history.length).toBeGreaterThan(0);

    act(() => {
      result.current.clearHistory();
    });

    expect(result.current.history).toHaveLength(0);
  });

  it("clearHistory remove a chave do localStorage", async () => {
    const latest = makeResponse({ failure_probability: 0.5 });
    const { result } = renderHook(() => usePredictionHistory(latest));
    await flushEffects();

    act(() => {
      result.current.clearHistory();
    });

    expect(localStorage.getItem(PREDICTION_HISTORY_STORAGE_KEY)).toBeNull();
  });

  // ── RF-08: riskLevel por threshold ──────────────────────────────────────

  it("RF-08: entrada com prob >= 0.65 tem riskLevel CRÍTICO", async () => {
    const latest = makeResponse({
      failure_probability: 0.8,
      timestamp: "2024-01-01T10:00:00.000Z",
    });
    const { result } = renderHook(() => usePredictionHistory(latest));
    await flushEffects();

    expect(result.current.history[0]?.riskLevel).toBe("CRÍTICO");
  });

  it("RF-08: entrada com 0.3 <= prob < 0.65 tem riskLevel ALERTA", async () => {
    const latest = makeResponse({
      failure_probability: 0.45,
      timestamp: "2024-01-01T10:00:00.000Z",
    });
    const { result } = renderHook(() => usePredictionHistory(latest));
    await flushEffects();

    expect(result.current.history[0]?.riskLevel).toBe("ALERTA");
  });

  it("RF-08: entrada com prob < 0.3 tem riskLevel NORMAL", async () => {
    const latest = makeResponse({
      failure_probability: 0.1,
      timestamp: "2024-01-01T10:00:00.000Z",
    });
    const { result } = renderHook(() => usePredictionHistory(latest));
    await flushEffects();

    expect(result.current.history[0]?.riskLevel).toBe("NORMAL");
  });

  // ── QA: valores inválidos de failure_probability ────────────────────────

  it("QA: descarta entrada com failure_probability NaN", async () => {
    const latest = {
      predicted_class: 0,
      failure_probability: NaN,
      timestamp: "2024-01-01T10:00:00.000Z",
    } as PredictResponse;

    const { result } = renderHook(() => usePredictionHistory(latest));
    await flushEffects();

    expect(result.current.history).toHaveLength(0);
  });

  it("QA: descarta entrada com failure_probability Infinity", async () => {
    const latest = {
      predicted_class: 0,
      failure_probability: Infinity,
      timestamp: "2024-01-01T10:00:00.000Z",
    } as PredictResponse;

    const { result } = renderHook(() => usePredictionHistory(latest));
    await flushEffects();

    expect(result.current.history).toHaveLength(0);
  });

  it("QA: descarta entrada com failure_probability null (runtime)", async () => {
    const latest = {
      predicted_class: 0,
      failure_probability: null as unknown as number,
      timestamp: "2024-01-01T10:00:00.000Z",
    } as PredictResponse;

    const { result } = renderHook(() => usePredictionHistory(latest));
    await flushEffects();

    expect(result.current.history).toHaveLength(0);
  });

  it("QA: fixa failure_probability negativo em 0 e classifica como NORMAL", async () => {
    const latest = {
      predicted_class: 0,
      failure_probability: -0.5,
      timestamp: "2024-01-01T10:00:00.000Z",
    } as PredictResponse;

    const { result } = renderHook(() => usePredictionHistory(latest));
    await flushEffects();

    expect(result.current.history).toHaveLength(1);
    expect(result.current.history[0]?.failure_probability).toBe(0);
    expect(result.current.history[0]?.riskLevel).toBe("NORMAL");
  });

  it("QA: fixa failure_probability > 1 em 1 e classifica como CRÍTICO", async () => {
    const latest = {
      predicted_class: 1,
      failure_probability: 1.5,
      timestamp: "2024-01-01T10:00:00.000Z",
    } as PredictResponse;

    const { result } = renderHook(() => usePredictionHistory(latest));
    await flushEffects();

    expect(result.current.history).toHaveLength(1);
    expect(result.current.history[0]?.failure_probability).toBe(1);
    expect(result.current.history[0]?.riskLevel).toBe("CRÍTICO");
  });
});
