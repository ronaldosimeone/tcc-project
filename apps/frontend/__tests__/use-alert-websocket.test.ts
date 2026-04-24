/**
 * Testes unitários para useAlertWebSocket — RF-14 / RF-16 / RNF-30 / RNF-34.
 *
 * MockWebSocket simula a API do browser WebSocket para testar:
 *  - criação de conexão e status
 *  - protocolo ping/pong (RNF-30)
 *  - ACK automático ao receber alertas
 *  - adição de alertas à fila
 *  - RNF-34: limite de 5 alertas (FIFO — oldest-out)
 *  - RF-16: acknowledge() remove alerta específico
 *  - clearAlerts() esvazia a fila
 *  - reconexão com exponential backoff após fechamento
 *  - cleanup ao desmontar (sem reconexão)
 *
 * Math.random é spionado para retornar 0, tornando o jitter determinístico:
 *   delay = backoff * (0.8 + 0 * 0.4) = backoff * 0.8
 */

import { renderHook, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useAlertWebSocket } from "@/hooks/use-alert-websocket";
import type { WsStatus } from "@/hooks/use-alert-websocket";

// ── MockWebSocket ─────────────────────────────────────────────────────────

class MockWebSocket {
  static instances: MockWebSocket[] = [];

  url: string;
  readyState = 0; // CONNECTING
  sent: string[] = [];

  onopen: (() => void) | null = null;
  onerror: ((e: Event) => void) | null = null;
  onclose: ((e: CloseEvent) => void) | null = null;
  onmessage: ((e: MessageEvent<string>) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  send(data: string): void {
    this.sent.push(data);
  }

  close(): void {
    this.readyState = 3; // CLOSED
    this.onclose?.(new CloseEvent("close", { wasClean: true, code: 1000 }));
  }

  // ── Helpers de simulação ──────────────────────────────────────────────────

  simulateOpen(): void {
    this.readyState = 1; // OPEN
    this.onopen?.();
  }

  simulateError(): void {
    this.onerror?.(new Event("error"));
  }

  simulateClose(): void {
    this.readyState = 3; // CLOSED
    this.onclose?.(new CloseEvent("close", { wasClean: false, code: 1006 }));
  }

  simulateMessage(data: object): void {
    this.onmessage?.(
      new MessageEvent("message", { data: JSON.stringify(data) }),
    );
  }

  static reset(): void {
    MockWebSocket.instances = [];
  }

  static get latest(): MockWebSocket {
    return MockWebSocket.instances[MockWebSocket.instances.length - 1]!;
  }
}

// ── Suite ─────────────────────────────────────────────────────────────────

describe("useAlertWebSocket", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.spyOn(Math, "random").mockReturnValue(0); // jitter determinístico: delay = backoff * 0.8
    MockWebSocket.reset();
    vi.stubGlobal("WebSocket", MockWebSocket);
  });

  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  // ── Conexão ──────────────────────────────────────────────────────────────

  it("cria WebSocket ao montar", () => {
    renderHook(() => useAlertWebSocket());
    expect(MockWebSocket.instances).toHaveLength(1);
  });

  it("URL do WebSocket termina com /ws/alerts", () => {
    renderHook(() => useAlertWebSocket());
    expect(MockWebSocket.instances[0]!.url).toMatch(/\/ws\/alerts$/);
  });

  it("inicia com status 'connecting'", () => {
    const { result } = renderHook(() => useAlertWebSocket());
    expect(result.current.status).toBe<WsStatus>("connecting");
  });

  it("inicia com fila de alertas vazia", () => {
    const { result } = renderHook(() => useAlertWebSocket());
    expect(result.current.alerts).toHaveLength(0);
  });

  it("transiciona para 'open' quando WebSocket abre", async () => {
    const { result } = renderHook(() => useAlertWebSocket());

    await act(async () => {
      MockWebSocket.instances[0]!.simulateOpen();
    });

    expect(result.current.status).toBe<WsStatus>("open");
  });

  it("transiciona para 'error' quando WebSocket emite erro", async () => {
    const { result } = renderHook(() => useAlertWebSocket());

    await act(async () => {
      MockWebSocket.instances[0]!.simulateError();
    });

    expect(result.current.status).toBe<WsStatus>("error");
  });

  // ── Protocolo ping/pong (RNF-30) ─────────────────────────────────────────

  it("RNF-30: responde a ping com pong", async () => {
    renderHook(() => useAlertWebSocket());
    const ws = MockWebSocket.instances[0]!;

    await act(async () => {
      ws.simulateMessage({ type: "ping" });
    });

    expect(ws.sent).toContain(JSON.stringify({ type: "pong" }));
  });

  it("RNF-30: ping não adiciona nada à fila de alertas", async () => {
    const { result } = renderHook(() => useAlertWebSocket());

    await act(async () => {
      MockWebSocket.instances[0]!.simulateMessage({ type: "ping" });
    });

    expect(result.current.alerts).toHaveLength(0);
  });

  // ── Recebimento de alertas ────────────────────────────────────────────────

  it("envia ACK ao receber frame de alerta", async () => {
    renderHook(() => useAlertWebSocket());
    const ws = MockWebSocket.instances[0]!;

    await act(async () => {
      ws.simulateMessage({
        type: "alert",
        message_id: "msg-abc",
        probability: 0.85,
        predicted_class: 1,
        timestamp: "2024-01-01T10:00:00.000Z",
      });
    });

    expect(ws.sent).toContain(
      JSON.stringify({ type: "ack", message_id: "msg-abc" }),
    );
  });

  it("adiciona alerta à fila ao receber frame do tipo 'alert'", async () => {
    const { result } = renderHook(() => useAlertWebSocket());

    await act(async () => {
      MockWebSocket.instances[0]!.simulateMessage({
        type: "alert",
        message_id: "alert-1",
        probability: 0.9,
        predicted_class: 1,
        timestamp: "2024-01-01T10:00:00.000Z",
      });
    });

    expect(result.current.alerts).toHaveLength(1);
    expect(result.current.alerts[0]!.message_id).toBe("alert-1");
    expect(result.current.alerts[0]!.probability).toBe(0.9);
    expect(result.current.alerts[0]!.predicted_class).toBe(1);
  });

  it("alerta mais recente fica no índice 0 da fila", async () => {
    const { result } = renderHook(() => useAlertWebSocket());
    const ws = MockWebSocket.instances[0]!;

    await act(async () => {
      ws.simulateMessage({
        type: "alert",
        message_id: "first",
        probability: 0.8,
        predicted_class: 1,
        timestamp: "2024-01-01T10:00:00.000Z",
      });
      ws.simulateMessage({
        type: "alert",
        message_id: "second",
        probability: 0.85,
        predicted_class: 1,
        timestamp: "2024-01-01T10:00:01.000Z",
      });
    });

    expect(result.current.alerts[0]!.message_id).toBe("second");
    expect(result.current.alerts[1]!.message_id).toBe("first");
  });

  it("ignora frames com JSON inválido sem lançar exceção", async () => {
    const { result } = renderHook(() => useAlertWebSocket());

    await act(async () => {
      MockWebSocket.instances[0]!.onmessage?.(
        new MessageEvent("message", { data: "not-valid-json" }),
      );
    });

    expect(result.current.alerts).toHaveLength(0);
  });

  // ── RNF-34: Limite da fila ────────────────────────────────────────────────

  it("RNF-34: mantém no máximo 5 alertas simultâneos na fila", async () => {
    const { result } = renderHook(() => useAlertWebSocket());
    const ws = MockWebSocket.instances[0]!;

    await act(async () => {
      for (let i = 1; i <= 6; i++) {
        ws.simulateMessage({
          type: "alert",
          message_id: `alert-${i}`,
          probability: 0.8,
          predicted_class: 1,
          timestamp: new Date().toISOString(),
        });
      }
    });

    expect(result.current.alerts).toHaveLength(5);
  });

  it("RNF-34 FIFO: ao receber o 6º alerta, o 1º (mais antigo) é descartado", async () => {
    const { result } = renderHook(() => useAlertWebSocket());
    const ws = MockWebSocket.instances[0]!;

    await act(async () => {
      for (let i = 1; i <= 6; i++) {
        ws.simulateMessage({
          type: "alert",
          message_id: `alert-${i}`,
          probability: 0.8,
          predicted_class: 1,
          timestamp: new Date().toISOString(),
        });
      }
    });

    const ids = result.current.alerts.map((a) => a.message_id);
    expect(ids).not.toContain("alert-1");
    expect(ids[0]).toBe("alert-6"); // mais recente no topo
    expect(ids[4]).toBe("alert-2"); // 2º mais antigo (o 1º foi evicted)
  });

  it("RNF-34: ao receber exatamente 5 alertas, todos permanecem", async () => {
    const { result } = renderHook(() => useAlertWebSocket());
    const ws = MockWebSocket.instances[0]!;

    await act(async () => {
      for (let i = 1; i <= 5; i++) {
        ws.simulateMessage({
          type: "alert",
          message_id: `alert-${i}`,
          probability: 0.8,
          predicted_class: 1,
          timestamp: new Date().toISOString(),
        });
      }
    });

    expect(result.current.alerts).toHaveLength(5);
    const ids = result.current.alerts.map((a) => a.message_id);
    expect(ids).toContain("alert-1");
    expect(ids).toContain("alert-5");
  });

  // ── RF-16: acknowledge ────────────────────────────────────────────────────

  it("RF-16: acknowledge remove o alerta com o id especificado", async () => {
    const { result } = renderHook(() => useAlertWebSocket());
    const ws = MockWebSocket.instances[0]!;

    await act(async () => {
      ws.simulateMessage({
        type: "alert",
        message_id: "to-remove",
        probability: 0.85,
        predicted_class: 1,
        timestamp: new Date().toISOString(),
      });
    });

    expect(result.current.alerts).toHaveLength(1);

    await act(async () => {
      result.current.acknowledge("to-remove");
    });

    expect(result.current.alerts).toHaveLength(0);
  });

  it("RF-16: acknowledge não afeta outros alertas presentes", async () => {
    const { result } = renderHook(() => useAlertWebSocket());
    const ws = MockWebSocket.instances[0]!;

    await act(async () => {
      ws.simulateMessage({
        type: "alert",
        message_id: "alert-a",
        probability: 0.85,
        predicted_class: 1,
        timestamp: new Date().toISOString(),
      });
      ws.simulateMessage({
        type: "alert",
        message_id: "alert-b",
        probability: 0.9,
        predicted_class: 1,
        timestamp: new Date().toISOString(),
      });
    });

    await act(async () => {
      result.current.acknowledge("alert-a");
    });

    expect(result.current.alerts).toHaveLength(1);
    expect(result.current.alerts[0]!.message_id).toBe("alert-b");
  });

  it("RF-16: acknowledge com id inexistente não altera a fila", async () => {
    const { result } = renderHook(() => useAlertWebSocket());
    const ws = MockWebSocket.instances[0]!;

    await act(async () => {
      ws.simulateMessage({
        type: "alert",
        message_id: "present",
        probability: 0.8,
        predicted_class: 1,
        timestamp: new Date().toISOString(),
      });
    });

    await act(async () => {
      result.current.acknowledge("nonexistent");
    });

    expect(result.current.alerts).toHaveLength(1);
  });

  // ── clearAlerts ──────────────────────────────────────────────────────────

  it("clearAlerts esvazia toda a fila de alertas", async () => {
    const { result } = renderHook(() => useAlertWebSocket());
    const ws = MockWebSocket.instances[0]!;

    await act(async () => {
      ws.simulateMessage({
        type: "alert",
        message_id: "c1",
        probability: 0.85,
        predicted_class: 1,
        timestamp: new Date().toISOString(),
      });
      ws.simulateMessage({
        type: "alert",
        message_id: "c2",
        probability: 0.9,
        predicted_class: 1,
        timestamp: new Date().toISOString(),
      });
    });

    await act(async () => {
      result.current.clearAlerts();
    });

    expect(result.current.alerts).toHaveLength(0);
  });

  // ── Reconexão com backoff ─────────────────────────────────────────────────

  it("transiciona para 'reconnecting' quando WebSocket fecha inesperadamente", async () => {
    const { result } = renderHook(() => useAlertWebSocket());
    const ws = MockWebSocket.instances[0]!;
    ws.simulateOpen();

    await act(async () => {
      ws.simulateClose();
    });

    expect(result.current.status).toBe<WsStatus>("reconnecting");
  });

  it("cria nova conexão após delay de backoff do attempt 1 (800ms)", async () => {
    renderHook(() => useAlertWebSocket());
    const ws = MockWebSocket.instances[0]!;

    await act(async () => {
      ws.simulateClose();
    });

    expect(MockWebSocket.instances).toHaveLength(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(800);
    });

    expect(MockWebSocket.instances).toHaveLength(2);
  });

  it("dobra o backoff a cada tentativa (exponential backoff)", async () => {
    renderHook(() => useAlertWebSocket());

    // Attempt 1: backoff = 1000ms * 0.8 = 800ms
    await act(async () => {
      MockWebSocket.instances[0]!.simulateClose();
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(800);
    });
    expect(MockWebSocket.instances).toHaveLength(2);

    // Attempt 2: backoff = 2000ms * 0.8 = 1600ms
    await act(async () => {
      MockWebSocket.instances[1]!.simulateClose();
    });

    // Após 800ms ainda NÃO reconectou
    await act(async () => {
      await vi.advanceTimersByTimeAsync(800);
    });
    expect(MockWebSocket.instances).toHaveLength(2);

    // Após mais 800ms (total 1600ms) reconecta
    await act(async () => {
      await vi.advanceTimersByTimeAsync(800);
    });
    expect(MockWebSocket.instances).toHaveLength(3);
  });

  it("reseta attempt para 0 quando a reconexão é bem-sucedida (onopen)", async () => {
    const { result } = renderHook(() => useAlertWebSocket());

    await act(async () => {
      MockWebSocket.instances[0]!.simulateClose();
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(800);
    });

    // 2ª conexão — fica "connecting" / "reconnecting"
    expect(result.current.status).toBe<WsStatus>("reconnecting");

    await act(async () => {
      MockWebSocket.instances[1]!.simulateOpen();
    });

    expect(result.current.status).toBe<WsStatus>("open");
  });

  // ── Cleanup ───────────────────────────────────────────────────────────────

  it("fecha o WebSocket ao desmontar o hook", () => {
    const { unmount } = renderHook(() => useAlertWebSocket());
    const ws = MockWebSocket.instances[0]!;
    unmount();
    expect(ws.readyState).toBe(3); // CLOSED
  });

  it("não cria nova conexão se desmontado durante o backoff", async () => {
    const { unmount } = renderHook(() => useAlertWebSocket());

    await act(async () => {
      MockWebSocket.instances[0]!.simulateClose();
    });

    unmount(); // cancela o timer de reconexão

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });

    expect(MockWebSocket.instances).toHaveLength(1);
  });
});
