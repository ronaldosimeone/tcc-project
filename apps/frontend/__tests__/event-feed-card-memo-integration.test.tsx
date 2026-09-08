/**
 * Teste de integração — EventFeedCard + React.memo + WebSocket real (RNF-39).
 *
 * Diferente de `event-feed-card.test.tsx` (que substitui `useAlertWebSocket`
 * por um valor estático via `vi.mock`), este teste usa o HOOK REAL com um
 * `WebSocket` mockado. Isso é necessário para provar a afirmação exata do
 * comentário em `EventFeedCard`: o `React.memo` só intercepta re-renders
 * disparados pelo PAI com as mesmas props — ele NÃO impede o componente de
 * re-renderizar quando o seu próprio estado interno (aqui, o `setAlerts`
 * dentro de `useAlertWebSocket`) muda, porque esse é o mecanismo real pelo
 * qual um alerta novo chega em produção (mensagem WS → setState local),
 * nunca via re-render forçado pelo `FleetDashboard` pai.
 *
 * MockWebSocket copiado do padrão já usado em use-alert-websocket.test.ts.
 */

import type React from "react";
import { render, screen, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import EventFeedCard from "@/components/dashboard/EventFeedCard";

vi.mock("@/components/ui/scroll-area", () => ({
  ScrollArea: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="scroll-area">{children}</div>
  ),
  ScrollBar: () => null,
}));

class MockWebSocket {
  static instances: MockWebSocket[] = [];
  url: string;
  readyState = 0;
  onopen: (() => void) | null = null;
  onerror: ((e: Event) => void) | null = null;
  onclose: ((e: CloseEvent) => void) | null = null;
  onmessage: ((e: MessageEvent<string>) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  send(): void {}
  close(): void {
    this.readyState = 3;
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

beforeEach(() => {
  MockWebSocket.reset();
  vi.stubGlobal("WebSocket", MockWebSocket);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("EventFeedCard + React.memo (integração com WebSocket real)", () => {
  it("atualiza a UI quando o WebSocket real entrega um novo alerta, mesmo memoizado", () => {
    render(<EventFeedCard />);
    expect(screen.queryByText(/Anomalia detectada/)).not.toBeInTheDocument();

    act(() => {
      MockWebSocket.latest.simulateMessage({
        type: "alert",
        message_id: "live-integration-1",
        probability: 0.91,
        predicted_class: 1,
        timestamp: new Date().toISOString(),
      });
    });

    // O alerta chegou via setState INTERNO do hook (não via prop do pai) —
    // o React.memo em EventFeedCard não deve (e não pode) bloquear isto.
    expect(
      screen.getByText(/Anomalia detectada \(91\.0%\)/),
    ).toBeInTheDocument();
  });
});
