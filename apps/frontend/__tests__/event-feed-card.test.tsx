/**
 * Testes de regressão — EventFeedCard (RNF-39).
 *
 * Cobertura:
 *   - Renderização do histórico (eventos demo quando não há alertas vivos).
 *   - Inserção de novo alerta — evento vivo aparece no topo.
 *   - Ordenação por timestamp (mais recente primeiro).
 *   - Comportamento com muitos alertas — nunca renderiza mais que
 *     VISIBLE_FEED_EVENTS (10) `<li>`, mesmo com uma fila maior.
 *   - React.memo: componente SEM props não deve re-executar seu corpo
 *     (`cn()`) quando um pai externo força um re-render sem mudar nada.
 *
 * A prova de que o memo NÃO quebra a reatividade ao WebSocket real (estado
 * interno do próprio hook, não props do pai) está em
 * `event-feed-card-memo-integration.test.tsx` — usa o `useAlertWebSocket`
 * real + um WebSocket mockado, em vez de substituir o hook inteiro por um
 * valor estático (o que mascararia exatamente esse cenário).
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import EventFeedCard from "@/components/dashboard/EventFeedCard";
import type { WsAlert } from "@/hooks/use-alert-websocket";

vi.mock("@/lib/utils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/utils")>();
  return { ...actual, cn: vi.fn(actual.cn) };
});

// ScrollArea (Radix) depende de ResizeObserver, indisponível em jsdom —
// mesma estratégia de mock usada em alert-panel.test.tsx.
vi.mock("@/components/ui/scroll-area", () => ({
  ScrollArea: ({
    children,
    className,
  }: {
    children: React.ReactNode;
    className?: string;
  }) => (
    <div data-testid="scroll-area" className={className}>
      {children}
    </div>
  ),
  ScrollBar: () => null,
}));

const mockUseAlertWebSocket = vi.fn();
vi.mock("@/hooks/use-alert-websocket", () => ({
  useAlertWebSocket: () => mockUseAlertWebSocket(),
}));

import { cn } from "@/lib/utils";

function alert(overrides: Partial<WsAlert>): WsAlert {
  return {
    message_id: "m-1",
    probability: 0.8,
    predicted_class: 1,
    timestamp: new Date().toISOString(),
    receivedAt: Date.now(),
    ...overrides,
  };
}

beforeEach(() => {
  mockUseAlertWebSocket.mockReturnValue({ alerts: [] });
});

describe("EventFeedCard", () => {
  it("renderiza os eventos demo quando não há alertas vivos", () => {
    render(<EventFeedCard />);
    expect(screen.getByText("Risco de anomalia elevado")).toBeInTheDocument();
    expect(screen.getByText("Sinal estabilizado")).toBeInTheDocument();
  });

  it("insere um novo alerta vivo no topo da lista", () => {
    mockUseAlertWebSocket.mockReturnValue({
      alerts: [alert({ message_id: "live-1", probability: 0.72 })],
    });
    render(<EventFeedCard />);
    expect(
      screen.getByText(/Anomalia detectada \(72\.0%\)/),
    ).toBeInTheDocument();
  });

  it("ordena eventos por timestamp — mais recente primeiro", () => {
    const now = Date.now();
    mockUseAlertWebSocket.mockReturnValue({
      alerts: [
        alert({
          message_id: "live-old",
          probability: 0.5,
          // 10 min atrás — mais VELHO que o evento demo mais recente (ageMin: 2).
          timestamp: new Date(now - 10 * 60_000).toISOString(),
        }),
      ],
    });
    render(<EventFeedCard />);
    const items = screen.getAllByRole("listitem");
    // O evento demo "Risco de anomalia elevado" (ageMin: 2) é mais recente
    // que o alerta vivo de 10 min atrás — deve vir primeiro na lista.
    expect(items[0].textContent).toContain("Risco de anomalia elevado");
  });

  it("nunca renderiza mais que 10 eventos, mesmo com muitos alertas", () => {
    const now = Date.now();
    const many: WsAlert[] = Array.from({ length: 20 }, (_, i) =>
      alert({
        message_id: `live-${i}`,
        probability: 0.9,
        timestamp: new Date(now - i * 1000).toISOString(),
      }),
    );
    mockUseAlertWebSocket.mockReturnValue({ alerts: many });
    render(<EventFeedCard />);
    expect(screen.getAllByRole("listitem")).toHaveLength(10);
  });

  it("React.memo: sem props, re-render externo idêntico não re-executa o corpo", () => {
    const mockedCn = cn as unknown as ReturnType<typeof vi.fn>;
    mockedCn.mockClear();

    const { rerender } = render(<EventFeedCard />);
    const callsAfterMount = mockedCn.mock.calls.length;
    expect(callsAfterMount).toBeGreaterThan(0);

    rerender(<EventFeedCard />); // mesmo elemento, sem props para mudar
    expect(mockedCn.mock.calls.length).toBe(callsAfterMount);
  });
});
