/**
 * Testes do AlertToastQueue (+ ConnectionBanner) — RF-16/RNF-34 — RNF-59.
 *
 * Componente 100% de apresentação (props puras) — cobre a fila vazia vs.
 * populada, o card individual (crítico vs. degradação), o botão
 * "Reconhecer" e o banner de reconexão do WS nos 3 estados possíveis.
 */
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  AlertToastQueue,
  ConnectionBanner,
} from "@/components/alert-toast-queue";
import type { WsAlert } from "@/hooks/use-alert-websocket";

function makeAlert(overrides: Partial<WsAlert> = {}): WsAlert {
  return {
    message_id: "msg-1",
    probability: 0.91,
    predicted_class: 1,
    timestamp: "2026-01-01T12:00:00.000Z",
    receivedAt: Date.now(),
    ...overrides,
  };
}

describe("AlertToastQueue", () => {
  it("não renderiza nada quando a fila está vazia", () => {
    const { container } = render(
      <AlertToastQueue alerts={[]} status="open" onAcknowledge={vi.fn()} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("renderiza um toast 'Falha Crítica' para probabilidade >= 0.65", () => {
    render(
      <AlertToastQueue
        alerts={[makeAlert({ probability: 0.91 })]}
        status="open"
        onAcknowledge={vi.fn()}
      />,
    );
    expect(screen.getByTestId("alert-toast-queue")).toBeInTheDocument();
    expect(screen.getByText("Falha Crítica")).toBeInTheDocument();
    expect(screen.getByText("91.0%")).toBeInTheDocument();
  });

  it("renderiza um toast 'Degradação Detectada' para probabilidade < 0.65", () => {
    render(
      <AlertToastQueue
        alerts={[makeAlert({ probability: 0.5 })]}
        status="open"
        onAcknowledge={vi.fn()}
      />,
    );
    expect(screen.getByText("Degradação Detectada")).toBeInTheDocument();
    expect(screen.getByText("50.0%")).toBeInTheDocument();
  });

  it("renderiza múltiplos toasts na ordem recebida e chama onAcknowledge com o message_id correto", () => {
    const onAcknowledge = vi.fn();
    render(
      <AlertToastQueue
        alerts={[
          makeAlert({ message_id: "msg-a", probability: 0.9 }),
          makeAlert({ message_id: "msg-b", probability: 0.4 }),
        ]}
        status="open"
        onAcknowledge={onAcknowledge}
      />,
    );

    const toasts = screen.getAllByTestId("alert-toast");
    expect(toasts).toHaveLength(2);
    expect(toasts[0]).toHaveAttribute("data-message-id", "msg-a");
    expect(toasts[1]).toHaveAttribute("data-message-id", "msg-b");

    const buttons = screen.getAllByRole("button", { name: /reconhecer/i });
    fireEvent.click(buttons[1]);
    expect(onAcknowledge).toHaveBeenCalledWith("msg-b");
  });

  it("não renderiza o banner de conexão quando status='open'", () => {
    render(<ConnectionBanner status="open" />);
    expect(
      screen.queryByTestId("ws-connection-banner"),
    ).not.toBeInTheDocument();
  });

  it("mostra 'Reconectando alertas…' para status='reconnecting'/'connecting'", () => {
    const { rerender } = render(<ConnectionBanner status="reconnecting" />);
    expect(screen.getByText("Reconectando alertas…")).toBeInTheDocument();

    rerender(<ConnectionBanner status="connecting" />);
    expect(screen.getByText("Reconectando alertas…")).toBeInTheDocument();
  });

  it("mostra 'Canal de alertas offline' para status='closed'/'error'", () => {
    const { rerender } = render(<ConnectionBanner status="closed" />);
    expect(screen.getByText("Canal de alertas offline")).toBeInTheDocument();

    rerender(<ConnectionBanner status="error" />);
    expect(screen.getByText("Canal de alertas offline")).toBeInTheDocument();
  });
});
