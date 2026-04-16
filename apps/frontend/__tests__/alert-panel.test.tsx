/**
 * Testes de renderização para AlertPanel — RF-08 / RNF-14.
 *
 * Estratégia de isolamento:
 *   usePredictionHistory é mockado para controlar o estado do histórico
 *   sem depender de localStorage ou efeitos assíncronos.
 *
 *   ScrollArea é mockado com uma <div> simples para evitar dependências
 *   de ResizeObserver do Radix UI em jsdom.
 *
 * Cobertura:
 *   - Renderização nos 3 estados visuais (Normal, Alerta, Crítico)
 *   - RF-08: banner crítico (presença, ausência, role="alert", texto)
 *   - RF-08: classes CSS corretas por threshold
 *   - data-risk atributos para seletores de teste
 *   - Estado vazio (sem eventos)
 *   - Lista de eventos com dados corretos
 *   - Botão "Limpar" — visibilidade e callback
 *   - QA: latest=null não causa crash
 *   - QA: probability clamped (negativo, > 1)
 */

import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AlertPanel } from "@/components/alert-panel";
import type { PredictResponse } from "@/lib/api-client";
import type { PredictionHistoryEntry } from "@/hooks/use-prediction-history";

// ── Mocks ─────────────────────────────────────────────────────────────────

// ScrollArea — substitui o componente Radix UI por uma div simples
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

// usePredictionHistory — controle total do estado do histórico
vi.mock("@/hooks/use-prediction-history", async (importOriginal) => {
  const actual = await importOriginal<
    typeof import("@/hooks/use-prediction-history")
  >();
  return {
    ...actual,
    usePredictionHistory: vi.fn(),
  };
});

// ── Polyfill — segurança para qualquer componente Radix restante ───────────

class MockResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = MockResizeObserver;

// ── Helpers ───────────────────────────────────────────────────────────────

async function getMockHook() {
  const mod = await import("@/hooks/use-prediction-history");
  return vi.mocked(mod.usePredictionHistory);
}

function makeHistoryEntry(
  overrides: Partial<PredictionHistoryEntry> = {},
): PredictionHistoryEntry {
  return {
    id: "ts-2024-01-01T10:00:00Z-abc12",
    timestamp: "2024-01-01T10:00:00.000Z",
    failure_probability: 0.1,
    predicted_class: 0,
    riskLevel: "NORMAL",
    ...overrides,
  };
}

function makePredictResponse(
  overrides: Partial<PredictResponse> = {},
): PredictResponse {
  return {
    predicted_class: 0,
    failure_probability: 0.1,
    timestamp: "2024-01-01T10:00:00.000Z",
    ...overrides,
  };
}

// ── Suite ──────────────────────────────────────────────────────────────────

describe("AlertPanel", () => {
  const mockClear = vi.fn();

  // Configura estado padrão do hook mockado antes de cada teste
  beforeEach(async () => {
    const mockHook = await getMockHook();
    mockHook.mockReturnValue({
      history: [],
      isMounted: true,
      clearHistory: mockClear,
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  // ── Renderização básica ─────────────────────────────────────────────────

  it("renderiza sem lançar erros com latest=null e NORMAL", () => {
    expect(() =>
      render(<AlertPanel latest={null} riskLevel="NORMAL" />),
    ).not.toThrow();
  });

  it("exibe o título 'Auditoria'", () => {
    render(<AlertPanel latest={null} riskLevel="NORMAL" />);
    expect(screen.getByText("Auditoria")).toBeInTheDocument();
  });

  it("exibe a seção 'Status Atual'", () => {
    render(<AlertPanel latest={null} riskLevel="NORMAL" />);
    expect(screen.getByText(/status atual/i)).toBeInTheDocument();
  });

  // ── RF-08: Banner crítico ───────────────────────────────────────────────

  it("RF-08: exibe banner crítico quando riskLevel=CRÍTICO", () => {
    render(
      <AlertPanel
        latest={makePredictResponse({ failure_probability: 0.8 })}
        riskLevel="CRÍTICO"
      />,
    );
    expect(screen.getByTestId("critical-banner")).toBeInTheDocument();
  });

  it("RF-08: banner tem role='alert'", () => {
    render(
      <AlertPanel
        latest={makePredictResponse({ failure_probability: 0.8 })}
        riskLevel="CRÍTICO"
      />,
    );
    expect(screen.getByRole("alert")).toBeInTheDocument();
  });

  it("RF-08: banner contém o texto 'FALHA CRÍTICA DETECTADA'", () => {
    render(
      <AlertPanel
        latest={makePredictResponse({ failure_probability: 0.8 })}
        riskLevel="CRÍTICO"
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent(
      /falha crítica detectada/i,
    );
  });

  it("RF-08: NÃO exibe banner quando riskLevel=ALERTA", () => {
    render(
      <AlertPanel
        latest={makePredictResponse({ failure_probability: 0.5 })}
        riskLevel="ALERTA"
      />,
    );
    expect(screen.queryByTestId("critical-banner")).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("RF-08: NÃO exibe banner quando riskLevel=NORMAL", () => {
    render(<AlertPanel latest={null} riskLevel="NORMAL" />);
    expect(screen.queryByTestId("critical-banner")).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  // ── Estados visuais — data-risk e classes CSS ──────────────────────────

  it("panel tem data-risk='NORMAL' no estado normal", () => {
    render(<AlertPanel latest={null} riskLevel="NORMAL" />);
    expect(screen.getByTestId("alert-panel")).toHaveAttribute(
      "data-risk",
      "NORMAL",
    );
  });

  it("panel tem data-risk='ALERTA' no estado de alerta", () => {
    render(
      <AlertPanel
        latest={makePredictResponse({ failure_probability: 0.5 })}
        riskLevel="ALERTA"
      />,
    );
    expect(screen.getByTestId("alert-panel")).toHaveAttribute(
      "data-risk",
      "ALERTA",
    );
  });

  it("panel tem data-risk='CRÍTICO' no estado crítico", () => {
    render(
      <AlertPanel
        latest={makePredictResponse({ failure_probability: 0.8 })}
        riskLevel="CRÍTICO"
      />,
    );
    expect(screen.getByTestId("alert-panel")).toHaveAttribute(
      "data-risk",
      "CRÍTICO",
    );
  });

  it("CRÍTICO aplica classe de borda vermelha", () => {
    render(
      <AlertPanel
        latest={makePredictResponse({ failure_probability: 0.8 })}
        riskLevel="CRÍTICO"
      />,
    );
    expect(screen.getByTestId("alert-panel").className).toContain(
      "border-red-500/30",
    );
  });

  it("ALERTA aplica classe de borda âmbar", () => {
    render(
      <AlertPanel
        latest={makePredictResponse({ failure_probability: 0.5 })}
        riskLevel="ALERTA"
      />,
    );
    expect(screen.getByTestId("alert-panel").className).toContain(
      "border-amber-500/30",
    );
  });

  it("NORMAL aplica classe de borda padrão (border-border)", () => {
    render(<AlertPanel latest={null} riskLevel="NORMAL" />);
    expect(screen.getByTestId("alert-panel").className).toContain(
      "border-border",
    );
  });

  // ── Probabilidade atual ────────────────────────────────────────────────

  it("exibe a probabilidade atual como percentual", () => {
    render(
      <AlertPanel
        latest={makePredictResponse({ failure_probability: 0.427 })}
        riskLevel="ALERTA"
      />,
    );
    // 0.427 * 100 = 42.7%
    const { container } = render(
      <AlertPanel
        latest={makePredictResponse({ failure_probability: 0.427 })}
        riskLevel="ALERTA"
      />,
    );
    expect(container.textContent).toContain("42.7%");
  });

  it("QA: exibe 0.0% quando latest é null", () => {
    const { container } = render(
      <AlertPanel latest={null} riskLevel="NORMAL" />,
    );
    expect(container.textContent).toContain("0.0%");
  });

  it("QA: clamp — exibe 0.0% quando failure_probability negativo", () => {
    const { container } = render(
      <AlertPanel
        latest={makePredictResponse({ failure_probability: -0.5 })}
        riskLevel="NORMAL"
      />,
    );
    expect(container.textContent).toContain("0.0%");
  });

  it("QA: clamp — exibe 100.0% quando failure_probability > 1", () => {
    const { container } = render(
      <AlertPanel
        latest={makePredictResponse({ failure_probability: 1.5 })}
        riskLevel="CRÍTICO"
      />,
    );
    expect(container.textContent).toContain("100.0%");
  });

  // ── Estado vazio ────────────────────────────────────────────────────────

  it("exibe 'Sem eventos registrados' quando histórico está vazio", () => {
    render(<AlertPanel latest={null} riskLevel="NORMAL" />);
    expect(screen.getByTestId("empty-state")).toBeInTheDocument();
    expect(screen.getByText(/sem eventos registrados/i)).toBeInTheDocument();
  });

  it("NÃO exibe ScrollArea quando histórico está vazio", () => {
    render(<AlertPanel latest={null} riskLevel="NORMAL" />);
    expect(screen.queryByTestId("scroll-area")).not.toBeInTheDocument();
  });

  // ── Lista de eventos ────────────────────────────────────────────────────

  it("exibe a ScrollArea quando há entradas no histórico", async () => {
    const mockHook = await getMockHook();
    mockHook.mockReturnValue({
      history: [makeHistoryEntry()],
      isMounted: true,
      clearHistory: mockClear,
    });

    render(<AlertPanel latest={null} riskLevel="NORMAL" />);
    expect(screen.getByTestId("scroll-area")).toBeInTheDocument();
  });

  it("exibe a probabilidade de cada entrada no histórico", async () => {
    const mockHook = await getMockHook();
    mockHook.mockReturnValue({
      history: [
        makeHistoryEntry({ failure_probability: 0.75, riskLevel: "CRÍTICO" }),
      ],
      isMounted: true,
      clearHistory: mockClear,
    });

    const { container } = render(
      <AlertPanel latest={null} riskLevel="NORMAL" />,
    );
    // 0.75 * 100 = 75.0%
    expect(container.textContent).toContain("75.0%");
  });

  it("exibe o riskLevel de cada entrada como badge", async () => {
    const mockHook = await getMockHook();
    mockHook.mockReturnValue({
      history: [
        makeHistoryEntry({ riskLevel: "ALERTA", failure_probability: 0.4 }),
      ],
      isMounted: true,
      clearHistory: mockClear,
    });

    render(<AlertPanel latest={null} riskLevel="NORMAL" />);
    // O badge da entry de histórico deve mostrar ALERTA
    const alertaBadges = screen.getAllByText("ALERTA");
    expect(alertaBadges.length).toBeGreaterThan(0);
  });

  it("entrada CRÍTICO no histórico tem data-risk='CRÍTICO'", async () => {
    const mockHook = await getMockHook();
    mockHook.mockReturnValue({
      history: [
        makeHistoryEntry({ riskLevel: "CRÍTICO", failure_probability: 0.8 }),
      ],
      isMounted: true,
      clearHistory: mockClear,
    });

    const { container } = render(
      <AlertPanel latest={null} riskLevel="NORMAL" />,
    );
    expect(container.querySelector("[data-risk='CRÍTICO']")).not.toBeNull();
  });

  // ── Botão "Limpar" ──────────────────────────────────────────────────────

  it("exibe o botão 'Limpar' quando há entradas no histórico", async () => {
    const mockHook = await getMockHook();
    mockHook.mockReturnValue({
      history: [makeHistoryEntry()],
      isMounted: true,
      clearHistory: mockClear,
    });

    render(<AlertPanel latest={null} riskLevel="NORMAL" />);
    expect(screen.getByRole("button", { name: /limpar/i })).toBeInTheDocument();
  });

  it("NÃO exibe o botão 'Limpar' quando histórico está vazio", () => {
    render(<AlertPanel latest={null} riskLevel="NORMAL" />);
    expect(
      screen.queryByRole("button", { name: /limpar/i }),
    ).not.toBeInTheDocument();
  });

  it("clicar em 'Limpar' chama clearHistory", async () => {
    const mockHook = await getMockHook();
    mockHook.mockReturnValue({
      history: [makeHistoryEntry()],
      isMounted: true,
      clearHistory: mockClear,
    });

    render(<AlertPanel latest={null} riskLevel="NORMAL" />);
    fireEvent.click(screen.getByRole("button", { name: /limpar/i }));
    expect(mockClear).toHaveBeenCalledTimes(1);
  });

  // ── Estado !isMounted (SSR / pré-hidratação) ─────────────────────────────

  it("não exibe a lista nem o estado vazio antes da hidratação (!isMounted)", async () => {
    const mockHook = await getMockHook();
    mockHook.mockReturnValue({
      history: [],
      isMounted: false,
      clearHistory: mockClear,
    });

    render(<AlertPanel latest={null} riskLevel="NORMAL" />);
    expect(screen.queryByTestId("empty-state")).not.toBeInTheDocument();
    expect(screen.queryByTestId("scroll-area")).not.toBeInTheDocument();
  });

  it("não exibe o botão 'Limpar' antes da hidratação (!isMounted)", async () => {
    const mockHook = await getMockHook();
    mockHook.mockReturnValue({
      history: [makeHistoryEntry()],
      isMounted: false,
      clearHistory: mockClear,
    });

    render(<AlertPanel latest={null} riskLevel="NORMAL" />);
    expect(
      screen.queryByRole("button", { name: /limpar/i }),
    ).not.toBeInTheDocument();
  });
});
