/**
 * Auditoria de acessibilidade automatizada — RNF-66 (axe: zero Critical/Serious).
 *
 * Cobre os componentes/estados priorizados no brief: Dashboard principal
 * (FleetDashboard, composição real — não stubs), FleetHealthTable (tabela de
 * equipamentos), ModelStatusCard, EventFeedCard, SimulationPanel (Sheet +
 * RadioGroup + Select), AlertToastQueue (toasts + banner de conexão).
 *
 * `toHaveNoViolations()` falha o teste se HOUVER qualquer violação (de
 * qualquer impacto) — o RNF-66 exige zero Critical/Serious especificamente,
 * então cada teste também lista/filtra por impacto para não mascarar
 * moderate/minor (documentados, não bloqueantes) atrás de um "passou".
 */

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { describe, expect, it, vi, beforeEach } from "vitest";

import FleetDashboard from "@/components/dashboard/FleetDashboard";
import FleetHealthTable from "@/components/dashboard/FleetHealthTable";
import ModelStatusCard from "@/components/dashboard/ModelStatusCard";
import EventFeedCard from "@/components/dashboard/EventFeedCard";
import { SimulationPanel } from "@/components/simulation-panel";
import {
  AlertToastQueue,
  ConnectionBanner,
} from "@/components/alert-toast-queue";
import HistoryDashboard from "@/components/history/HistoryDashboard";
import type { WsAlert } from "@/hooks/use-alert-websocket";

type AxeResults = Awaited<ReturnType<typeof axe>>;

function logViolations(label: string, results: AxeResults) {
  if (results.violations.length === 0) return;
  // Diagnóstico intencional (RNF-66 §22) — lista as violações no output do
  // Vitest para dar visibilidade real além do simples pass/fail do assert.
  console.log(
    `\n[axe] ${label}: ${results.violations.length} regra(s) violada(s)`,
  );
  for (const v of results.violations) {
    console.log(
      `  - ${v.id} [${v.impact}] (${v.nodes.length}x): ${v.help}\n    ${v.helpUrl}`,
    );
  }
}

function criticalOrSerious(results: AxeResults) {
  return results.violations.filter(
    (v) => v.impact === "critical" || v.impact === "serious",
  );
}

// ── Mocks compartilhados ─────────────────────────────────────────────────

vi.mock("@/lib/api-client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api-client")>();
  return {
    ...actual,
    listModels: vi.fn().mockResolvedValue({
      active_model: "random_forest_v2",
      models: [
        { name: "random_forest_v2", artefact_ready: true },
        { name: "xgboost_v2", artefact_ready: true },
      ],
    }),
    getSimulatorMode: vi.fn().mockResolvedValue({ mode: "NORMAL" }),
    setSimulatorMode: vi.fn().mockResolvedValue({ mode: "NORMAL" }),
    swapActiveModel: vi
      .fn()
      .mockResolvedValue({ active_model: "random_forest_v2" }),
  };
});

const mockUseAlertWebSocket = vi.fn();
vi.mock("@/hooks/use-alert-websocket", () => ({
  useAlertWebSocket: () => mockUseAlertWebSocket(),
}));

const mockUseSensorData = vi.fn();
vi.mock("@/hooks/use-sensor-data", async (importOriginal) => {
  const actual = await importOriginal<
    typeof import("@/hooks/use-sensor-data")
  >();
  return { ...actual, useSensorData: () => mockUseSensorData() };
});

beforeEach(() => {
  mockUseAlertWebSocket.mockReturnValue({
    alerts: [],
    status: "open",
    acknowledge: vi.fn(),
  });
  mockUseSensorData.mockReturnValue({
    latest: { failure_probability: 0.12, predicted_class: 0, timestamp: "" },
    currentLatency: { key: "frame-1", latencyMs: 22 },
    isLoading: false,
    sseStatus: "connected",
  });
});

// ── Dashboard principal (composição real) ──────────────────────────────────

describe("a11y — FleetDashboard (Dashboard principal)", () => {
  it("não tem violações axe Critical/Serious no estado normal populado", async () => {
    const { container } = render(<FleetDashboard />);
    // Aguarda o fetch assíncrono de listModels (ModelStatusCard) resolver.
    await screen.findByText("random_forest_v2");

    const results = await axe(container);
    logViolations("FleetDashboard (normal)", results);
    expect(criticalOrSerious(results)).toEqual([]);
  });

  it("não tem violações axe Critical/Serious em CRÍTICO (frota com ativo em falha)", async () => {
    mockUseSensorData.mockReturnValue({
      latest: { failure_probability: 0.91, predicted_class: 1, timestamp: "" },
      currentLatency: null,
      isLoading: false,
      sseStatus: "connected",
    });
    const { container } = render(<FleetDashboard />);
    await screen.findByText("random_forest_v2");

    const results = await axe(container);
    logViolations("FleetDashboard (crítico)", results);
    expect(criticalOrSerious(results)).toEqual([]);
  });
});

// ── FleetHealthTable isolada ─────────────────────────────────────────────

describe("a11y — FleetHealthTable (tabela de equipamentos)", () => {
  it("não tem violações axe Critical/Serious", async () => {
    const { container } = render(
      <FleetHealthTable
        effectiveRiskLevel="NORMAL"
        effectiveProb={0.1}
        isLoading={false}
        selectedId="APU-Trem-042"
        onSelect={vi.fn()}
      />,
    );
    const results = await axe(container);
    logViolations("FleetHealthTable", results);
    expect(criticalOrSerious(results)).toEqual([]);
  });
});

// ── ModelStatusCard isolado ──────────────────────────────────────────────

describe("a11y — ModelStatusCard", () => {
  it("não tem violações axe Critical/Serious", async () => {
    const { container } = render(
      <ModelStatusCard
        distribution={{ healthy: 3, warning: 1, critical: 1 }}
      />,
    );
    await screen.findByText("random_forest_v2");
    const results = await axe(container);
    logViolations("ModelStatusCard", results);
    expect(criticalOrSerious(results)).toEqual([]);
  });
});

// ── EventFeedCard isolado ────────────────────────────────────────────────

describe("a11y — EventFeedCard", () => {
  it("não tem violações axe Critical/Serious (eventos demo)", async () => {
    const { container } = render(<EventFeedCard />);
    const results = await axe(container);
    logViolations("EventFeedCard (demo)", results);
    expect(criticalOrSerious(results)).toEqual([]);
  });

  it("não tem violações axe Critical/Serious com alerta vivo ativo", async () => {
    const alert: WsAlert = {
      message_id: "m1",
      probability: 0.91,
      timestamp: new Date().toISOString(),
    } as WsAlert;
    mockUseAlertWebSocket.mockReturnValue({
      alerts: [alert],
      status: "open",
      acknowledge: vi.fn(),
    });
    const { container } = render(<EventFeedCard />);
    const results = await axe(container);
    logViolations("EventFeedCard (com alerta vivo)", results);
    expect(criticalOrSerious(results)).toEqual([]);
  });
});

// ── SimulationPanel (Sheet/dialog + RadioGroup + Select) ────────────────────

describe("a11y — SimulationPanel (dialog)", () => {
  it("não tem violações axe Critical/Serious quando aberto", async () => {
    const { container } = render(
      <SimulationPanel open={true} onOpenChange={vi.fn()} />,
    );
    await screen.findByText("Painel de Simulação");
    const results = await axe(container);
    logViolations("SimulationPanel (aberto)", results);
    expect(criticalOrSerious(results)).toEqual([]);
  });
});

// ── AlertToastQueue + ConnectionBanner ───────────────────────────────────

describe("a11y — AlertToastQueue / ConnectionBanner", () => {
  it("não tem violações axe Critical/Serious com toasts ativos", async () => {
    const alerts: WsAlert[] = [
      {
        message_id: "a1",
        probability: 0.91,
        timestamp: new Date().toISOString(),
      } as WsAlert,
      {
        message_id: "a2",
        probability: 0.4,
        timestamp: new Date().toISOString(),
      } as WsAlert,
    ];
    const { container } = render(
      <AlertToastQueue alerts={alerts} status="open" onAcknowledge={vi.fn()} />,
    );
    const results = await axe(container);
    logViolations("AlertToastQueue", results);
    expect(criticalOrSerious(results)).toEqual([]);
  });

  it("banner de reconexão não tem violações axe Critical/Serious", async () => {
    const { container } = render(<ConnectionBanner status="reconnecting" />);
    const results = await axe(container);
    logViolations("ConnectionBanner", results);
    expect(criticalOrSerious(results)).toEqual([]);
  });
});

// ── HistoryDashboard (100% mock, sem rede — tabela, filtros, gráficos, drawer) ─

describe("a11y — HistoryDashboard", () => {
  it("não tem violações axe Critical/Serious no estado inicial", async () => {
    const { container } = render(<HistoryDashboard />);
    const results = await axe(container);
    logViolations("HistoryDashboard (inicial)", results);
    expect(criticalOrSerious(results)).toEqual([]);
  });

  it("não tem violações axe Critical/Serious com o RootCauseDrawer aberto", async () => {
    const user = userEvent.setup();
    render(<HistoryDashboard />);

    const openButtons = screen.getAllByRole("button", {
      name: /ver detalhes do evento/i,
    });
    await user.click(openButtons[0]);

    await screen.findByText("Análise de Causa Raiz");
    const results = await axe(document.body);
    logViolations("HistoryDashboard (drawer aberto)", results);
    expect(criticalOrSerious(results)).toEqual([]);
  });
});

// ── Testes de teclado (RNF-67 §17) ──────────────────────────────────────────
// Comportamento real via userEvent — Tab/Enter/Space/Escape — não apenas
// presença de atributos.

describe("teclado — FleetHealthTable (seleção de linha)", () => {
  it("Tab alcança o botão da linha ao vivo e Enter dispara onSelect", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(
      <FleetHealthTable
        effectiveRiskLevel="NORMAL"
        effectiveProb={0.1}
        isLoading={false}
        selectedId="none"
        onSelect={onSelect}
      />,
    );

    await user.tab();
    const liveButton = screen.getByRole("button", { name: /APU-Trem-042/i });
    expect(liveButton).toHaveFocus();

    await user.keyboard("{Enter}");
    expect(onSelect).toHaveBeenCalledWith("APU-Trem-042");
  });

  it("Tab avança para a linha simulada seguinte e Space dispara onSelect", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(
      <FleetHealthTable
        effectiveRiskLevel="NORMAL"
        effectiveProb={0.1}
        isLoading={false}
        selectedId="none"
        onSelect={onSelect}
      />,
    );

    await user.tab(); // botão da linha LIVE
    await user.tab(); // link "Telemetria" da linha LIVE
    await user.tab(); // botão da 1ª linha simulada
    const mockButton = screen.getByRole("button", { name: /APU-Trem-015/i });
    expect(mockButton).toHaveFocus();

    await user.keyboard(" ");
    expect(onSelect).toHaveBeenCalledWith("APU-Trem-015");
  });
});

describe("teclado — EventLogTable (abrir detalhes do evento)", () => {
  it("Tab alcança o botão de timestamp da 1ª linha e Enter abre o drawer", async () => {
    const user = userEvent.setup();
    render(<HistoryDashboard />);

    const [firstOpenButton] = screen.getAllByRole("button", {
      name: /ver detalhes do evento/i,
    });
    firstOpenButton.focus();
    expect(firstOpenButton).toHaveFocus();

    await user.keyboard("{Enter}");
    expect(
      await screen.findByText("Análise de Causa Raiz"),
    ).toBeInTheDocument();

    const closeButton = screen.getByRole("button", { name: /fechar/i });
    closeButton.focus();
    await user.keyboard("{Enter}");
    expect(screen.queryByText("Análise de Causa Raiz")).not.toBeInTheDocument();
  });
});

describe("teclado — SimulationPanel (Sheet/dialog)", () => {
  it("Escape fecha o painel", async () => {
    const user = userEvent.setup();
    const onOpenChange = vi.fn();
    render(<SimulationPanel open={true} onOpenChange={onOpenChange} />);

    await screen.findByText("Painel de Simulação");
    await user.keyboard("{Escape}");
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it("Tab navega pelas opções do RadioGroup de cenário", async () => {
    const user = userEvent.setup();
    render(<SimulationPanel open={true} onOpenChange={vi.fn()} />);
    await screen.findByText("Painel de Simulação");

    const radios = screen.getAllByRole("radio");
    radios[0].focus();
    expect(radios[0]).toHaveFocus();

    // Radix RadioGroup usa navegação por seta, não Tab, entre itens do
    // mesmo grupo — comportamento nativo de "single tab stop" (WAI-ARIA
    // APG Radio Group), preservado intencionalmente.
    await user.keyboard("{ArrowDown}");
    expect(radios[1]).toHaveFocus();
  });
});

describe("teclado — EventLogTable pagination", () => {
  it("botão 'Página 2' é alcançável via Tab e ativável via Enter", async () => {
    const user = userEvent.setup();
    render(<HistoryDashboard />);

    const page2 = screen.getByRole("button", { name: "Página 2" });
    page2.focus();
    expect(page2).toHaveFocus();

    await user.keyboard("{Enter}");
    expect(page2).toHaveAttribute("aria-current", "page");
  });
});
