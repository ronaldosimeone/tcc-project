/**
 * Testes do FleetDashboard — RNF-59.
 *
 * Escopo: a lógica de ORQUESTRAÇÃO própria deste componente — status SSE no
 * cabeçalho, adaptação `currentLatency` → `latencyTelemetry`, cálculo da
 * distribuição de saúde da frota e a wiring de `effectiveRiskLevel`/
 * `effectiveProb` para FleetKPIs/FleetHealthTable (ambos reais, sem rede).
 *
 * `useSensorData` é mockado diretamente (`getRiskLevel` real, via
 * `importActual`) — testar a stream SSE em si é responsabilidade de
 * `use-sse.test.ts`/`sensor-monitor.test.tsx`, não deste arquivo.
 * `EventFeedCard`/`ModelStatusCard` são mockados como stubs simples: cada
 * um já tem cobertura própria dedicada (event-feed-card*.test.tsx,
 * model-status-card.test.tsx) e depende de rede/WS independente do que
 * este componente orquestra.
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import FleetDashboard from "@/components/dashboard/FleetDashboard";

vi.mock("@/components/dashboard/EventFeedCard", () => ({
  default: () => <div data-testid="event-feed-stub" />,
}));
vi.mock("@/components/dashboard/ModelStatusCard", () => ({
  default: ({ distribution }: { distribution: Record<string, number> }) => (
    <div data-testid="model-status-stub">{JSON.stringify(distribution)}</div>
  ),
}));

const mockUseSensorData = vi.fn();
vi.mock("@/hooks/use-sensor-data", async () => {
  const actual = await vi.importActual<
    typeof import("@/hooks/use-sensor-data")
  >("@/hooks/use-sensor-data");
  return {
    ...actual,
    useSensorData: () => mockUseSensorData(),
  };
});

describe("FleetDashboard", () => {
  it("mostra 'Telemetria ao vivo' quando o SSE está conectado", () => {
    mockUseSensorData.mockReturnValue({
      latest: { failure_probability: 0.12, predicted_class: 0, timestamp: "" },
      currentLatency: null,
      isLoading: false,
      sseStatus: "connected",
    });

    render(<FleetDashboard />);
    expect(screen.getByText("Telemetria ao vivo")).toBeInTheDocument();
    expect(screen.getByText("Cockpit Operacional")).toBeInTheDocument();
  });

  it("mostra 'Reconectando…' quando o SSE não está conectado", () => {
    mockUseSensorData.mockReturnValue({
      latest: null,
      currentLatency: null,
      isLoading: true,
      sseStatus: "reconnecting",
    });

    render(<FleetDashboard />);
    expect(screen.getByText("Reconectando…")).toBeInTheDocument();
  });

  it("sem predição ainda (latest=null): trata a probabilidade efetiva como 0 (NORMAL)", () => {
    mockUseSensorData.mockReturnValue({
      latest: null,
      currentLatency: null,
      isLoading: true,
      sseStatus: "connecting",
    });

    render(<FleetDashboard />);
    // distribution: NORMAL contribui +1 aos 3 mocks fixos = 4 saudáveis.
    const distribution = JSON.parse(
      screen.getByTestId("model-status-stub").textContent!,
    );
    expect(distribution).toEqual({ healthy: 4, warning: 1, critical: 0 });
  });

  it("estado CRÍTICO: soma +1 ao contador `critical` da distribuição", () => {
    mockUseSensorData.mockReturnValue({
      latest: { failure_probability: 0.91, predicted_class: 1, timestamp: "" },
      currentLatency: null,
      isLoading: false,
      sseStatus: "connected",
    });

    render(<FleetDashboard />);
    const distribution = JSON.parse(
      screen.getByTestId("model-status-stub").textContent!,
    );
    expect(distribution).toEqual({ healthy: 3, warning: 1, critical: 1 });
  });

  it("adapta currentLatency.key para latencyTelemetry.messageId e repassa ao FleetKPIs", () => {
    mockUseSensorData.mockReturnValue({
      latest: { failure_probability: 0.1, predicted_class: 0, timestamp: "" },
      currentLatency: { key: "frame-42", latencyMs: 37 },
      isLoading: false,
      sseStatus: "connected",
    });

    render(<FleetDashboard />);
    // FleetKPIs real renderiza a leitura de latência adaptada.
    expect(screen.getByText("37")).toBeInTheDocument();
    expect(screen.getByText("ao vivo · 1/24 amostras")).toBeInTheDocument();
  });

  it("renderiza os stubs de EventFeedCard e ModelStatusCard", () => {
    mockUseSensorData.mockReturnValue({
      latest: null,
      currentLatency: null,
      isLoading: true,
      sseStatus: "connecting",
    });

    render(<FleetDashboard />);
    expect(screen.getByTestId("event-feed-stub")).toBeInTheDocument();
    expect(screen.getByTestId("model-status-stub")).toBeInTheDocument();
  });
});
