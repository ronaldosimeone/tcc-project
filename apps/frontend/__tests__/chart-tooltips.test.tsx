/**
 * Testes dos tooltips customizados do Recharts espalhados pelos gráficos
 * decompostos (RNF-58) — RNF-59.
 *
 * Cada `content={<XyzTooltip />}` do Recharts é, em si, apenas um
 * componente React — o Recharts só o invoca com `active`/`payload` reais
 * durante hover num chart renderizado, o que não acontece em jsdom (sem
 * layout engine, `ResponsiveContainer` computa largura/altura 0). Por
 * isso cada tooltip é testado aqui isoladamente, chamando-o como qualquer
 * componente com as props que o Recharts passaria em runtime.
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ChartTooltip as SensorChartTooltip } from "@/components/sensor-chart/chart-tooltip";
import { ChartTooltip as RadarTooltip } from "@/components/dashboard/asset-radar-chart/radar-tooltip";
import { ChartTooltip as EfficiencyTooltip } from "@/components/dashboard/asset-efficiency-chart/chart-tooltip";
import { CustomTooltip as FrequencyTooltip } from "@/components/history/alert-frequency-chart/chart-tooltip";
import { PredictiveTooltip } from "@/components/history/root-cause-drawer/predictive-tooltip";

describe("SensorChart ChartTooltip", () => {
  it("não renderiza nada quando inativo ou sem payload", () => {
    const { container: c1 } = render(<SensorChartTooltip active={false} />);
    expect(c1).toBeEmptyDOMElement();
    const { container: c2 } = render(
      <SensorChartTooltip active payload={[]} />,
    );
    expect(c2).toBeEmptyDOMElement();
  });

  it("renderiza o label e cada série com seu valor", () => {
    render(
      <SensorChartTooltip
        active
        label="10:00:05"
        payload={[
          { name: "TP2", value: 8.1, color: "#60a5fa" },
          { name: "TP3", value: 7.9, color: "#4ade80" },
        ]}
      />,
    );
    expect(screen.getByText("10:00:05")).toBeInTheDocument();
    expect(screen.getByText("TP2")).toBeInTheDocument();
    expect(screen.getByText("8.1")).toBeInTheDocument();
    expect(screen.getByText("TP3")).toBeInTheDocument();
    expect(screen.getByText("7.9")).toBeInTheDocument();
  });
});

describe("AssetRadarChart ChartTooltip", () => {
  it("não renderiza nada quando inativo", () => {
    const { container } = render(<RadarTooltip active={false} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renderiza o valor normalizado com sufixo /100", () => {
    render(
      <RadarTooltip
        active
        label="TP2"
        payload={[{ name: "Ótimo", value: 84.2, color: "#4ade80" }]}
      />,
    );
    expect(screen.getByText("TP2")).toBeInTheDocument();
    expect(screen.getByText("84")).toBeInTheDocument();
    expect(screen.getByText("/100")).toBeInTheDocument();
  });
});

describe("AssetEfficiencyChart ChartTooltip", () => {
  it("não renderiza nada sem payload", () => {
    const { container } = render(<EfficiencyTooltip active payload={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("calcula a % de eficiência e mostra carga/ocioso em horas", () => {
    render(
      <EfficiencyTooltip
        active
        label="Seg"
        payload={[
          { name: "Em Carga", value: 20, color: "#60a5fa", dataKey: "carga" },
          { name: "Ocioso", value: 4, color: "#c084fc", dataKey: "ocioso" },
        ]}
      />,
    );
    expect(screen.getByText("Seg")).toBeInTheDocument();
    expect(screen.getByText("83% efic.")).toBeInTheDocument();
    expect(screen.getByText("20h")).toBeInTheDocument();
    expect(screen.getByText("4h")).toBeInTheDocument();
    expect(screen.getByText("24h")).toBeInTheDocument(); // total
  });
});

describe("AlertFrequencyChart CustomTooltip", () => {
  it("mostra 'Sem ocorrências' quando total é 0", () => {
    render(
      <FrequencyTooltip
        active
        label="01/04"
        payload={[
          { dataKey: "critico", value: 0 },
          { dataKey: "alerta", value: 0 },
        ]}
      />,
    );
    expect(screen.getByText("Sem ocorrências")).toBeInTheDocument();
  });

  it("mostra crítico e alerta separadamente + total quando > 0", () => {
    render(
      <FrequencyTooltip
        active
        label="02/04"
        payload={[
          { dataKey: "critico", value: 2 },
          { dataKey: "alerta", value: 3 },
        ]}
      />,
    );
    expect(screen.getByText("Crítico")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText("Alerta")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument();
    expect(screen.getByText("Total")).toBeInTheDocument();
    expect(screen.getByText("5")).toBeInTheDocument();
  });

  it("omite a linha de 'Crítico' quando não há eventos críticos", () => {
    render(
      <FrequencyTooltip
        active
        label="03/04"
        payload={[
          { dataKey: "critico", value: 0 },
          { dataKey: "alerta", value: 4 },
        ]}
      />,
    );
    expect(screen.queryByText("Crítico")).not.toBeInTheDocument();
    expect(screen.getByText("Alerta")).toBeInTheDocument();
  });
});

describe("RootCauseDrawer PredictiveTooltip", () => {
  it("não renderiza nada quando inativo", () => {
    const { container } = render(<PredictiveTooltip active={false} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("mostra a probabilidade em % com estilo de alerta abaixo do threshold crítico", () => {
    render(
      <PredictiveTooltip active label="-20min" payload={[{ value: 0.42 }]} />,
    );
    expect(screen.getByText("-20min")).toBeInTheDocument();
    expect(screen.getByText("42.0% prob.")).toBeInTheDocument();
  });

  it("mostra a probabilidade em % no estilo crítico acima do threshold (0.65)", () => {
    render(<PredictiveTooltip active label="0" payload={[{ value: 0.91 }]} />);
    expect(screen.getByText("91.0% prob.")).toBeInTheDocument();
  });
});
