/**
 * Testes do AssetRadarChart — RNF-59.
 *
 * Componente 100% controlado por props, sem rede/hooks — cobre os estados
 * reais: live vs. idle, loading, e a badge de score de anomalia por faixa
 * (NORMAL/ALERTA/CRÍTICO, ver `toAnomalyLevel`).
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import AssetRadarChart from "@/components/dashboard/AssetRadarChart";
import type { PredictPayload } from "@/lib/api-client";

const SENSOR_DATA: PredictPayload = {
  TP2: 10.1,
  TP3: 10.1,
  H1: 8.5,
  DV_pressure: 1.0,
  Reservoirs: 7.0,
  Motor_current: 3.8,
  Oil_temperature: 64,
  COMP: 1,
  DV_eletric: 0,
  Towers: 1,
  MPG: 0,
  Oil_level: 1,
};

describe("AssetRadarChart", () => {
  it("modo idle: mostra o badge IDLE e nenhum readout bruto", () => {
    render(
      <AssetRadarChart
        sensorData={SENSOR_DATA}
        isLive={false}
        assetId="APU-Trem-015"
      />,
    );
    expect(screen.getByText("IDLE")).toBeInTheDocument();
    expect(screen.queryByText("LIVE")).not.toBeInTheDocument();
    expect(screen.getByText(/Ótimo vs Estático/)).toBeInTheDocument();
    // Sem live, a tira de valores brutos (TP2/Temp/Motor) não é renderizada.
    expect(screen.queryByText("TP2")).not.toBeInTheDocument();
  });

  it("modo live: mostra o badge LIVE e os valores brutos instantâneos", () => {
    render(
      <AssetRadarChart
        sensorData={SENSOR_DATA}
        isLive
        assetId="APU-Trem-042"
        anomalyScore={0.1}
      />,
    );
    expect(screen.getByText("LIVE")).toBeInTheDocument();
    expect(screen.getByText(/Ótimo vs Atual/)).toBeInTheDocument();
    expect(screen.getByText("TP2")).toBeInTheDocument();
    expect(screen.getByText("10.10")).toBeInTheDocument();
    expect(screen.getByText("bar")).toBeInTheDocument();
  });

  it("badge de anomalia NORMAL quando o score é baixo (live)", () => {
    render(
      <AssetRadarChart
        sensorData={SENSOR_DATA}
        isLive
        assetId="APU-Trem-042"
        anomalyScore={0.1}
      />,
    );
    expect(screen.getByText(/NORMAL 10\.0%/)).toBeInTheDocument();
  });

  it("badge de anomalia CRÍTICO quando o score é alto (live)", () => {
    render(
      <AssetRadarChart
        sensorData={SENSOR_DATA}
        isLive
        assetId="APU-Trem-042"
        anomalyScore={0.91}
      />,
    );
    expect(screen.getByText(/CRÍTICO 91\.0%/)).toBeInTheDocument();
  });

  it("badge de anomalia ALERTA para score intermediário (live)", () => {
    render(
      <AssetRadarChart
        sensorData={SENSOR_DATA}
        isLive
        assetId="APU-Trem-042"
        anomalyScore={0.45}
      />,
    );
    expect(screen.getByText(/ALERTA 45\.0%/)).toBeInTheDocument();
  });

  it("exibe o skeleton do gráfico quando isLoading=true", () => {
    const { container } = render(
      <AssetRadarChart
        sensorData={SENSOR_DATA}
        isLive
        assetId="APU-Trem-042"
        isLoading
      />,
    );
    expect(container.querySelector(".animate-pulse")).not.toBeNull();
    // Em loading, a tira de valores brutos também não é exibida.
    expect(screen.queryByText("TP2")).not.toBeInTheDocument();
  });
});
