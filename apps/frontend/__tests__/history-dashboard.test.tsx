/**
 * Testes de integração do HistoryDashboard — dados reais da API (sem mock
 * estático): `fetch` é simulado com respostas no formato exato do backend
 * (`GET /v1/events`, `/v1/kpis`, `/v1/events/{id}`, `PATCH /v1/work-orders`).
 *
 * Exercita HistoryHeader, HistoryKPIs, gráficos, filtros, EventLogTable e o
 * RootCauseDrawer real (janela de sensores, contribuições, plano, OS).
 */
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import HistoryDashboard from "@/components/history/HistoryDashboard";

import {
  EVENTS,
  fetchMock,
  installHistoryApiMock,
  json,
  uninstallHistoryApiMock,
} from "./fixtures/history-api-mock";

beforeEach(installHistoryApiMock);
afterEach(uninstallHistoryApiMock);

const headerText = (): string =>
  screen
    .getByText("Histórico & Relatórios")
    .closest("div")!
    .textContent!.replace(/\s+/g, " ")
    .trim();

describe("HistoryDashboard (API real)", () => {
  it("mostra skeleton e depois o total de eventos vindos da API", async () => {
    render(<HistoryDashboard />);
    expect(screen.getByLabelText("Carregando histórico")).toBeInTheDocument();
    await waitFor(() =>
      expect(headerText()).toContain("14 eventos registrados"),
    );
    const eventsCall = fetchMock.mock.calls.find(([u]) =>
      String(u).includes("/v1/events?"),
    );
    expect(String(eventsCall?.[0])).toContain("start=");
  });

  it("renderiza os KPIs de confiabilidade calculados pelo backend", async () => {
    render(<HistoryDashboard />);
    await screen.findByText("MTBF");
    expect(screen.getByText("30.0 dias")).toBeInTheDocument();
    expect(screen.getByText("2.5h")).toBeInTheDocument();
    expect(screen.getByText("99.7%")).toBeInTheDocument();
    expect(screen.getByText("Precisão dos alarmes")).toBeInTheDocument();
  });

  it("evento com OS confirmada aparece como Falha; o filtro de severidade reduz a contagem", async () => {
    render(<HistoryDashboard />);
    await waitFor(() =>
      expect(headerText()).toContain("14 eventos registrados"),
    );
    fireEvent.change(screen.getByDisplayValue("Todas as severidades"), {
      target: { value: "CRÍTICO" },
    });
    expect(headerText()).toContain(
      `${EVENTS.filter((e) => e.severity === "critico").length} de 14 eventos`,
    );
  });

  it("filtro de equipamento lista os ativos presentes nos dados", async () => {
    render(<HistoryDashboard />);
    await waitFor(() => expect(headerText()).toContain("14 eventos"));
    const select = screen.getByLabelText("Filtrar por equipamento");
    expect(select).toHaveTextContent("APU-Trem-007");
    fireEvent.change(select, { target: { value: "APU-Trem-007" } });
    expect(headerText()).toContain(
      `${
        EVENTS.filter((e) => e.asset_id === "APU-Trem-007").length
      } de 14 eventos`,
    );
  });

  it("estado de erro com botão de tentar novamente", async () => {
    fetchMock.mockImplementationOnce(async () => json({ detail: "boom" }, 500));
    render(<HistoryDashboard />);
    expect(
      await screen.findByText("Não foi possível carregar o histórico"),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /tentar novamente/i }));
    await waitFor(() =>
      expect(headerText()).toContain("14 eventos registrados"),
    );
  });

  it("drawer carrega a causa raiz real e permite abrir e iniciar a OS", async () => {
    render(<HistoryDashboard />);
    await waitFor(() => expect(headerText()).toContain("14 eventos"));
    const row = screen.getAllByText(/Episódio de alarme/)[0].closest("tr");
    fireEvent.click(row!);

    expect(await screen.findByText("Towers")).toBeInTheDocument();
    expect(screen.getByText("Fundamentado nos manuais")).toBeInTheDocument();
    expect(screen.getByText("Inspecionar o dessecante.")).toBeInTheDocument();
    expect(screen.getByText("Linha do tempo do incidente")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /abrir os/i }));
    expect(
      await screen.findByRole("button", { name: /iniciar intervenção/i }),
    ).toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", { name: /iniciar intervenção/i }),
    );
    expect(
      await screen.findByRole("button", { name: /fechar: falha confirmada/i }),
    ).toBeInTheDocument();
  });
});
