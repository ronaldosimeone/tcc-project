/**
 * Testes de integração do HistoryDashboard — RNF-59.
 *
 * `HistoryDashboard` não depende de rede/backend (dados 100% de
 * `lib/history-mock.ts`) — um único render de árvore completa já exercita
 * HistoryHeader, HistoryKPIs, AlertFrequencyChart, EventHeatmap,
 * TopEquipamentos, TiposEvento, HistoryFilters, EventLogTable e
 * RootCauseDrawer (com seus sub-componentes RNF-58) de forma real,
 * interagindo como um usuário faria — filtro, paginação, clique em linha.
 *
 * Botões de exportação (CSV/PDF) do HistoryHeader são testados à parte
 * (history-header.test.tsx) — dependem de APIs de browser que precisam de
 * stub dedicado (URL.createObjectURL, window.print).
 */
import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import HistoryDashboard from "@/components/history/HistoryDashboard";
import { HISTORY_EVENTS } from "@/lib/history-mock";

describe("HistoryDashboard", () => {
  /** Texto normalizado do parágrafo "Últimos N dias · X [de Y] eventos" do
   * HistoryHeader — usar `.textContent` (em vez de `getByText`) porque o
   * JSX quebra essa frase em vários nós de texto irmãos (spans + literais),
   * e o mesmo número (ex.: "50") também aparece em KPIs/rankings. */
  function getHeaderSummaryText(): string {
    const div = screen.getByText("Histórico & Relatórios").closest("div")!;
    return div.textContent!.replace(/\s+/g, " ").trim();
  }

  it("renderiza o cabeçalho com a contagem total de eventos (sem filtro ativo)", () => {
    render(<HistoryDashboard />);
    expect(screen.getByText("Histórico & Relatórios")).toBeInTheDocument();
    // Todos os 50 eventos do mock caem dentro da janela padrão de 30 dias.
    expect(getHeaderSummaryText()).toContain(
      `${HISTORY_EVENTS.length} eventos registrados`,
    );
  });

  it("renderiza os 3 KPIs operacionais (MTTR, Ativo mais Crítico, Downtime)", () => {
    render(<HistoryDashboard />);
    expect(
      screen.getByText("Tempo Médio de Recuperação (MTTR)"),
    ).toBeInTheDocument();
    expect(screen.getByText("Ativo mais Crítico")).toBeInTheDocument();
    expect(screen.getByText("Total de Downtime")).toBeInTheDocument();
  });

  it("renderiza o gráfico de frequência, o heatmap e os rankings de equipamento/tipo", () => {
    render(<HistoryDashboard />);
    expect(screen.getByText("Frequência de Ocorrências")).toBeInTheDocument();
    expect(
      screen.getByText("Mapa de Calor de Ocorrências"),
    ).toBeInTheDocument();
    expect(screen.getByText("Top equipamentos")).toBeInTheDocument();
    expect(screen.getByText("Tipos de evento")).toBeInTheDocument();
  });

  it("renderiza a tabela de eventos paginada (10 por página) com o botão de reset oculto por padrão", () => {
    render(<HistoryDashboard />);
    expect(screen.getByText("Log de Eventos")).toBeInTheDocument();
    // EVT-001 é o evento mais recente do mock — deve estar na 1ª página
    // (ordenação padrão: timestamp desc).
    expect(
      screen.getByText(/Vazamento de ar detectado no compressor/),
    ).toBeInTheDocument();
    // Nenhum filtro ativo ainda — botão "Limpar" dos filtros não aparece.
    expect(
      screen.queryByRole("button", { name: /limpar/i }),
    ).not.toBeInTheDocument();
  });

  it("filtra por severidade CRÍTICO e reduz a contagem de eventos exibidos", () => {
    render(<HistoryDashboard />);
    const criticalCount = HISTORY_EVENTS.filter(
      (e) => e.severity === "CRÍTICO",
    ).length;

    const severitySelect = screen.getByDisplayValue("Todas as severidades");
    fireEvent.change(severitySelect, { target: { value: "CRÍTICO" } });

    // O cabeçalho passa a mostrar "X de 50 eventos" com X = críticos.
    expect(getHeaderSummaryText()).toContain(
      `${criticalCount} de ${HISTORY_EVENTS.length} eventos`,
    );

    // Botão "Limpar" agora aparece, pois há um filtro ativo.
    expect(screen.getByRole("button", { name: /limpar/i })).toBeInTheDocument();
  });

  it("busca por texto filtra a tabela e o botão 'Limpar' reseta os filtros", () => {
    render(<HistoryDashboard />);

    const searchInput = screen.getByPlaceholderText(
      "Buscar eventos ou equipamentos…",
    );
    // "APU-Trem-055" aparece em poucos eventos — filtra a lista visivelmente.
    fireEvent.change(searchInput, { target: { value: "APU-Trem-055" } });

    const expectedCount = HISTORY_EVENTS.filter((e) =>
      e.equipment.toLowerCase().includes("apu-trem-055"),
    ).length;
    expect(getHeaderSummaryText()).toContain(
      `${expectedCount} de ${HISTORY_EVENTS.length} eventos`,
    );

    fireEvent.click(screen.getByRole("button", { name: /limpar/i }));

    expect(searchInput).toHaveValue("");
    expect(getHeaderSummaryText()).toContain(
      `${HISTORY_EVENTS.length} eventos registrados`,
    );
  });

  it("navega para a 2ª página da tabela via paginação", () => {
    render(<HistoryDashboard />);
    // EVT-011 só aparece na 2ª página (10 eventos por página, ordenado por
    // timestamp desc — EVT-011 é o 11º mais recente).
    expect(
      screen.queryByText(/Ciclo de compressão com frequência anômala/),
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Página 2" }));

    expect(
      screen.getByText(/Ciclo de compressão com frequência anômala/),
    ).toBeInTheDocument();
  });

  it("abre o RootCauseDrawer ao clicar numa linha e fecha ao clicar em 'Fechar'", () => {
    render(<HistoryDashboard />);

    const row = screen
      .getByText(/Vazamento de ar detectado no compressor/)
      .closest("tr");
    expect(row).not.toBeNull();
    fireEvent.click(row!);

    // O drawer mostra a descrição completa do evento como título + análise.
    expect(screen.getByText("Análise de Causa Raiz")).toBeInTheDocument();
    expect(
      screen.getByText("Janela preditiva · 2 horas antes da falha"),
    ).toBeInTheDocument();
    expect(screen.getByText("Linha do tempo do incidente")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /fechar/i }));
    expect(screen.queryByText("Análise de Causa Raiz")).not.toBeInTheDocument();
  });

  it("mostra a legenda de intensidade do heatmap com as 4 faixas", () => {
    render(<HistoryDashboard />);
    const legend = screen.getByText("Intensidade").closest("div");
    expect(legend).not.toBeNull();
    expect(within(legend!).getByText("0")).toBeInTheDocument();
    expect(within(legend!).getByText("1–2")).toBeInTheDocument();
    expect(within(legend!).getByText("3–5")).toBeInTheDocument();
    expect(within(legend!).getByText("≥ 6")).toBeInTheDocument();
  });
});
