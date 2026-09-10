/**
 * Testes do HistoryHeader — foco nos botões de exportação (CSV/PDF), que
 * dependem de APIs de browser ausentes/mockáveis em jsdom
 * (URL.createObjectURL, window.print, download via <a>) — por isso vivem
 * separados do teste de integração do HistoryDashboard.
 */
import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import HistoryHeader from "@/components/history/HistoryHeader";

describe("HistoryHeader", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    // jsdom não implementa createObjectURL/revokeObjectURL nem window.print.
    URL.createObjectURL = vi.fn(() => "blob:mock-url");
    URL.revokeObjectURL = vi.fn();
    window.print = vi.fn();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("mostra a contagem total quando não há filtro ativo", () => {
    render(<HistoryHeader totalEvents={50} filteredCount={50} />);
    expect(screen.getByText(/50/).closest("p")?.textContent).toContain(
      "50 eventos registrados",
    );
  });

  it("mostra 'X de Y eventos' quando o filtro reduz a contagem", () => {
    render(<HistoryHeader totalEvents={50} filteredCount={7} />);
    expect(screen.getByText(/7/).closest("p")?.textContent).toContain(
      "7 de 50 eventos",
    );
  });

  it("exporta CSV: cria o blob, dispara o download e volta ao estado normal após o timeout", () => {
    render(<HistoryHeader totalEvents={50} filteredCount={50} />);

    const csvButton = screen.getByRole("button", { name: /exportar csv/i });
    fireEvent.click(csvButton);

    expect(screen.getByText("Exportando…")).toBeInTheDocument();
    expect(csvButton).toBeDisabled();
    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
    // O Blob passado deve carregar o cabeçalho CSV esperado.
    const blobArg = (URL.createObjectURL as ReturnType<typeof vi.fn>).mock
      .calls[0][0] as Blob;
    expect(blobArg.type).toBe("text/csv;charset=utf-8;");

    act(() => {
      vi.advanceTimersByTime(1500);
    });

    expect(screen.getByText("Exportar CSV")).toBeInTheDocument();
    expect(csvButton).not.toBeDisabled();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:mock-url");
  });

  it("exporta PDF: chama window.print() após o timeout e reabilita os botões", () => {
    render(<HistoryHeader totalEvents={50} filteredCount={50} />);

    const pdfButton = screen.getByRole("button", { name: /exportar pdf/i });
    fireEvent.click(pdfButton);

    expect(screen.getByText("Exportando…")).toBeInTheDocument();
    expect(window.print).not.toHaveBeenCalled();

    act(() => {
      vi.advanceTimersByTime(400);
    });

    expect(window.print).toHaveBeenCalledTimes(1);
    expect(screen.getByText("Exportar PDF")).toBeInTheDocument();
    expect(pdfButton).not.toBeDisabled();
  });

  it("ignora cliques repetidos enquanto uma exportação já está em andamento", () => {
    render(<HistoryHeader totalEvents={50} filteredCount={50} />);

    const csvButton = screen.getByRole("button", { name: /exportar csv/i });
    const pdfButton = screen.getByRole("button", { name: /exportar pdf/i });

    fireEvent.click(csvButton);
    fireEvent.click(pdfButton); // ambos os botões ficam disabled — clique é ignorado

    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
    expect(window.print).not.toHaveBeenCalled();

    act(() => {
      vi.advanceTimersByTime(1500);
    });
  });
});
