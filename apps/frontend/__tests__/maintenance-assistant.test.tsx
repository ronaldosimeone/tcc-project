/**
 * Testes de `MaintenanceAssistant` (RF-23 / RNF-47).
 *
 * `streamMaintenanceSuggestion` é mockado como um async generator controlado
 * pelo teste — o hook real (`useMaintenanceStream`) e o componente real são
 * exercitados, então isto testa comportamento/renderização de verdade, não
 * só a existência do componente.
 */

import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { MaintenanceAssistant } from "@/components/maintenance-assistant";
import type { MaintenanceStreamEvent } from "@/lib/maintenance-stream";

const { streamMaintenanceSuggestion } = vi.hoisted(() => ({
  streamMaintenanceSuggestion: vi.fn(),
}));

vi.mock("@/lib/maintenance-stream", () => ({
  streamMaintenanceSuggestion,
}));

async function* eventsOf(...events: MaintenanceStreamEvent[]) {
  for (const event of events) yield event;
}

function submit(): void {
  fireEvent.click(screen.getByRole("button", { name: /gerar sugestão/i }));
}

describe("MaintenanceAssistant", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  // ── A) Markdown simples: heading, paragraph, bold, list ─────────────────────

  it("A) renderiza heading, parágrafo, negrito e lista a partir do Markdown", async () => {
    streamMaintenanceSuggestion.mockImplementation(() =>
      eventsOf({
        type: "done",
        markdown:
          "# Título\n\nTexto com **negrito**.\n\n- item um\n- item dois",
        references: [],
      }),
    );

    render(<MaintenanceAssistant open onOpenChange={vi.fn()} />);
    submit();

    await waitFor(() => {
      expect(
        screen.getByRole("heading", { name: "Título" }),
      ).toBeInTheDocument();
    });
    // "Texto com **negrito**." vira nós de texto separados ao redor de
    // <strong> — verifica o parágrafo inteiro, não um match de texto parcial.
    const paragraph = screen.getByText("negrito").closest("p");
    expect(paragraph?.textContent).toBe("Texto com negrito.");
    expect(screen.getByText("negrito").tagName).toBe("STRONG");
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
  });

  // ── B) Markdown longo ────────────────────────────────────────────────────────

  it("B) renderiza um Markdown longo (100+ linhas) sem lançar exceção", async () => {
    const longMarkdown = Array.from(
      { length: 120 },
      (_, i) => `## Seção ${i}\n\nConteúdo da seção ${i}.`,
    ).join("\n\n");
    streamMaintenanceSuggestion.mockImplementation(() =>
      eventsOf({ type: "done", markdown: longMarkdown, references: [] }),
    );

    render(<MaintenanceAssistant open onOpenChange={vi.fn()} />);
    submit();

    await waitFor(() => {
      expect(screen.getByText("Conteúdo da seção 119.")).toBeInTheDocument();
    });
    // +1 = o <h2> do próprio SheetTitle ("Assistente de Manutenção") do Radix.
    const sectionHeadings = screen
      .getAllByRole("heading", { level: 2 })
      .filter((h) => h.textContent?.startsWith("Seção"));
    expect(sectionHeadings).toHaveLength(120);
  });

  // ── C) Markdown incremental ──────────────────────────────────────────────────

  it("C) atualiza o Markdown progressivamente conforme os tokens chegam", async () => {
    let resolveNext: (() => void) | null = null;
    const gate = () =>
      new Promise<void>((resolve) => {
        resolveNext = resolve;
      });

    streamMaintenanceSuggestion.mockImplementation(async function* () {
      yield { type: "token", token: "# Plano" } as const;
      await gate();
      yield { type: "token", token: " de Manutenção" } as const;
    });

    render(<MaintenanceAssistant open onOpenChange={vi.fn()} />);
    submit();

    // "# Plano" já virou <h1>Plano</h1> (a sintaxe `#` é consumida pelo
    // parser Markdown, nunca aparece literalmente na tela).
    await waitFor(() => {
      expect(
        screen.getByRole("heading", { name: "Plano" }),
      ).toBeInTheDocument();
    });
    // Ainda não recebeu o segundo token — o texto completo não deve existir.
    expect(
      screen.queryByRole("heading", { name: "Plano de Manutenção" }),
    ).not.toBeInTheDocument();

    await act(async () => {
      resolveNext?.();
    });

    await waitFor(() => {
      expect(
        screen.getByRole("heading", { name: "Plano de Manutenção" }),
      ).toBeInTheDocument();
    });
  });

  // ── D/E) Referências — texto (nunca link inventado) + página preservada ────

  it("D/E) exibe as referências como texto com nome de arquivo e página, sem criar link", async () => {
    streamMaintenanceSuggestion.mockImplementation(() =>
      eventsOf({
        type: "done",
        markdown: "# Plano",
        references: [
          {
            file_name: "manual-bomba-centrifuga.pdf",
            page: 3,
            chunk_index: 0,
            source: "manual-bomba-centrifuga.pdf",
            score: 0.71,
          },
        ],
      }),
    );

    render(<MaintenanceAssistant open onOpenChange={vi.fn()} />);
    submit();

    await waitFor(() => {
      expect(
        screen.getByText("manual-bomba-centrifuga.pdf"),
      ).toBeInTheDocument();
    });
    expect(screen.getByText(/página 3/)).toBeInTheDocument();
    // RF-23 §7: sem URL segura conhecida, a referência NUNCA vira <a>.
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });

  // ── F) estado gerando ────────────────────────────────────────────────────────

  it("F) mostra o estado 'Gerando…' enquanto tokens chegam", async () => {
    streamMaintenanceSuggestion.mockImplementation(async function* () {
      yield { type: "token", token: "parcial" } as const;
      await new Promise(() => {}); // nunca resolve — mantém em "generating"
    });

    render(<MaintenanceAssistant open onOpenChange={vi.fn()} />);
    submit();

    await waitFor(() => {
      expect(screen.getByText("Gerando…")).toBeInTheDocument();
    });
  });

  // ── G) estado concluído ──────────────────────────────────────────────────────

  it("G) mostra o estado 'Concluído' após o evento done", async () => {
    streamMaintenanceSuggestion.mockImplementation(() =>
      eventsOf({ type: "done", markdown: "# Ok", references: [] }),
    );

    render(<MaintenanceAssistant open onOpenChange={vi.fn()} />);
    submit();

    await waitFor(() => {
      expect(screen.getByText("Concluído")).toBeInTheDocument();
    });
  });

  // ── H) estado de erro ────────────────────────────────────────────────────────

  it("H) mostra uma mensagem de erro clara quando o serviço reporta uma falha", async () => {
    streamMaintenanceSuggestion.mockImplementation(() =>
      eventsOf({ type: "error", message: "MCP indisponível.", offline: false }),
    );

    render(<MaintenanceAssistant open onOpenChange={vi.fn()} />);
    submit();

    await waitFor(() => {
      expect(screen.getByText("MCP indisponível.")).toBeInTheDocument();
    });
    expect(screen.getByText("Erro")).toBeInTheDocument();
  });

  // ── I) estado offline ────────────────────────────────────────────────────────

  it("I) diferencia o estado offline de um erro reportado pelo backend", async () => {
    streamMaintenanceSuggestion.mockImplementation(() =>
      eventsOf({
        type: "error",
        message: "Não foi possível conectar ao servidor.",
        offline: true,
      }),
    );

    render(<MaintenanceAssistant open onOpenChange={vi.fn()} />);
    submit();

    await waitFor(() => {
      expect(screen.getByText("Offline")).toBeInTheDocument();
    });
    expect(
      screen.getByText("Não foi possível conectar ao servidor."),
    ).toBeInTheDocument();
  });

  // ── J) ausência de referências ───────────────────────────────────────────────

  it("J) informa claramente quando nenhuma referência foi retornada", async () => {
    streamMaintenanceSuggestion.mockImplementation(() =>
      eventsOf({ type: "done", markdown: "# Sem referências", references: [] }),
    );

    render(<MaintenanceAssistant open onOpenChange={vi.fn()} />);
    submit();

    await waitFor(() => {
      expect(
        screen.getByText(
          "Nenhum manual foi citado como referência para este plano.",
        ),
      ).toBeInTheDocument();
    });
  });

  // ── K) conteúdo malicioso ────────────────────────────────────────────────────

  it("K) neutraliza HTML bruto e links javascript: vindos do Markdown gerado", async () => {
    streamMaintenanceSuggestion.mockImplementation(() =>
      eventsOf({
        type: "done",
        markdown:
          "# Plano\n\n<script>window.__xss = true</script>\n\n[clique aqui](javascript:alert(1))",
        references: [],
      }),
    );

    const { container } = render(
      <MaintenanceAssistant open onOpenChange={vi.fn()} />,
    );
    submit();

    await waitFor(() => {
      expect(
        screen.getByRole("heading", { name: "Plano" }),
      ).toBeInTheDocument();
    });

    // Nenhum <script> real foi inserido no DOM — sem rehype-raw, a tag vira
    // texto literal, nunca um elemento executável.
    expect(container.querySelector("script")).not.toBeInTheDocument();
    expect((globalThis as Record<string, unknown>).__xss).toBeUndefined();

    // O link javascript: nunca vira um <a> clicável — isSafeHref bloqueia.
    const dangerousLink = container.querySelector('a[href^="javascript:"]');
    expect(dangerousLink).not.toBeInTheDocument();
  });

  // ── L) cancelamento ──────────────────────────────────────────────────────────

  it("L) cancelar para de processar novos tokens e volta ao estado idle", async () => {
    let yieldedSecondToken = false;
    streamMaintenanceSuggestion.mockImplementation(async function* (
      _payload: unknown,
      signal: AbortSignal,
    ) {
      yield { type: "token", token: "primeiro" } as const;
      await new Promise((resolve) => setTimeout(resolve, 20));
      if (signal.aborted) return;
      yieldedSecondToken = true;
      yield { type: "token", token: "segundo" } as const;
    });

    render(<MaintenanceAssistant open onOpenChange={vi.fn()} />);
    submit();

    await waitFor(() => {
      expect(screen.getByText("Cancelar")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText("Cancelar"));

    await waitFor(() => {
      expect(screen.getByText("Pronto")).toBeInTheDocument();
    });

    await new Promise((resolve) => setTimeout(resolve, 40));
    expect(yieldedSecondToken).toBe(false);
  });

  // ── M) resposta vazia ────────────────────────────────────────────────────────

  it("M) não quebra quando o evento done chega com markdown vazio", async () => {
    streamMaintenanceSuggestion.mockImplementation(() =>
      eventsOf({ type: "done", markdown: "", references: [] }),
    );

    render(<MaintenanceAssistant open onOpenChange={vi.fn()} />);
    submit();

    await waitFor(() => {
      expect(screen.getByText("Concluído")).toBeInTheDocument();
    });
  });
});
