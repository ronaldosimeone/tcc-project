/**
 * Testes do ErrorBoundary — RNF-59.
 */
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ErrorBoundary } from "@/components/error-boundary";

/** Componente que lança na primeira renderização; some após um reset externo
 * via a prop `shouldThrow` (controlada pelo teste, não por estado interno —
 * simula o cenário real de "o dado que causava o erro foi corrigido"). */
function Bomb({ shouldThrow }: { shouldThrow: boolean }) {
  if (shouldThrow) throw new Error("Falha simulada no Recharts");
  return <div>Conteúdo renderizado com sucesso</div>;
}

describe("ErrorBoundary", () => {
  it("renderiza os filhos normalmente quando não há erro", () => {
    render(
      <ErrorBoundary>
        <div>Painel OK</div>
      </ErrorBoundary>,
    );
    expect(screen.getByText("Painel OK")).toBeInTheDocument();
  });

  it("exibe o fallback padrão e captura a mensagem do erro quando um filho lança", () => {
    // Recharts/React já loga o erro no console durante testes — silenciamos
    // apenas para não poluir a saída, sem afetar a asserção.
    vi.spyOn(console, "error").mockImplementation(() => {});

    render(
      <ErrorBoundary>
        <Bomb shouldThrow />
      </ErrorBoundary>,
    );

    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText("Erro ao renderizar o painel")).toBeInTheDocument();
    expect(screen.getByText("Falha simulada no Recharts")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /tentar novamente/i }),
    ).toBeInTheDocument();

    vi.restoreAllMocks();
  });

  it("exibe mensagem genérica quando o erro capturado não tem `message`", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});

    function ThrowsNonError(): never {
      // Testa deliberadamente o fallback de getDerivedStateFromError para um
      // valor lançado que não é uma instância de Error (React sempre tipa
      // como Error, mas em runtime qualquer valor pode ser lançado).
      throw "string literal, não Error";
    }

    render(
      <ErrorBoundary>
        <ThrowsNonError />
      </ErrorBoundary>,
    );

    expect(
      screen.getByText(
        "Ocorreu um erro inesperado nos componentes de visualização.",
      ),
    ).toBeInTheDocument();

    vi.restoreAllMocks();
  });

  it("renderiza o fallback customizado em vez do padrão quando fornecido", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});

    render(
      <ErrorBoundary fallback={<div>Fallback customizado</div>}>
        <Bomb shouldThrow />
      </ErrorBoundary>,
    );

    expect(screen.getByText("Fallback customizado")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();

    vi.restoreAllMocks();
  });

  it("volta a renderizar os filhos após clicar em 'Tentar novamente'", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});

    // `rerender` troca a prop do filho ANTES do clique — no mundo real isso
    // corresponde a um reload de dados que corrige a causa do erro. O clique
    // em "Tentar novamente" reseta `hasError`, permitindo que a nova árvore
    // (agora sem erro) seja renderizada em vez do fallback congelado.
    const { rerender } = render(
      <ErrorBoundary>
        <Bomb shouldThrow />
      </ErrorBoundary>,
    );
    expect(screen.getByRole("alert")).toBeInTheDocument();

    rerender(
      <ErrorBoundary>
        <Bomb shouldThrow={false} />
      </ErrorBoundary>,
    );
    fireEvent.click(screen.getByRole("button", { name: /tentar novamente/i }));

    expect(
      screen.getByText("Conteúdo renderizado com sucesso"),
    ).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();

    vi.restoreAllMocks();
  });
});
