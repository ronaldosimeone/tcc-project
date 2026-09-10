/**
 * Testes do MswProvider — RNF-59.
 *
 * `MSW_ENABLED` é lida de `process.env.NEXT_PUBLIC_MSW_ENABLED` no MOMENTO
 * DO IMPORT do módulo (constante top-level, não recalculada por render) —
 * por isso cada cenário usa `vi.resetModules()` + `await import(...)`
 * dinâmico para forçar uma reavaliação do módulo com o env stubado antes.
 */
import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

describe("MswProvider", () => {
  beforeEach(() => {
    vi.resetModules();
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.doUnmock("@/mocks/browser");
  });

  it("renderiza os filhos imediatamente quando MSW está desabilitado (produção)", async () => {
    vi.stubEnv("NEXT_PUBLIC_MSW_ENABLED", "false");
    const { MswProvider } = await import("@/components/msw-provider");

    render(
      <MswProvider>
        <div>App real</div>
      </MswProvider>,
    );

    expect(screen.getByText("App real")).toBeInTheDocument();
    // Sem MSW habilitado, o span de sinalização não deve existir.
    expect(screen.queryByTestId("msw-ready")).not.toBeInTheDocument();
  });

  it("segura os filhos até o Service Worker iniciar e então injeta o sinal msw-ready", async () => {
    vi.stubEnv("NEXT_PUBLIC_MSW_ENABLED", "true");

    let resolveStart: () => void = () => {};
    const startPromise = new Promise<void>((resolve) => {
      resolveStart = resolve;
    });
    vi.doMock("@/mocks/browser", () => ({
      worker: { start: vi.fn(() => startPromise) },
    }));

    const { MswProvider } = await import("@/components/msw-provider");

    render(
      <MswProvider>
        <div>App com MSW</div>
      </MswProvider>,
    );

    // Antes do worker.start() resolver, os filhos ficam ocultos — evita a
    // corrida com o primeiro tick de useSensorData contra a rede real.
    expect(screen.queryByText("App com MSW")).not.toBeInTheDocument();

    resolveStart();

    await waitFor(() => {
      expect(screen.getByText("App com MSW")).toBeInTheDocument();
    });
    expect(screen.getByTestId("msw-ready")).toBeInTheDocument();
  });

  it("libera os filhos mesmo se o Service Worker falhar ao iniciar", async () => {
    vi.stubEnv("NEXT_PUBLIC_MSW_ENABLED", "true");
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.doMock("@/mocks/browser", () => ({
      worker: { start: vi.fn(() => Promise.reject(new Error("SW falhou"))) },
    }));

    const { MswProvider } = await import("@/components/msw-provider");

    render(
      <MswProvider>
        <div>App após falha do worker</div>
      </MswProvider>,
    );

    await waitFor(() => {
      expect(screen.getByText("App após falha do worker")).toBeInTheDocument();
    });
    // O componente usa o mesmo estado `ready` para destravar a UI e para o
    // sinal do Playwright em ambos os casos (sucesso ou falha) — falhar ao
    // iniciar não deixa a página presa esperando por um sinal que nunca virá.
    expect(screen.getByTestId("msw-ready")).toBeInTheDocument();

    vi.restoreAllMocks();
  });
});
