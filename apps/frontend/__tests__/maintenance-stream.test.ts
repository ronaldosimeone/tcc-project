/**
 * Testes de `streamMaintenanceSuggestion` (RF-23 / RNF-47).
 *
 * Simula `fetch` devolvendo um `ReadableStream` controlado — cobre parsing
 * SSE robusto: múltiplos eventos por chunk, evento dividido entre dois
 * chunks, evento inválido, encerramento prematuro e abort/cancelamento.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  streamMaintenanceSuggestion,
  type MaintenanceStreamEvent,
} from "@/lib/maintenance-stream";

function streamFromChunks(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  let index = 0;
  return new ReadableStream({
    pull(controller) {
      if (index < chunks.length) {
        controller.enqueue(encoder.encode(chunks[index]));
        index += 1;
      } else {
        controller.close();
      }
    },
  });
}

function mockFetchWithChunks(chunks: string[], status = 200): void {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(
      new Response(streamFromChunks(chunks), {
        status,
        headers: { "Content-Type": "text/event-stream" },
      }),
    ),
  );
}

const PAYLOAD = { failure_probability: 0.9, equipment_name: "Bomba" };

async function collect(signal: AbortSignal): Promise<MaintenanceStreamEvent[]> {
  const events: MaintenanceStreamEvent[] = [];
  for await (const event of streamMaintenanceSuggestion(PAYLOAD, signal)) {
    events.push(event);
  }
  return events;
}

function isTokenEvent(
  event: MaintenanceStreamEvent,
): event is Extract<MaintenanceStreamEvent, { type: "token" }> {
  return event.type === "token";
}

describe("streamMaintenanceSuggestion", () => {
  beforeEach(() => {
    // resolveBaseUrl() (lib/api-client.ts) exige a variável — mesmo padrão
    // de connection-resilience.test.tsx / sensor-monitor.test.tsx.
    process.env.NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000";
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("concatena token1 + token2 + token3 no markdown esperado (via evento done)", async () => {
    mockFetchWithChunks([
      'event: token\ndata: {"token":"# X"}\n\n',
      'event: token\ndata: {"token":" Y"}\n\n',
      'event: token\ndata: {"token":" Z"}\n\n',
      'event: done\ndata: {"markdown":"# X Y Z","references":[]}\n\n',
    ]);

    const events = await collect(new AbortController().signal);

    const tokens = events.filter(isTokenEvent).map((e) => e.token);
    expect(tokens.join("")).toBe("# X Y Z");
    expect(events.at(-1)).toEqual({
      type: "done",
      markdown: "# X Y Z",
      references: [],
    });
  });

  it("processa múltiplos eventos presentes em um único chunk de rede", async () => {
    mockFetchWithChunks([
      'event: token\ndata: {"token":"a"}\n\nevent: token\ndata: {"token":"b"}\n\nevent: done\ndata: {"markdown":"ab","references":[]}\n\n',
    ]);

    const events = await collect(new AbortController().signal);

    expect(events).toEqual([
      { type: "token", token: "a" },
      { type: "token", token: "b" },
      { type: "done", markdown: "ab", references: [] },
    ]);
  });

  it("remonta um evento dividido no meio entre dois chunks", async () => {
    mockFetchWithChunks([
      "event: tok", // corta no meio da palavra "token"
      'en\ndata: {"tok', // corta no meio do JSON
      'en":"partido"}\n\n',
    ]);

    const events = await collect(new AbortController().signal);

    expect(events).toEqual([{ type: "token", token: "partido" }]);
  });

  it("evento com data inválido (JSON malformado) vira um evento de erro, sem lançar exceção", async () => {
    mockFetchWithChunks(["event: token\ndata: {isto nao e json valido\n\n"]);

    const events = await collect(new AbortController().signal);

    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("error");
  });

  it("ignora um tipo de evento SSE desconhecido sem quebrar o stream", async () => {
    mockFetchWithChunks([
      "event: heartbeat\ndata: {}\n\n",
      'event: token\ndata: {"token":"ok"}\n\n',
    ]);

    const events = await collect(new AbortController().signal);

    expect(events).toEqual([{ type: "token", token: "ok" }]);
  });

  it("stream encerrado prematuramente (sem evento done) apenas termina, sem lançar exceção", async () => {
    mockFetchWithChunks(['event: token\ndata: {"token":"parcial"}\n\n']);

    const events = await collect(new AbortController().signal);

    expect(events).toEqual([{ type: "token", token: "parcial" }]);
  });

  it("HTTP != 200 vira evento de erro (offline=false — o servidor respondeu)", async () => {
    mockFetchWithChunks([], 503);

    const events = await collect(new AbortController().signal);

    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({ type: "error", offline: false });
  });

  it("falha de rede (fetch rejeita) vira evento de erro com offline=true", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new TypeError("Failed to fetch")),
    );

    const events = await collect(new AbortController().signal);

    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({ type: "error", offline: true });
  });

  it("abort via AbortController encerra o generator sem emitir evento de erro", async () => {
    const controller = new AbortController();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() => {
        controller.abort();
        const abortError = new DOMException("Aborted", "AbortError");
        return Promise.reject(abortError);
      }),
    );

    const events = await collect(controller.signal);

    expect(events).toEqual([]);
  });
});
