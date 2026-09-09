/**
 * Cliente SSE de POST /v1/maintenance/suggest/stream (RF-23 / RNF-47).
 *
 * Canal INDEPENDENTE do SSE de sensores (`useSSE`, `/api/stream/sensors`,
 * RF-15) — não reaproveita esse hook. `EventSource` nativo não suporta POST
 * com corpo JSON (o payload aqui carrega failure_probability + equipamento/
 * sintoma), então este módulo consome o stream via `fetch()` +
 * `ReadableStream` com um parser SSE próprio, escrito à mão — sem lib
 * externa (nenhuma nova dependência SSE-over-fetch foi necessária).
 *
 * Protocolo (espelha `apps/backend/src/routers/maintenance.py`):
 *   event: searching\ndata: {}\n\n
 *   event: token\ndata: {"token": "..."}\n\n
 *   event: done\ndata: {"markdown": "...", "references": [...]}\n\n
 *   event: skipped\ndata: {"message": "..."}\n\n
 *   event: error\ndata: {"message": "..."}\n\n
 */

import {
  resolveBaseUrl,
  type ManualReference,
  type MaintenanceSuggestionPayload,
} from "@/lib/api-client";

export type MaintenanceStreamEvent =
  | { type: "searching" }
  | { type: "token"; token: string }
  | { type: "done"; markdown: string; references: ManualReference[] }
  | { type: "skipped"; message: string }
  // offline=true → fetch nunca chegou a conectar (rede/servidor fora do ar).
  // offline=false → conectamos, mas o backend/MCP/Ollama reportou uma falha.
  | { type: "error"; message: string; offline: boolean };

// ── Parsing de um bloco SSE (`event: ...\ndata: ...`) ────────────────────────

function isManualReference(value: unknown): value is ManualReference {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.file_name === "string" &&
    typeof v.page === "number" &&
    typeof v.chunk_index === "number" &&
    typeof v.source === "string" &&
    typeof v.score === "number"
  );
}

function parseSseBlock(block: string): MaintenanceStreamEvent | null {
  let eventType = "";
  let rawData = "";
  for (const line of block.split("\n")) {
    if (line.startsWith("event: ")) {
      eventType = line.slice("event: ".length).trim();
    } else if (line.startsWith("data: ")) {
      rawData += line.slice("data: ".length);
    }
  }
  if (!eventType) return null;

  let parsed: unknown;
  try {
    parsed = rawData ? JSON.parse(rawData) : {};
  } catch {
    // Evento SSE malformado (data não é JSON válido) — não derruba o stream
    // inteiro, mas o consumidor precisa saber que algo deu errado.
    return {
      type: "error",
      message: "Evento SSE malformado recebido.",
      offline: false,
    };
  }
  const data = (
    typeof parsed === "object" && parsed !== null ? parsed : {}
  ) as Record<string, unknown>;

  switch (eventType) {
    case "searching":
      return { type: "searching" };
    case "token":
      return {
        type: "token",
        token: typeof data.token === "string" ? data.token : "",
      };
    case "done": {
      const references = Array.isArray(data.references)
        ? data.references.filter(isManualReference)
        : [];
      return {
        type: "done",
        markdown: typeof data.markdown === "string" ? data.markdown : "",
        references,
      };
    }
    case "skipped":
      return {
        type: "skipped",
        message:
          typeof data.message === "string"
            ? data.message
            : "Sugestão não acionada.",
      };
    case "error":
      return {
        type: "error",
        message:
          typeof data.message === "string"
            ? data.message
            : "Erro desconhecido.",
        offline: false,
      };
    default:
      // Tipo de evento desconhecido — ignora silenciosamente (forward
      // compatível), não interrompe o stream por causa de um evento extra.
      return null;
  }
}

// ── Consumo do stream ─────────────────────────────────────────────────────────

/**
 * Gerador assíncrono de eventos do stream de sugestão de manutenção.
 * Aborte via `signal` (AbortController) para cancelar — fecha o `fetch` e
 * o `reader` corretamente (RF-23 §11).
 */
export async function* streamMaintenanceSuggestion(
  payload: MaintenanceSuggestionPayload,
  signal: AbortSignal,
): AsyncGenerator<MaintenanceStreamEvent> {
  const baseUrl = resolveBaseUrl();

  let response: Response;
  try {
    response = await fetch(`${baseUrl}/v1/maintenance/suggest/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
      cache: "no-store",
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") return;
    yield {
      type: "error",
      message: "Não foi possível conectar ao servidor.",
      offline: true,
    };
    return;
  }

  if (!response.ok || !response.body) {
    yield {
      type: "error",
      message: `O servidor respondeu com um erro (HTTP ${response.status}).`,
      offline: false,
    };
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Um chunk de rede pode conter 0, 1 ou vários eventos completos —
      // e um evento pode ficar dividido entre dois chunks. O `buffer`
      // acumula até `\n\n` aparecer; o que sobra fica pra próxima iteração.
      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const rawBlock = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const event = parseSseBlock(rawBlock);
        if (event) yield event;
        boundary = buffer.indexOf("\n\n");
      }
    }
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") return;
    yield {
      type: "error",
      message: "A conexão de streaming foi interrompida antes de terminar.",
      offline: true,
    };
  } finally {
    reader.releaseLock();
  }
}
