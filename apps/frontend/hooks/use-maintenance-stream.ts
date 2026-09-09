"use client";

/**
 * useMaintenanceStream — RF-23 / RNF-47.
 *
 * Consome `streamMaintenanceSuggestion` (lib/maintenance-stream.ts) e expõe
 * um estado de UI simples para o `MaintenanceAssistant`. Toda a lógica de
 * parsing SSE/fetch fica na lib — este hook só traduz eventos em estado
 * React e cuida do ciclo de vida (cancelamento, cleanup no unmount).
 */

import { useCallback, useEffect, useRef, useState } from "react";

import type {
  ManualReference,
  MaintenanceSuggestionPayload,
} from "@/lib/api-client";
import { streamMaintenanceSuggestion } from "@/lib/maintenance-stream";

export type MaintenanceAssistantStatus =
  | "idle"
  | "connecting"
  | "searching"
  | "generating"
  | "done"
  | "skipped"
  | "error"
  | "offline";

export interface UseMaintenanceStreamResult {
  status: MaintenanceAssistantStatus;
  markdown: string;
  references: ManualReference[];
  message: string | null;
  /** Dispara uma nova geração — cancela a anterior, se ainda em andamento. */
  start: (payload: MaintenanceSuggestionPayload) => void;
  /** Cancela a geração em andamento (RF-23 §11 — AbortController). */
  cancel: () => void;
}

export function useMaintenanceStream(): UseMaintenanceStreamResult {
  const [status, setStatus] = useState<MaintenanceAssistantStatus>("idle");
  const [markdown, setMarkdown] = useState("");
  const [references, setReferences] = useState<ManualReference[]>([]);
  const [message, setMessage] = useState<string | null>(null);

  const controllerRef = useRef<AbortController | null>(null);
  // Evita setState após unmount (ex.: usuário fecha o painel no meio da
  // geração) — sem isso, React acusaria update em componente desmontado.
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      controllerRef.current?.abort();
    };
  }, []);

  const cancel = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    if (mountedRef.current) setStatus("idle");
  }, []);

  const start = useCallback((payload: MaintenanceSuggestionPayload) => {
    // Cancela qualquer geração anterior ainda em andamento antes de iniciar
    // uma nova — nunca duas conexões simultâneas.
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;

    setStatus("connecting");
    setMarkdown("");
    setReferences([]);
    setMessage(null);

    void (async () => {
      try {
        for await (const event of streamMaintenanceSuggestion(
          payload,
          controller.signal,
        )) {
          if (!mountedRef.current || controller.signal.aborted) return;

          switch (event.type) {
            case "searching":
              setStatus("searching");
              break;
            case "token":
              setStatus("generating");
              setMarkdown((prev) => prev + event.token);
              break;
            case "done":
              setMarkdown(event.markdown);
              setReferences(event.references);
              setStatus("done");
              break;
            case "skipped":
              setMessage(event.message);
              setStatus("skipped");
              break;
            case "error":
              setMessage(event.message);
              setStatus(event.offline ? "offline" : "error");
              break;
          }
        }
      } catch {
        // Rede de segurança — a lib já trata os próprios erros de rede como
        // eventos "error"/offline; isto só cobre uma falha verdadeiramente
        // inesperada no próprio consumo do generator.
        if (mountedRef.current && !controller.signal.aborted) {
          setStatus("offline");
          setMessage("Não foi possível concluir a geração da sugestão.");
        }
      }
    })();
  }, []);

  return { status, markdown, references, message, start, cancel };
}
