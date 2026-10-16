"use client";

/**
 * Detalhe de um evento (causa raiz congelada, OS e leituras reais) +
 * ações da OS. `eventId = null` não busca nada (drawer fechado).
 *
 * Carregamento derivado: o resultado guarda (eventId, versão); enquanto não
 * corresponde ao pedido atual, `loading = true`. Ações da OS incrementam a
 * versão e o detalhe é recarregado.
 */

import { useCallback, useEffect, useState } from "react";

import {
  createWorkOrder,
  fetchEventDetail,
  updateWorkOrder,
  type FailureEventDetailDto,
  type WorkOrderUpdatePayload,
} from "@/lib/history-api";

interface Result {
  key: string;
  detail: FailureEventDetailDto | null;
  error: string | null;
}

interface UseEventDetailReturn {
  detail: FailureEventDetailDto | null;
  loading: boolean;
  error: string | null;
  actionError: string | null;
  busy: boolean;
  openWorkOrder: () => Promise<void>;
  changeWorkOrder: (payload: WorkOrderUpdatePayload) => Promise<void>;
}

export const useEventDetail = (
  eventId: number | null,
  onChanged?: () => void,
): UseEventDetailReturn => {
  const [version, setVersion] = useState(0);
  const [result, setResult] = useState<Result | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const key = `${eventId ?? "none"}:${version}`;

  useEffect(() => {
    if (eventId === null) return;
    let cancelled = false;
    fetchEventDetail(eventId)
      .then((detail) => {
        if (!cancelled) setResult({ key, detail, error: null });
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setResult({
            key,
            detail: null,
            error:
              err instanceof Error
                ? err.message
                : "Falha ao carregar o evento.",
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [eventId, key]);

  // Mantém o detalhe anterior do MESMO evento visível durante o refetch pós-ação.
  const sameEvent =
    result !== null && result.key.split(":")[0] === String(eventId);
  const current = sameEvent ? result : null;
  const detail = current?.detail ?? null;

  const run = useCallback(
    async (action: () => Promise<unknown>) => {
      if (eventId === null) return;
      setBusy(true);
      setActionError(null);
      try {
        await action();
        setVersion((v) => v + 1);
        onChanged?.();
      } catch (err: unknown) {
        setActionError(
          err instanceof Error ? err.message : "Ação não concluída.",
        );
      } finally {
        setBusy(false);
      }
    },
    [eventId, onChanged],
  );

  const openWorkOrder = useCallback(
    () =>
      run(() =>
        eventId === null ? Promise.resolve() : createWorkOrder(eventId),
      ),
    [eventId, run],
  );

  const changeWorkOrder = useCallback(
    (payload: WorkOrderUpdatePayload) =>
      run(() => {
        const id = detail?.work_order?.id;
        return id === undefined
          ? Promise.resolve()
          : updateWorkOrder(id, payload);
      }),
    [detail, run],
  );

  return {
    detail,
    loading: eventId !== null && result?.key !== key,
    error: current?.error ?? null,
    actionError: sameEvent ? actionError : null,
    busy,
    openWorkOrder,
    changeWorkOrder,
  };
};
