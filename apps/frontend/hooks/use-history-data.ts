"use client";

/**
 * Dados reais do Histórico: eventos (episódios RN-03) + KPIs do período.
 *
 * O estado de carregamento é DERIVADO: cada requisição tem uma chave
 * (`período:nonce`) e o resultado guarda a chave a que pertence — enquanto
 * não chega o resultado da chave atual, o hook está em "loading". Assim o
 * efeito só chama setState em callbacks assíncronos.
 */

import { useCallback, useEffect, useState } from "react";

import {
  fetchAssets,
  fetchEventsInWindow,
  fetchKpis,
  periodWindow,
  toHistoryEvent,
  type KpiReportDto,
} from "@/lib/history-api";
import type { HistoryEvent } from "@/lib/history-types";

type LoadState = "loading" | "ready" | "error";

interface Result {
  key: string;
  assets: string[];
  events: HistoryEvent[];
  kpis: KpiReportDto | null;
  error: string | null;
}

interface UseHistoryDataReturn {
  /** Ids da frota cadastrada (GET /v1/assets). */
  assets: string[];
  events: HistoryEvent[];
  kpis: KpiReportDto | null;
  state: LoadState;
  error: string | null;
  reload: () => void;
}

export const useHistoryData = (period: string): UseHistoryDataReturn => {
  const [nonce, setNonce] = useState(0);
  const [result, setResult] = useState<Result | null>(null);
  const key = `${period}:${nonce}`;

  useEffect(() => {
    let cancelled = false;
    const window = periodWindow(period);
    Promise.all([fetchEventsInWindow(window), fetchKpis(window), fetchAssets()])
      .then(([dtos, report, assets]) => {
        if (!cancelled) {
          setResult({
            key,
            assets: assets.map((a) => a.id),
            events: dtos.map(toHistoryEvent),
            kpis: report,
            error: null,
          });
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setResult({
            key,
            assets: [],
            events: [],
            kpis: null,
            error:
              err instanceof Error
                ? err.message
                : "Falha ao carregar o histórico.",
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [key, period]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  const current = result?.key === key ? result : null;
  const state: LoadState =
    current === null ? "loading" : current.error ? "error" : "ready";

  return {
    assets: current?.assets ?? [],
    events: current?.events ?? [],
    kpis: current?.kpis ?? null,
    state,
    error: current?.error ?? null,
    reload,
  };
};
