"use client";

/**
 * useErrorRateStatus — RNF-77.
 *
 * Poll simples de `GET /observability/error-rate` a cada 15s — mesmo padrão
 * de `useSensorData`'s prediction poll (hooks/use-sensor-data.ts,
 * `POLL_INTERVAL_MS`), intervalo maior porque a janela do PromQL já é de 5
 * minutos (src/services/observability_service.py): consultar a cada 5s não
 * traria dado mais fresco, só carga extra no backend/Prometheus (RNF-77
 * Fase 9 — "solução deve ser... de baixa carga").
 *
 * Erros de rede são silenciosos (mesmo padrão do prediction poll) — o
 * backend já degrada para NORMAL/prometheus_reachable=false internamente
 * (ver observability_service.py); se a PRÓPRIA chamada a este endpoint
 * falhar (API fora do ar), o hook mantém o último status conhecido em vez
 * de travar a UI.
 */

import { useEffect, useState } from "react";
import { getErrorRateStatus, type ErrorRateStatus } from "@/lib/api-client";

const POLL_INTERVAL_MS = 15_000;

export function useErrorRateStatus(): ErrorRateStatus | null {
  const [status, setStatus] = useState<ErrorRateStatus | null>(null);

  useEffect(() => {
    let alive = true;

    const poll = async (): Promise<void> => {
      try {
        const result = await getErrorRateStatus();
        if (alive) setStatus(result.status);
      } catch {
        // silencioso — mantém o último status conhecido (mesma política do
        // prediction poll em use-sensor-data.ts)
      }
    };

    void poll();
    const id = setInterval(() => void poll(), POLL_INTERVAL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  return status;
}
