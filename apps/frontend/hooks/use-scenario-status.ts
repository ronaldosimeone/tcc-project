"use client";

/**
 * Acompanha o cenário de falha real em reprodução (fase e progresso),
 * consultando GET /simulator/scenario a cada `intervalMs` enquanto houver
 * um cenário ativo.
 */

import { useCallback, useEffect, useState } from "react";

import {
  fetchScenarioStatus,
  type ScenarioStatus,
} from "@/lib/failure-scenarios-api";

interface UseScenarioStatusReturn {
  status: ScenarioStatus | null;
  refresh: () => void;
}

export const useScenarioStatus = (
  intervalMs = 2000,
): UseScenarioStatusReturn => {
  const [status, setStatus] = useState<ScenarioStatus | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let alive = true;
    const tick = async (): Promise<void> => {
      try {
        const fresh = await fetchScenarioStatus();
        if (alive) setStatus(fresh);
      } catch {
        // Painel continua utilizável sem o progresso.
      }
    };
    void tick();
    const id = window.setInterval(() => void tick(), intervalMs);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [intervalMs, nonce]);

  const refresh = useCallback(() => setNonce((n) => n + 1), []);
  return { status, refresh };
};
