"use client";

/**
 * Modelo ativo e seu limiar calibrado (GET /models), para telas que explicam
 * um nível de risco em termos do modelo — ex.: Configurações de Alertas.
 * Falha silenciosa: sem a informação, a tela continua funcional.
 */

import { useEffect, useState } from "react";

import { listModels } from "@/lib/api-client";
import { formatModelName } from "@/lib/model-name";
import { publishActiveThreshold } from "@/hooks/use-active-threshold";

export interface ActiveModelInfo {
  name: string;
  label: string;
  threshold: number;
}

export const useActiveModelInfo = (): ActiveModelInfo | null => {
  const [info, setInfo] = useState<ActiveModelInfo | null>(null);
  useEffect(() => {
    let alive = true;
    void (async () => {
      try {
        const body = await listModels();
        const thr = body.active_threshold;
        if (!alive || !body.active_model || typeof thr !== "number") return;
        publishActiveThreshold(thr);
        setInfo({
          name: body.active_model,
          label: formatModelName(body.active_model),
          threshold: thr,
        });
      } catch {
        // sem modelo ativo / endpoint indisponível: mantém a explicação genérica
      }
    })();
    return () => {
      alive = false;
    };
  }, []);
  return info;
};
