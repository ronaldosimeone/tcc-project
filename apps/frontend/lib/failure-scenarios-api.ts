/**
 * Cenários de falha REAIS do simulador (GET/PUT /simulator/scenario[s]).
 *
 * `seen_in_training` é o selo de honestidade da apresentação:
 * - true  → os modelos servidos treinaram com essa falha: é DEMONSTRAÇÃO;
 * - false → falha posterior a todo o treino (MetroPT2): é TESTE HONESTO.
 */

import { apiFetch } from "@/lib/auth";
import { resolveBaseUrl } from "@/lib/api-client";

export interface FailureScenario {
  id: string;
  label: string;
  failure_type: string;
  dataset: string;
  start: string;
  end: string;
  seen_in_training: boolean;
  available: boolean;
  note: string;
}

export interface ScenarioStatus {
  mode: string;
  scenario: FailureScenario | null;
  step: number | null;
  total: number | null;
  phase: "pre_failure" | "failure" | null;
}

const getJson = async <T>(path: string, init?: RequestInit): Promise<T> => {
  const response = await apiFetch(`${resolveBaseUrl()}${path}`, {
    cache: "no-store",
    ...init,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail =
      body && typeof body === "object" && "detail" in body
        ? String(body.detail)
        : `HTTP ${response.status}`;
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
};

export const fetchFailureScenarios = (): Promise<FailureScenario[]> =>
  getJson<FailureScenario[]>("/simulator/scenarios");

export const fetchScenarioStatus = (): Promise<ScenarioStatus> =>
  getJson<ScenarioStatus>("/simulator/scenario");

export const playFailureScenario = ({
  id,
  leadMinutes,
}: {
  id: string | null;
  leadMinutes: number;
}): Promise<ScenarioStatus> =>
  getJson<ScenarioStatus>("/simulator/scenario", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, lead_minutes: leadMinutes }),
  });
