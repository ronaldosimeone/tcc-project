/**
 * RF-38 — feedback do técnico sobre um plano gerado pelo assistente.
 *
 * `POST /v1/maintenance/feedback`: o plano foi útil? e, por trecho de manual
 * citado, ele era relevante? Os votos reordenam a recuperação do mesmo
 * equipamento (prior de Laplace no backend) e nunca alteram o limiar RN-04.
 */

import { apiFetch } from "@/lib/auth";
import { resolveBaseUrl } from "@/lib/api-client";
import type { ManualReference } from "@/lib/api-client";

export type GroundingStatus =
  | "pending"
  | "grounded"
  | "partially_grounded"
  | "ungrounded";

export interface ReferenceVote {
  file_name: string;
  page: number;
  helpful: boolean;
}

export interface SuggestionFeedbackPayload {
  equipment_name: string;
  symptom_description?: string;
  grounding_status?: GroundingStatus;
  useful: boolean;
  reference_votes: ReferenceVote[];
  comment?: string;
}

export interface SuggestionFeedbackResponse {
  id: number;
  created_at: string;
}

/** Chave estável de um trecho para voto — o backend agrega por arquivo+página. */
export const referenceKey = (
  ref: Pick<ManualReference, "file_name" | "page">,
): string => `${ref.file_name}#${ref.page}`;

/** Um voto por arquivo+página, na ordem em que aparecem. */
export const uniqueVoteTargets = (
  references: ManualReference[],
): ManualReference[] => {
  const seen = new Set<string>();
  return references.filter((ref) => {
    const key = referenceKey(ref);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
};

export const submitSuggestionFeedback = async (
  payload: SuggestionFeedbackPayload,
): Promise<SuggestionFeedbackResponse> => {
  const response = await apiFetch(
    `${resolveBaseUrl()}/v1/maintenance/feedback`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      cache: "no-store",
    },
  );
  if (!response.ok) {
    if (response.status === 401 || response.status === 403) {
      throw new Error("Entre como técnico para registrar o feedback.");
    }
    throw new Error(`Falha ao enviar o feedback (HTTP ${response.status}).`);
  }
  return response.json() as Promise<SuggestionFeedbackResponse>;
};
