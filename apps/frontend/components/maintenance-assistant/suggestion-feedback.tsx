"use client";

// ── RF-38 — voto do técnico por trecho citado + utilidade do plano ──────────
// Aparece só quando o plano termina (status "done"). Um voto por
// arquivo+página; o técnico pode deixar trechos sem voto.

import { memo, useState } from "react";
import { CheckCircle2, ThumbsDown, ThumbsUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import type {
  ManualReference,
  MaintenanceSuggestionPayload,
} from "@/lib/api-client";
import {
  referenceKey,
  submitSuggestionFeedback,
  uniqueVoteTargets,
  type GroundingStatus,
} from "@/lib/suggestion-feedback-api";

type SendState = "idle" | "sending" | "sent" | "error";

interface VoteToggleProps {
  label: string;
  value: boolean | undefined;
  onChange: (value: boolean | undefined) => void;
  disabled: boolean;
}

const VoteToggle = ({ label, value, onChange, disabled }: VoteToggleProps) => (
  <div className="flex items-center gap-1" role="group" aria-label={label}>
    <Button
      type="button"
      size="icon"
      variant={value === true ? "default" : "outline"}
      className="h-7 w-7"
      aria-pressed={value === true}
      aria-label={`${label}: relevante`}
      disabled={disabled}
      onClick={() => onChange(value === true ? undefined : true)}
    >
      <ThumbsUp className="h-3.5 w-3.5" />
    </Button>
    <Button
      type="button"
      size="icon"
      variant={value === false ? "destructive" : "outline"}
      className="h-7 w-7"
      aria-pressed={value === false}
      aria-label={`${label}: não relevante`}
      disabled={disabled}
      onClick={() => onChange(value === false ? undefined : false)}
    >
      <ThumbsDown className="h-3.5 w-3.5" />
    </Button>
  </div>
);

export interface SuggestionFeedbackProps {
  request: MaintenanceSuggestionPayload;
  references: ManualReference[];
  groundingStatus: GroundingStatus | null;
}

const SuggestionFeedbackComponent = ({
  request,
  references,
  groundingStatus,
}: SuggestionFeedbackProps) => {
  const targets = uniqueVoteTargets(references);
  const [votes, setVotes] = useState<Record<string, boolean>>({});
  const [useful, setUseful] = useState<boolean | undefined>(undefined);
  const [comment, setComment] = useState("");
  const [state, setState] = useState<SendState>("idle");
  const [error, setError] = useState<string | null>(null);

  const locked = state === "sending" || state === "sent";

  const setVote = (key: string, value: boolean | undefined): void => {
    setVotes((prev) => {
      const next = { ...prev };
      if (value === undefined) delete next[key];
      else next[key] = value;
      return next;
    });
  };

  const send = async (): Promise<void> => {
    if (useful === undefined) return;
    setState("sending");
    setError(null);
    try {
      await submitSuggestionFeedback({
        equipment_name: request.equipment_name,
        symptom_description: request.symptom_description || undefined,
        grounding_status: groundingStatus ?? undefined,
        useful,
        reference_votes: targets
          .filter((ref) => referenceKey(ref) in votes)
          .map((ref) => ({
            file_name: ref.file_name,
            page: ref.page,
            helpful: votes[referenceKey(ref)],
          })),
        comment: comment.trim() || undefined,
      });
      setState("sent");
    } catch (err: unknown) {
      setError(
        err instanceof Error ? err.message : "Falha ao enviar o feedback.",
      );
      setState("error");
    }
  };

  if (state === "sent") {
    return (
      <p
        data-testid="suggestion-feedback-sent"
        className="flex items-center gap-2 text-xs text-muted-foreground"
      >
        <CheckCircle2 className="h-3.5 w-3.5 text-primary" />
        Feedback registrado. Os votos ajustam a ordem dos trechos nas próximas
        consultas deste equipamento.
      </p>
    );
  }

  return (
    <section
      data-testid="suggestion-feedback"
      aria-label="Avaliação do plano"
      className="flex flex-col gap-3 border-t border-border pt-3"
    >
      {targets.length > 0 && (
        <div className="flex flex-col gap-2">
          <span className="text-xs font-medium text-muted-foreground">
            Cada trecho ajudou no diagnóstico?
          </span>
          <ul className="flex flex-col gap-1.5">
            {targets.map((ref) => {
              const key = referenceKey(ref);
              const label = `${ref.file_name}, página ${ref.page}`;
              return (
                <li
                  key={key}
                  className="flex items-center justify-between gap-2 text-xs"
                >
                  <span
                    className="truncate font-mono text-muted-foreground"
                    title={label}
                  >
                    {ref.file_name} — p. {ref.page}
                  </span>
                  <VoteToggle
                    label={label}
                    value={votes[key]}
                    onChange={(v) => setVote(key, v)}
                    disabled={locked}
                  />
                </li>
              );
            })}
          </ul>
        </div>
      )}

      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-muted-foreground">
          O plano foi útil?
        </span>
        <VoteToggle
          label="Plano"
          value={useful}
          onChange={setUseful}
          disabled={locked}
        />
      </div>

      <Textarea
        aria-label="Comentário opcional"
        placeholder="Comentário opcional (ex.: faltou o intervalo de troca)"
        maxLength={1000}
        value={comment}
        disabled={locked}
        onChange={(e) => setComment(e.target.value)}
        className="min-h-16 text-xs"
      />

      {error && (
        <p role="alert" className="text-xs text-destructive">
          {error}
        </p>
      )}

      <Button
        type="button"
        size="sm"
        disabled={useful === undefined || locked}
        onClick={() => void send()}
      >
        {state === "sending" ? "Enviando…" : "Enviar avaliação"}
      </Button>
    </section>
  );
};

export const SuggestionFeedback = memo(SuggestionFeedbackComponent);
