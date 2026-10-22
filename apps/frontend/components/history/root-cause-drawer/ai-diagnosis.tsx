// ── Plano de manutenção gerado (RAG via MCP) + relatório de fundamentação ────

import ReactMarkdown from "react-markdown";
import { ShieldAlert, Sparkles } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { markdownComponents } from "@/components/maintenance-assistant/markdown-components";
import type { RootCauseDto } from "@/lib/history-api";

const GROUNDING_LABEL: Record<string, string> = {
  grounded: "Fundamentado nos manuais",
  partially_grounded: "Parcialmente fundamentado",
  ungrounded: "Sem trecho de manual suficiente",
};

interface AiDiagnosisProps {
  rootCause: RootCauseDto | null;
}

export function AiDiagnosis({ rootCause }: AiDiagnosisProps) {
  const grounding = rootCause?.grounding ?? null;
  return (
    <section
      className="rounded-lg border border-border border-l-4 border-l-primary bg-card p-4"
      aria-label="Plano sugerido"
    >
      <div className="flex items-start gap-3">
        <span className="mt-0.5 shrink-0 rounded-md bg-primary/10 p-1.5 text-primary">
          <Sparkles className="h-3.5 w-3.5" aria-hidden="true" />
        </span>
        <div className="min-w-0 flex-1 space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-xs font-bold uppercase tracking-wider text-primary">
              Plano sugerido (RAG)
            </p>
            {grounding && (
              <Badge variant="outline" className="text-xs">
                {GROUNDING_LABEL[grounding.status] ?? grounding.status}
              </Badge>
            )}
            {rootCause?.llm_model && (
              <span className="font-mono text-xs text-muted-foreground">
                {rootCause.llm_model}
              </span>
            )}
          </div>

          {rootCause?.suggestion_markdown ? (
            <div className="prose-sm text-sm text-foreground">
              <ReactMarkdown components={markdownComponents}>
                {rootCause.suggestion_markdown}
              </ReactMarkdown>
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">
              Plano ainda não gerado (o assistente roda em segundo plano após a
              abertura do evento, ou estava indisponível).
            </p>
          )}

          {grounding &&
            (grounding.invalid_citations.length > 0 ||
              grounding.safety_violations.length > 0) && (
              <p className="flex items-start gap-1.5 text-xs text-destructive">
                <ShieldAlert
                  className="mt-0.5 h-3 w-3 shrink-0"
                  aria-hidden="true"
                />
                {grounding.safety_violations.length > 0
                  ? "Plano bloqueado: continha instrução para desativar dispositivo de segurança."
                  : `Citação a página não recuperada: ${grounding.invalid_citations
                      .map((c) => `${c.file_name} p.${c.page}`)
                      .join(", ")}`}
              </p>
            )}
        </div>
      </div>
    </section>
  );
}
