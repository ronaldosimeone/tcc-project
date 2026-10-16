"use client";

// ── Corpo do painel — Markdown incremental + estados — RNF-58: extraído de
// maintenance-assistant.tsx ─────────────────────────────────────────────────

import ReactMarkdown from "react-markdown";
import { AlertTriangle, Loader2, Search, WifiOff } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import type { MaintenanceAssistantStatus } from "@/hooks/use-maintenance-stream";
import type {
  ManualReference,
  MaintenanceSuggestionPayload,
} from "@/lib/api-client";
import type { GroundingStatus } from "@/lib/suggestion-feedback-api";
import { markdownComponents } from "./markdown-components";
import { ReferencesList } from "./references-list";
import { SuggestionFeedback } from "./suggestion-feedback";

export const STATUS_LABEL: Record<MaintenanceAssistantStatus, string> = {
  idle: "Pronto",
  connecting: "Conectando…",
  searching: "Buscando manual…",
  generating: "Gerando…",
  done: "Concluído",
  skipped: "Não acionada",
  error: "Erro",
  offline: "Offline",
};

export interface AssistantBodyProps {
  status: MaintenanceAssistantStatus;
  markdown: string;
  references: ManualReference[];
  message: string | null;
  /** RF-38 — quando presente, o plano concluído mostra o voto do técnico. */
  request?: MaintenanceSuggestionPayload | null;
  groundingStatus?: GroundingStatus | null;
}

interface DoneFooterProps {
  markdown: string;
  references: ManualReference[];
  request: MaintenanceSuggestionPayload | null;
  groundingStatus: GroundingStatus | null;
}

const DoneFooter = ({
  markdown,
  references,
  request,
  groundingStatus,
}: DoneFooterProps) => (
  <>
    <div className="flex flex-col gap-2 border-t border-border pt-3">
      <span className="text-xs font-medium text-muted-foreground">
        Manuais consultados
      </span>
      <ReferencesList references={references} />
    </div>
    {request && (
      // `key`: um novo plano zera os votos do anterior.
      <SuggestionFeedback
        key={markdown}
        request={request}
        references={references}
        groundingStatus={groundingStatus}
      />
    )}
  </>
);

export function AssistantBody({
  status,
  markdown,
  references,
  message,
  request = null,
  groundingStatus = null,
}: AssistantBodyProps) {
  if (status === "idle") {
    return (
      <p className="text-sm text-muted-foreground">
        Preencha os campos acima e clique em &ldquo;Gerar sugestão&rdquo; para
        consultar os manuais técnicos e gerar um plano de manutenção.
      </p>
    );
  }

  if (status === "skipped") {
    return (
      <div className="flex items-start gap-2 rounded-md border border-border bg-muted/50 p-3 text-sm text-foreground/80">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground/70" />
        <span>{message ?? "Sugestão automática não acionada."}</span>
      </div>
    );
  }

  if ((status === "error" || status === "offline") && !markdown) {
    return (
      <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
        {status === "offline" ? (
          <WifiOff className="mt-0.5 h-4 w-4 shrink-0" />
        ) : (
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
        )}
        <span>
          {message ?? "Não foi possível gerar a sugestão. Tente novamente."}
        </span>
      </div>
    );
  }

  if ((status === "connecting" || status === "searching") && !markdown) {
    return (
      <div className="flex flex-col gap-3">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Search className="h-3.5 w-3.5 animate-pulse" />
          {STATUS_LABEL[status]}
        </div>
        <Skeleton className="h-4 w-3/4" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-5/6" />
      </div>
    );
  }

  // generating (já com tokens) | done | error-com-tokens-parciais
  return (
    <div className="flex flex-col gap-4">
      {status === "generating" && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Loader2 className="h-3 w-3 animate-spin" />
          Gerando plano…
        </div>
      )}
      {(status === "error" || status === "offline") && markdown && (
        <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 p-2 text-xs text-destructive">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>
            {message ?? "A geração foi interrompida."} O texto abaixo é parcial.
          </span>
        </div>
      )}

      <div className="max-w-none break-words">
        <ReactMarkdown components={markdownComponents}>
          {markdown}
        </ReactMarkdown>
      </div>

      {status === "done" && (
        <DoneFooter
          markdown={markdown}
          references={references}
          request={request}
          groundingStatus={groundingStatus}
        />
      )}
    </div>
  );
}
