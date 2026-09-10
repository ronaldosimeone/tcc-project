"use client";

/**
 * MaintenanceAssistant — RF-23 / RNF-47.
 *
 * Painel lateral (Sheet, mesmo padrão de `SimulationPanel`) que dispara
 * `POST /v1/maintenance/suggest/stream` e renderiza o plano de manutenção
 * incrementalmente, token a token, conforme o Llama 3.2 3B gera (via
 * `useMaintenanceStream`).
 *
 * RNF-58: decomposto em `components/maintenance-assistant/*` —
 * `SuggestionForm`, `AssistantBody` (+ `markdownComponents`/`isSafeHref`),
 * `ReferencesList`. Este arquivo é só o orquestrador (Sheet + wiring do
 * hook de streaming). Nenhuma mudança de comportamento/DOM.
 */

import { Sparkles, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { useMaintenanceStream } from "@/hooks/use-maintenance-stream";
import type { MaintenanceAssistantStatus } from "@/hooks/use-maintenance-stream";
import { cn } from "@/lib/utils";
import {
  AssistantBody,
  STATUS_LABEL,
} from "./maintenance-assistant/assistant-body";
import { SuggestionForm } from "./maintenance-assistant/suggestion-form";

interface MaintenanceAssistantProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const ACTIVE_STATUSES: readonly MaintenanceAssistantStatus[] = [
  "connecting",
  "searching",
  "generating",
];

export function MaintenanceAssistant({
  open,
  onOpenChange,
}: MaintenanceAssistantProps) {
  const { status, markdown, references, message, start, cancel } =
    useMaintenanceStream();
  const isActive = ACTIVE_STATUSES.includes(status);

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="flex w-full flex-col sm:max-w-lg">
        <SheetHeader>
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-primary" />
            <SheetTitle>Assistente de Manutenção</SheetTitle>
          </div>
          <SheetDescription>
            Gera um plano de manutenção com IA local (Llama 3.2 3B),
            fundamentado exclusivamente nos manuais técnicos recuperados via
            MCP.
          </SheetDescription>
        </SheetHeader>

        <div className="flex flex-1 flex-col gap-4 overflow-hidden px-6 py-4">
          <SuggestionForm disabled={isActive} onSubmit={start} />

          <div className="flex items-center gap-2">
            {isActive && (
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={cancel}
              >
                <X className="h-3.5 w-3.5" />
                Cancelar
              </Button>
            )}
            <Badge
              variant="outline"
              data-status={status}
              className={cn(
                "font-mono text-[10px]",
                status === "error" || status === "offline"
                  ? "border-destructive/40 text-destructive"
                  : status === "done"
                  ? "border-emerald-400/40 text-emerald-600"
                  : "",
              )}
            >
              {STATUS_LABEL[status]}
            </Badge>
          </div>

          <ScrollArea className="flex-1 rounded-md border border-slate-200 bg-white">
            <div className="p-4">
              <AssistantBody
                status={status}
                markdown={markdown}
                references={references}
                message={message}
              />
            </div>
          </ScrollArea>
        </div>
      </SheetContent>
    </Sheet>
  );
}
