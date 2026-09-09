"use client";

/**
 * MaintenanceAssistant — RF-23 / RNF-47.
 *
 * Painel lateral (Sheet, mesmo padrão de `SimulationPanel`) que dispara
 * `POST /v1/maintenance/suggest/stream` e renderiza o plano de manutenção
 * incrementalmente, token a token, conforme o Llama 3.2 3B gera (via
 * `useMaintenanceStream`). Markdown renderizado com `react-markdown` — sem
 * HTML arbitrário (`rehype-raw` não é usado) e com links sanitizados
 * (`isSafeHref`, RF-23 §7/§8).
 *
 * Estados visuais: idle · connecting · searching · generating · done ·
 * skipped · error · offline — cada um com feedback claro, nunca preso
 * indefinidamente em "Gerando…".
 */

import { type FormEvent, useId, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import {
  AlertTriangle,
  FileText,
  Loader2,
  Search,
  Sparkles,
  WifiOff,
  X,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import {
  useMaintenanceStream,
  type MaintenanceAssistantStatus,
} from "@/hooks/use-maintenance-stream";
import type { ManualReference } from "@/lib/api-client";
import { cn } from "@/lib/utils";

interface MaintenanceAssistantProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const STATUS_LABEL: Record<MaintenanceAssistantStatus, string> = {
  idle: "Pronto",
  connecting: "Conectando…",
  searching: "Buscando manual…",
  generating: "Gerando…",
  done: "Concluído",
  skipped: "Não acionada",
  error: "Erro",
  offline: "Offline",
};

const ACTIVE_STATUSES: readonly MaintenanceAssistantStatus[] = [
  "connecting",
  "searching",
  "generating",
];

// ── Markdown seguro ───────────────────────────────────────────────────────────
//
// react-markdown, por padrão, NUNCA interpreta HTML bruto do texto de origem
// (nenhum `rehype-raw` é usado aqui) — um `<script>...</script>` no Markdown
// vira texto literal, não um elemento executado. O único ponto de risco real
// é o `href` de um link Markdown (`[texto](url)`) virar um `<a>` de verdade.
// `isSafeHref` bloqueia `javascript:`, `data:` e qualquer esquema que não
// seja http/https antes de renderizar como link clicável.
function isSafeHref(href: string): boolean {
  try {
    const url = new URL(href, "http://localhost");
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}

// Sem plugin de tipografia (`@tailwindcss/typography` não é usado em
// nenhum outro lugar do projeto) — classes explícitas por elemento,
// reaproveitando só os tokens de cor/tema já existentes (`text-foreground`,
// `text-muted-foreground`), em vez de puxar uma dependência nova só para
// este painel.
const markdownComponents: Components = {
  h1: ({ children, ...props }) => (
    <h1 className="mb-2 text-lg font-bold text-foreground" {...props}>
      {children}
    </h1>
  ),
  h2: ({ children, ...props }) => (
    <h2
      className="mt-4 mb-1.5 text-base font-semibold text-foreground"
      {...props}
    >
      {children}
    </h2>
  ),
  h3: ({ children, ...props }) => (
    <h3 className="mt-3 mb-1 text-sm font-semibold text-foreground" {...props}>
      {children}
    </h3>
  ),
  p: ({ children, ...props }) => (
    <p className="mb-2 text-sm leading-relaxed text-foreground" {...props}>
      {children}
    </p>
  ),
  ul: ({ children, ...props }) => (
    <ul
      className="mb-2 list-disc space-y-1 pl-5 text-sm text-foreground"
      {...props}
    >
      {children}
    </ul>
  ),
  ol: ({ children, ...props }) => (
    <ol
      className="mb-2 list-decimal space-y-1 pl-5 text-sm text-foreground"
      {...props}
    >
      {children}
    </ol>
  ),
  code: ({ children, ...props }) => (
    <code
      className="rounded bg-slate-100 px-1 py-0.5 font-mono text-xs"
      {...props}
    >
      {children}
    </code>
  ),
  a: ({ href, children, ...props }) => {
    if (!href || !isSafeHref(href)) {
      // Link inseguro/sem URL — nunca renderiza como <a>, mostra só o texto.
      return <span {...props}>{children}</span>;
    }
    return (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="text-primary underline underline-offset-2"
        {...props}
      >
        {children}
      </a>
    );
  },
};

// ── Formulário de disparo ──────────────────────────────────────────────────────

interface SuggestionFormProps {
  disabled: boolean;
  onSubmit: (input: {
    failure_probability: number;
    equipment_name: string;
    symptom_description?: string;
  }) => void;
}

function SuggestionForm({ disabled, onSubmit }: SuggestionFormProps) {
  const probabilityId = useId();
  const [equipmentName, setEquipmentName] = useState(
    "Compressor de ar industrial",
  );
  const [symptomDescription, setSymptomDescription] = useState("");
  const [failureProbability, setFailureProbability] = useState("0.85");

  const handleSubmit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    const probability = Number(failureProbability);
    if (Number.isNaN(probability)) return;
    onSubmit({
      failure_probability: probability,
      equipment_name: equipmentName.trim() || "Equipamento",
      symptom_description: symptomDescription.trim() || undefined,
    });
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-3">
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="maintenance-equipment-name">Equipamento</Label>
        <Input
          id="maintenance-equipment-name"
          value={equipmentName}
          onChange={(e) => setEquipmentName(e.target.value)}
          disabled={disabled}
          maxLength={200}
        />
      </div>

      <div className="flex flex-col gap-1.5">
        <Label htmlFor="maintenance-symptom">Sintoma observado</Label>
        <Textarea
          id="maintenance-symptom"
          value={symptomDescription}
          onChange={(e) => setSymptomDescription(e.target.value)}
          disabled={disabled}
          maxLength={500}
          placeholder="ex.: vazamento de óleo, ruído excessivo na sucção…"
        />
      </div>

      <div className="flex flex-col gap-1.5">
        <Label htmlFor={probabilityId}>Probabilidade de falha</Label>
        <Input
          id={probabilityId}
          type="number"
          min={0}
          max={1}
          step={0.01}
          value={failureProbability}
          onChange={(e) => setFailureProbability(e.target.value)}
          disabled={disabled}
        />
        <p className="text-xs text-muted-foreground">
          Sugestão automática só é gerada quando a probabilidade excede 0.7
          (RF-22).
        </p>
      </div>

      <Button
        type="submit"
        disabled={disabled}
        size="sm"
        className="self-start"
      >
        {disabled ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
        ) : (
          <Sparkles className="h-3.5 w-3.5" />
        )}
        Gerar sugestão
      </Button>
    </form>
  );
}

// ── Referências (texto, nunca link inventado — RF-23 §7) ─────────────────────

function ReferencesList({ references }: { references: ManualReference[] }) {
  if (references.length === 0) {
    return (
      <p
        data-testid="maintenance-references"
        className="text-xs text-muted-foreground"
      >
        Nenhum manual foi citado como referência para este plano.
      </p>
    );
  }
  return (
    <ul data-testid="maintenance-references" className="flex flex-col gap-1.5">
      {references.map((ref, i) => (
        <li
          key={`${ref.file_name}-${ref.page}-${ref.chunk_index}-${i}`}
          className="flex items-start gap-2 text-xs text-muted-foreground"
        >
          <FileText className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          {/* Sem URL segura conhecida para o PDF de origem (nenhum endpoint
              de arquivo estático existe hoje) — mostrado como texto, nunca
              como link inventado (RF-23 §7). */}
          <span>
            <span className="font-mono">{ref.file_name}</span> — página{" "}
            {ref.page}
            <span className="ml-1 text-[10px] opacity-70">
              (score {ref.score.toFixed(2)})
            </span>
          </span>
        </li>
      ))}
    </ul>
  );
}

// ── Corpo do painel — Markdown incremental + estados ──────────────────────────

interface AssistantBodyProps {
  status: MaintenanceAssistantStatus;
  markdown: string;
  references: ManualReference[];
  message: string | null;
}

function AssistantBody({
  status,
  markdown,
  references,
  message,
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
      <div className="flex items-start gap-2 rounded-md border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-slate-400" />
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
        <div className="flex flex-col gap-2 border-t border-slate-200 pt-3">
          <span className="text-xs font-medium text-slate-500">
            Manuais consultados
          </span>
          <ReferencesList references={references} />
        </div>
      )}
    </div>
  );
}

// ── Painel ─────────────────────────────────────────────────────────────────────

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
