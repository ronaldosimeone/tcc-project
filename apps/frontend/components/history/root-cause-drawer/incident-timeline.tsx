// ── Linha do tempo real do incidente (evento + OS) ──────────────────────────

import dayjs from "dayjs";

import type { FailureEventDto, WorkOrderDto } from "@/lib/history-api";
import { FAILURE_CAUSE_LABEL } from "@/lib/history-types";
import { cn } from "@/lib/utils";

interface Step {
  label: string;
  at: string;
  detail: string;
  tone: "warn" | "danger" | "neutral";
}

const DOT: Record<Step["tone"], string> = {
  warn: "bg-chart-4 ring-chart-4/30",
  danger: "bg-destructive ring-destructive/30",
  neutral: "bg-muted-foreground ring-muted",
};

const fmt = (ts: string): string => dayjs(ts).format("DD/MM/YYYY HH:mm:ss");

const buildSteps = (
  event: FailureEventDto,
  wo: WorkOrderDto | null,
): Step[] => {
  const steps: Step[] = [
    {
      label: "Episódio aberto",
      at: event.started_at,
      detail: `P(falha) ≥ ${Math.round(
        event.threshold * 100,
      )}% por amostras consecutivas (RN-03)`,
      tone: "warn",
    },
  ];
  if (event.ended_at) {
    steps.push({
      label: "Episódio encerrado",
      at: event.ended_at,
      detail: `${event.n_samples} amostras · pico ${Math.round(
        event.peak_probability * 100,
      )}%`,
      tone: event.severity === "critico" ? "danger" : "warn",
    });
  }
  if (wo) {
    steps.push({
      label: "OS aberta",
      at: wo.opened_at,
      detail: wo.assignee ?? "Sem responsável",
      tone: "neutral",
    });
    if (wo.started_at)
      steps.push({
        label: "Intervenção iniciada",
        at: wo.started_at,
        detail: "",
        tone: "neutral",
      });
    if (wo.closed_at) {
      const verdict =
        wo.is_false_alarm === true
          ? "Falso alarme"
          : wo.confirmed_cause
          ? `Causa: ${FAILURE_CAUSE_LABEL[wo.confirmed_cause]}`
          : wo.status;
      steps.push({
        label: "OS encerrada",
        at: wo.closed_at,
        detail: verdict,
        tone: "neutral",
      });
    }
  }
  return steps;
};

interface IncidentTimelineProps {
  event: FailureEventDto;
  workOrder: WorkOrderDto | null;
}

export function IncidentTimeline({ event, workOrder }: IncidentTimelineProps) {
  const steps = buildSteps(event, workOrder);
  return (
    <section>
      <p className="mb-3 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
        Linha do tempo do incidente
      </p>
      <ol className="relative space-y-3">
        <span
          aria-hidden="true"
          className="absolute left-[5px] top-2 bottom-2 w-px bg-muted"
        />
        {steps.map((step, i) => (
          <li
            key={`${step.label}-${i}`}
            className="relative flex items-start gap-3 pl-1"
          >
            <span
              className={cn(
                "mt-1 inline-flex h-2.5 w-2.5 shrink-0 rounded-full ring-4",
                DOT[step.tone],
              )}
              aria-hidden="true"
            />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold text-foreground">
                {i + 1}. {step.label}
              </p>
              <p className="font-mono text-xs text-muted-foreground">
                {fmt(step.at)}
              </p>
              {step.detail && (
                <p className="text-xs text-muted-foreground">{step.detail}</p>
              )}
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}
