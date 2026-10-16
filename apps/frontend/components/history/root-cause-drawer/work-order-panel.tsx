// ── Ordem de serviço do evento: abrir, executar e registrar o veredito ──────
// O veredito do técnico (causa confirmada ou falso alarme) é o rótulo que
// alimenta MTBF/MTTR/precisão e o re-treino — por isso fechar a OS exige um.

import { useState } from "react";
import { ClipboardCheck, ClipboardPlus, Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import type { WorkOrderDto, WorkOrderUpdatePayload } from "@/lib/history-api";
import {
  FAILURE_CAUSE_LABEL,
  WORK_ORDER_STATUS_LABEL,
  type FailureCause,
} from "@/lib/history-types";

interface WorkOrderPanelProps {
  workOrder: WorkOrderDto | null;
  busy: boolean;
  error: string | null;
  onOpen: () => void;
  onChange: (payload: WorkOrderUpdatePayload) => void;
}

const CAUSES = Object.keys(FAILURE_CAUSE_LABEL) as FailureCause[];

export function WorkOrderPanel({
  workOrder,
  busy,
  error,
  onOpen,
  onChange,
}: WorkOrderPanelProps) {
  const [cause, setCause] = useState<FailureCause>("air_leak");

  return (
    <section
      className="rounded-lg border border-border bg-card p-4"
      aria-label="Ordem de serviço"
    >
      <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
        Ordem de serviço
      </p>

      {!workOrder && (
        <Button size="sm" onClick={onOpen} disabled={busy}>
          {busy ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <ClipboardPlus className="h-3.5 w-3.5" />
          )}
          Abrir OS para este evento
        </Button>
      )}

      {workOrder && (
        <div className="space-y-3 text-xs">
          <p className="text-foreground">
            OS #{workOrder.id} ·{" "}
            <span className="font-semibold">
              {WORK_ORDER_STATUS_LABEL[workOrder.status]}
            </span>
            {workOrder.is_false_alarm === true && " · falso alarme"}
            {workOrder.confirmed_cause &&
              ` · ${FAILURE_CAUSE_LABEL[workOrder.confirmed_cause]}`}
          </p>

          {workOrder.status === "open" && (
            <Button
              size="sm"
              variant="outline"
              disabled={busy}
              onClick={() => onChange({ status: "in_progress" })}
            >
              Iniciar intervenção
            </Button>
          )}

          {workOrder.status === "in_progress" && (
            <div className="flex flex-wrap items-center gap-2">
              <label className="sr-only" htmlFor="wo-cause">
                Causa confirmada
              </label>
              <select
                id="wo-cause"
                value={cause}
                onChange={(e) => setCause(e.target.value as FailureCause)}
                className="h-8 rounded-md border border-border bg-background px-2 text-xs text-foreground"
              >
                {CAUSES.map((c) => (
                  <option key={c} value={c}>
                    {FAILURE_CAUSE_LABEL[c]}
                  </option>
                ))}
              </select>
              <Button
                size="sm"
                disabled={busy}
                onClick={() =>
                  onChange({ status: "closed", confirmed_cause: cause })
                }
              >
                <ClipboardCheck className="h-3.5 w-3.5" />
                Fechar: falha confirmada
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={busy}
                onClick={() =>
                  onChange({ status: "closed", is_false_alarm: true })
                }
              >
                Fechar: falso alarme
              </Button>
            </div>
          )}
        </div>
      )}

      {error && (
        <p role="alert" className="mt-2 text-[11px] text-destructive">
          {error}
        </p>
      )}
    </section>
  );
}
