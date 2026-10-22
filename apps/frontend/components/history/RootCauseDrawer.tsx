"use client";

/**
 * Root Cause Drawer — análise de um evento real (`GET /v1/events/{id}`).
 *
 *  1. Cabeçalho — descrição + severidade.
 *  2. P(falha) real de 2 h antes a 30 min depois do episódio.
 *  3. Contribuição por sensor congelada na abertura (oclusão, RF-32).
 *  4. Plano sugerido pelo assistente (RAG via MCP) + fundamentação (RN-04).
 *  5. Linha do tempo real (episódio + OS).
 *  6. Ordem de serviço — o veredito do técnico vira rótulo (ground truth).
 */

import { AlertOctagon } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { useEventDetail } from "@/hooks/use-event-detail";
import type {
  FailureEventDetailDto,
  WorkOrderUpdatePayload,
} from "@/lib/history-api";
import type { HistoryEvent } from "@/lib/history-types";
import { cn } from "@/lib/utils";
import { AiDiagnosis } from "./root-cause-drawer/ai-diagnosis";
import { SEVERITY_BADGE, SEVERITY_ICON } from "./root-cause-drawer/constants";
import { IncidentTimeline } from "./root-cause-drawer/incident-timeline";
import { PredictiveWindowChart } from "./root-cause-drawer/predictive-window-chart";
import { SensorContributions } from "./root-cause-drawer/sensor-contributions";
import { WorkOrderPanel } from "./root-cause-drawer/work-order-panel";

interface DrawerBodyProps {
  detail: FailureEventDetailDto;
  busy: boolean;
  actionError: string | null;
  onOpen: () => void;
  onChange: (payload: WorkOrderUpdatePayload) => void;
}

const DrawerBody = ({
  detail,
  busy,
  actionError,
  onOpen,
  onChange,
}: DrawerBodyProps) => (
  <>
    <PredictiveWindowChart
      points={detail.sensor_window}
      startedAt={detail.event.started_at}
      threshold={detail.event.threshold}
    />
    {detail.root_cause && detail.root_cause.sensor_contributions.length > 0 ? (
      <SensorContributions
        contributions={detail.root_cause.sensor_contributions}
      />
    ) : (
      <p className="text-xs text-muted-foreground">
        Sem snapshot de causa raiz (evento reconstruído do histórico ou buffer
        ainda frio).
      </p>
    )}
    <AiDiagnosis rootCause={detail.root_cause} />
    <IncidentTimeline event={detail.event} workOrder={detail.work_order} />
    <WorkOrderPanel
      workOrder={detail.work_order}
      busy={busy}
      error={actionError}
      onOpen={onOpen}
      onChange={onChange}
    />
  </>
);

interface RootCauseDrawerProps {
  event: HistoryEvent | null;
  onClose: () => void;
  /** Chamado após ações da OS, para recarregar KPIs/tabela. */
  onChanged?: () => void;
}

export default function RootCauseDrawer({
  event,
  onClose,
  onChanged,
}: RootCauseDrawerProps) {
  const {
    detail,
    loading,
    error,
    actionError,
    busy,
    openWorkOrder,
    changeWorkOrder,
  } = useEventDetail(event?.eventId ?? null, onChanged);
  const SevIcon = event ? SEVERITY_ICON[event.severity] : AlertOctagon;
  const badgeCls = event
    ? SEVERITY_BADGE[event.severity]
    : SEVERITY_BADGE.ALERTA;

  return (
    <Sheet
      open={event !== null}
      onOpenChange={(o) => {
        if (!o) onClose();
      }}
    >
      <SheetContent
        side="right"
        className="w-full overflow-y-auto sm:max-w-[520px]"
      >
        <SheetHeader>
          <div className="flex items-start justify-between gap-3 pr-8">
            <div className="min-w-0 flex-1">
              <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
                Análise de Causa Raiz · {event?.id ?? ""}
              </p>
              <SheetTitle className="mt-0.5 text-base font-semibold text-foreground">
                {event?.description ?? "—"}
              </SheetTitle>
              <SheetDescription className="mt-1 font-mono text-xs text-muted-foreground">
                {event?.equipment ?? ""} · {event?.timestamp ?? ""}
              </SheetDescription>
            </div>
            {event && (
              <Badge
                variant="outline"
                className={cn("gap-1 text-xs font-semibold", badgeCls)}
              >
                <SevIcon className="h-3 w-3" />
                {event.severity}
              </Badge>
            )}
          </div>
        </SheetHeader>

        {event && (
          <div className="flex flex-col gap-5 px-6 pb-6">
            {loading && !detail && (
              <div className="space-y-3" aria-busy="true">
                <Skeleton className="h-40" />
                <Skeleton className="h-24" />
                <Skeleton className="h-32" />
              </div>
            )}
            {error && (
              <p role="alert" className="text-sm text-destructive">
                {error}
              </p>
            )}
            {detail && (
              <DrawerBody
                detail={detail}
                busy={busy}
                actionError={actionError}
                onOpen={() => void openWorkOrder()}
                onChange={(payload) => void changeWorkOrder(payload)}
              />
            )}
          </div>
        )}
      </SheetContent>
    </Sheet>
  );
}
