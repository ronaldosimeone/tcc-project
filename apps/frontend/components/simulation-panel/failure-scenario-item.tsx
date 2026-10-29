"use client";

// ── Item de "Falhas reais" — extraído de real-failure-section.tsx (RNF-58).

import { Loader2, Play } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { FailureScenario } from "@/lib/failure-scenarios-api";

interface HonestyBadgeProps {
  seen: boolean;
}

const HonestyBadge = ({ seen }: HonestyBadgeProps) =>
  seen ? (
    <Badge variant="secondary" className="text-xs">
      Vista no treino · demonstração
    </Badge>
  ) : (
    <Badge className="text-xs">Inédita · teste honesto</Badge>
  );

interface FailureScenarioItemProps {
  scenario: FailureScenario;
  isActive: boolean;
  isPending: boolean;
  disabled: boolean;
  onPlay: (id: string) => void;
}

export const FailureScenarioItem = ({
  scenario: s,
  isActive,
  isPending,
  disabled,
  onPlay,
}: FailureScenarioItemProps) => (
  <li
    data-active={isActive}
    className="flex items-start justify-between gap-2 rounded-md border border-border bg-card px-3 py-2 data-[active=true]:border-primary"
  >
    <div className="flex min-w-0 flex-col gap-1">
      <span className="text-sm font-medium text-foreground">{s.label}</span>
      <div className="flex flex-wrap items-center gap-1.5">
        <HonestyBadge seen={s.seen_in_training} />
        <span className="text-xs text-muted-foreground">{s.dataset}</span>
      </div>
    </div>
    <Button
      type="button"
      size="sm"
      variant={isActive ? "secondary" : "outline"}
      disabled={!s.available || disabled}
      aria-label={`Reproduzir ${s.label}`}
      title={s.available ? undefined : "Dados não disponíveis neste servidor"}
      onClick={() => onPlay(s.id)}
    >
      {isPending ? (
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
      ) : (
        <Play className="h-3.5 w-3.5" />
      )}
    </Button>
  </li>
);
