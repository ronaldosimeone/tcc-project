"use client";

// ── Seção "Falhas reais" — reproduz uma falha documentada, com selo de
// honestidade: "vista no treino" (demonstração) × "inédita" (teste honesto).

import { memo, useCallback, useEffect, useState } from "react";
import { CircleAlert, FlaskConical, Loader2, Play, Square } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useScenarioStatus } from "@/hooks/use-scenario-status";
import {
  fetchFailureScenarios,
  playFailureScenario,
  type FailureScenario,
} from "@/lib/failure-scenarios-api";

const LEAD_OPTIONS: readonly number[] = [10, 30, 60, 360];

interface HonestyBadgeProps {
  seen: boolean;
}

const HonestyBadge = ({ seen }: HonestyBadgeProps) =>
  seen ? (
    <Badge variant="secondary" className="text-[10px]">
      Vista no treino · demonstração
    </Badge>
  ) : (
    <Badge className="text-[10px]">Inédita · teste honesto</Badge>
  );

const RealFailureSectionComponent = () => {
  const [scenarios, setScenarios] = useState<FailureScenario[] | null>(null);
  const [lead, setLead] = useState(30);
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { status, refresh } = useScenarioStatus();

  useEffect(() => {
    let alive = true;
    fetchFailureScenarios()
      .then((list) => alive && setScenarios(list))
      .catch((err: unknown) => {
        if (alive) {
          setError(
            err instanceof Error ? err.message : "Falha ao listar cenários",
          );
        }
      });
    return () => {
      alive = false;
    };
  }, []);

  const play = useCallback(
    async (id: string | null) => {
      setPending(id ?? "stop");
      setError(null);
      try {
        await playFailureScenario({ id, leadMinutes: lead });
        refresh();
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Falha ao iniciar cenário",
        );
      } finally {
        setPending(null);
      }
    },
    [lead, refresh],
  );

  const active = status?.scenario ?? null;
  const progress =
    status?.step != null && status.total
      ? (status.step / status.total) * 100
      : 0;

  return (
    <section className="flex flex-col gap-3" aria-label="Falhas reais">
      <div className="flex items-center justify-between gap-2">
        <Label className="flex items-center gap-1.5 text-sm">
          <FlaskConical className="h-3.5 w-3.5" aria-hidden="true" />
          Falhas reais
        </Label>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Antecedência</span>
          <Select
            value={String(lead)}
            onValueChange={(v) => setLead(Number(v))}
          >
            <SelectTrigger
              className="h-7 w-[92px] text-xs"
              aria-label="Antecedência"
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {LEAD_OPTIONS.map((m) => (
                <SelectItem key={m} value={String(m)} className="text-xs">
                  {m >= 60 ? `${m / 60} h` : `${m} min`}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <p className="text-xs text-muted-foreground">
        Reproduz a sequência real de sensores de uma falha documentada. Falhas
        “inéditas” ocorreram depois de todo o treino: o que o modelo fizer é
        resultado, não demonstração.
      </p>

      {scenarios === null && !error && (
        <p
          role="status"
          className="flex items-center gap-2 text-xs text-muted-foreground"
        >
          <Loader2 className="h-3 w-3 animate-spin" aria-hidden="true" />
          Carregando cenários…
        </p>
      )}

      <ul className="flex flex-col gap-1.5">
        {(scenarios ?? []).map((s) => {
          const isActive = active?.id === s.id;
          return (
            <li
              key={s.id}
              data-active={isActive}
              className="flex items-start justify-between gap-2 rounded-md border border-border bg-card px-3 py-2 data-[active=true]:border-primary"
            >
              <div className="flex min-w-0 flex-col gap-1">
                <span className="text-sm font-medium text-foreground">
                  {s.label}
                </span>
                <div className="flex flex-wrap items-center gap-1.5">
                  <HonestyBadge seen={s.seen_in_training} />
                  <span className="text-[10px] text-muted-foreground">
                    {s.dataset}
                  </span>
                </div>
                <span className="text-xs text-muted-foreground">{s.note}</span>
              </div>
              <Button
                type="button"
                size="sm"
                variant={isActive ? "secondary" : "outline"}
                disabled={!s.available || pending !== null}
                aria-label={`Reproduzir ${s.label}`}
                title={
                  s.available
                    ? undefined
                    : "Dados não disponíveis neste servidor"
                }
                onClick={() => void play(s.id)}
              >
                {pending === s.id ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Play className="h-3.5 w-3.5" />
                )}
              </Button>
            </li>
          );
        })}
      </ul>

      {active && (
        <div
          role="status"
          data-testid="scenario-progress"
          className="flex flex-col gap-1.5 rounded-md border border-border bg-muted/40 p-2"
        >
          <div className="flex items-center justify-between text-xs">
            <span className="font-medium text-foreground">
              {status?.phase === "pre_failure"
                ? "Antes da falha (janela de antecedência)"
                : "Durante a falha"}
            </span>
            <Button
              type="button"
              size="sm"
              variant="ghost"
              className="h-6 px-2 text-xs"
              onClick={() => void play(null)}
              disabled={pending !== null}
            >
              <Square className="mr-1 h-3 w-3" /> Parar
            </Button>
          </div>
          <Progress value={progress} aria-label="Progresso do cenário" />
        </div>
      )}

      {error && (
        <p
          role="alert"
          className="flex items-start gap-2 text-xs text-destructive"
        >
          <CircleAlert className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
          {error}
        </p>
      )}
    </section>
  );
};

export const RealFailureSection = memo(RealFailureSectionComponent);
