"use client";

// ── Seção "Falhas reais" — reproduz uma falha documentada, com selo de
// honestidade: "vista no treino" (demonstração) × "inédita" (teste honesto).

import { memo, useCallback, useEffect, useState } from "react";
import { CircleAlert, FlaskConical, Loader2, Square } from "lucide-react";

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
import { FailureScenarioItem } from "./failure-scenario-item";

const LEAD_OPTIONS: readonly number[] = [10, 30, 60, 360];

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

      {scenarios?.length === 0 && (
        <p className="text-xs text-muted-foreground">
          Nenhum cenário de falha disponível neste servidor.
        </p>
      )}

      <ul className="flex flex-col gap-1.5">
        {(scenarios ?? []).map((s) => (
          <FailureScenarioItem
            key={s.id}
            scenario={s}
            isActive={active?.id === s.id}
            isPending={pending === s.id}
            disabled={pending !== null}
            onPlay={(id) => void play(id)}
          />
        ))}
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
