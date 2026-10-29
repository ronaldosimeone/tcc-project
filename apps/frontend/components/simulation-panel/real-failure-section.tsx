"use client";

// ── Seção "Cenário de dados" — operação normal ou uma falha real documentada,
// com selo de honestidade: "vista no treino" (demonstração) × "inédita"
// (teste honesto). Ao fim da falha o simulador volta sozinho ao normal.

import { memo, useCallback, useEffect, useState } from "react";
import {
  Activity,
  CircleAlert,
  FlaskConical,
  Loader2,
  Square,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { useScenarioStatus } from "@/hooks/use-scenario-status";
import { setSimulatorMode } from "@/lib/api-client";
import {
  fetchFailureScenarios,
  playFailureScenario,
  type FailureScenario,
} from "@/lib/failure-scenarios-api";
import { FailureScenarioItem } from "./failure-scenario-item";

const NORMAL = "normal";

const RealFailureSectionComponent = () => {
  const [scenarios, setScenarios] = useState<FailureScenario[] | null>(null);
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

  const run = useCallback(
    async (key: string, action: () => Promise<unknown>) => {
      setPending(key);
      setError(null);
      try {
        await action();
        refresh();
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Falha ao trocar cenário",
        );
      } finally {
        setPending(null);
      }
    },
    [refresh],
  );

  const playNormal = useCallback(
    () => void run(NORMAL, () => setSimulatorMode("NORMAL")),
    [run],
  );
  const play = useCallback(
    (id: string) => void run(id, () => playFailureScenario(id)),
    [run],
  );

  const active = status?.scenario ?? null;
  const isNormal = status?.mode === "NORMAL" && !active;
  const progress =
    status?.step != null && status.total
      ? (status.step / status.total) * 100
      : 0;

  return (
    <section className="flex flex-col gap-3" aria-label="Cenário de dados">
      <div className="flex items-baseline justify-between">
        <Label className="text-sm">Cenário de dados</Label>
        {status && (
          <Badge variant="outline" className="font-mono text-xs">
            {active ? "FALHA REAL" : status.mode}
          </Badge>
        )}
      </div>

      <Button
        type="button"
        variant="outline"
        aria-pressed={isNormal}
        data-active={isNormal}
        disabled={pending !== null}
        onClick={playNormal}
        className="h-auto justify-start gap-3 px-3 py-2.5 text-left data-[active=true]:border-primary data-[active=true]:shadow-[0_0_0_1px_var(--primary)]"
      >
        {pending === NORMAL ? (
          <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" />
        ) : (
          <Activity
            className="h-3.5 w-3.5 shrink-0 text-muted-foreground data-[active=true]:text-primary"
            data-active={isNormal}
          />
        )}
        <span className="flex min-w-0 flex-col">
          <span className="text-sm font-medium text-foreground">Normal</span>
          <span className="whitespace-normal text-xs font-normal text-muted-foreground">
            Operação real, sem falha.
          </span>
        </span>
      </Button>

      <Label className="mt-1 flex items-center gap-1.5 text-sm">
        <FlaskConical className="h-3.5 w-3.5" aria-hidden="true" />
        Falha real
      </Label>

      <p className="text-xs text-muted-foreground">
        Reproduz a sequência real de sensores de uma falha documentada, a partir
        do momento em que ela começou. Falhas “inéditas” ocorreram depois de
        todo o treino: o que o modelo fizer é resultado, não demonstração.
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
            onPlay={play}
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
            <span className="font-medium text-foreground">Durante a falha</span>
            <Button
              type="button"
              size="sm"
              variant="ghost"
              className="h-6 px-2 text-xs"
              onClick={playNormal}
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
