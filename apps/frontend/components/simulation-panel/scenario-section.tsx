"use client";

// ── Seção "Cenário do simulador" — RNF-58: extraído de simulation-panel.tsx ─

import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CircleAlert,
  Loader2,
  TrendingDown,
  type LucideIcon,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import {
  getSimulatorMode,
  setSimulatorMode,
  type SimulatorMode,
} from "@/lib/api-client";
import { cn } from "@/lib/utils";

interface ScenarioOption {
  readonly value: SimulatorMode;
  readonly label: string;
  readonly description: string;
  readonly icon: LucideIcon;
  readonly tone: "ok" | "warn" | "danger";
}

const SCENARIOS: readonly ScenarioOption[] = [
  {
    value: "NORMAL",
    label: "Normal",
    description: "Operação estável dentro dos limites.",
    icon: Activity,
    tone: "ok",
  },
  {
    value: "DEGRADATION",
    label: "Degradação",
    description: "Drift gradual rumo à falha (lerp 0→1).",
    icon: TrendingDown,
    tone: "warn",
  },
  {
    value: "FAILURE",
    label: "Falha",
    description: "Replay de janelas reais de fuga de ar.",
    icon: AlertTriangle,
    tone: "danger",
  },
];

const TONE_ICON: Record<ScenarioOption["tone"], string> = {
  ok: "text-emerald-600 dark:text-emerald-400",
  warn: "text-amber-600 dark:text-amber-400",
  danger: "text-rose-600 dark:text-rose-400",
};

export function ScenarioSection() {
  const [mode, setMode] = useState<SimulatorMode | null>(null);
  const [loading, setLoading] = useState(true);
  const [pending, setPending] = useState<SimulatorMode | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const fresh = await getSimulatorMode();
        if (alive) setMode(fresh.mode);
      } catch (err) {
        if (alive) {
          setError(err instanceof Error ? err.message : "Falha ao ler cenário");
        }
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const handleChange = useCallback(
    async (next: string) => {
      const candidate = next as SimulatorMode;
      if (candidate === mode) return;
      setPending(candidate);
      setError(null);
      try {
        const fresh = await setSimulatorMode(candidate);
        setMode(fresh.mode);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Falha ao trocar cenário",
        );
      } finally {
        setPending(null);
      }
    },
    [mode],
  );

  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-baseline justify-between">
        <Label className="text-sm">Cenário de dados</Label>
        {mode && (
          <Badge variant="outline" className="font-mono text-[10px]">
            {mode}
          </Badge>
        )}
      </div>

      <RadioGroup
        value={mode ?? ""}
        onValueChange={handleChange}
        disabled={loading || pending !== null}
        className="gap-1.5"
        aria-label="Cenário de simulação"
      >
        {SCENARIOS.map((scenario) => {
          const Icon = scenario.icon;
          const id = `scenario-${scenario.value}`;
          const checked = mode === scenario.value;
          return (
            <Label
              key={scenario.value}
              htmlFor={id}
              data-state={checked ? "checked" : "unchecked"}
              className={cn(
                "group flex cursor-pointer items-center gap-3 rounded-md border border-border bg-card px-3 py-2.5 transition-all",
                "hover:border-input hover:bg-muted/50",
                "data-[state=checked]:border-primary data-[state=checked]:bg-card data-[state=checked]:shadow-[0_0_0_1px_var(--primary)]",
              )}
            >
              <RadioGroupItem
                id={id}
                value={scenario.value}
                className="border-input data-[state=checked]:border-primary"
              />
              <Icon
                className={cn(
                  "h-3.5 w-3.5 shrink-0 text-muted-foreground/70 transition-colors",
                  checked && TONE_ICON[scenario.tone],
                )}
              />
              <div className="flex min-w-0 flex-1 flex-col">
                <span className="text-sm font-medium text-foreground">
                  {scenario.label}
                </span>
                <span className="truncate text-xs text-muted-foreground">
                  {scenario.description}
                </span>
              </div>
            </Label>
          );
        })}
      </RadioGroup>

      {pending && (
        <p
          role="status"
          className="flex items-center gap-2 text-xs text-muted-foreground"
        >
          <Loader2 className="h-3 w-3 animate-spin" aria-hidden="true" />
          Aplicando cenário {pending}…
        </p>
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
}
