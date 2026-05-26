"use client";

/**
 * Painel de simulação MLOps.
 *
 * Aberto a partir do botão "Simulação" da Sidebar. Permite:
 *  - Trocar o modelo activo de inferência (RF-11)
 *  - Trocar o cenário de dados do simulador (RNF-29)
 *
 * Regra crítica: o `value` enviado para a API é SEMPRE o nome bruto
 * (`random_forest_v2`), enquanto o utilizador vê apenas a label limpa
 * (`Random Forest`). Toda a lógica de filtragem/formatação vive em
 * `lib/model-name.ts` e é puramente sincrona.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CircleAlert,
  Loader2,
  Sparkles,
  TrendingDown,
  type LucideIcon,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  getSimulatorMode,
  listModels,
  setSimulatorMode,
  swapActiveModel,
  type ModelsListResponse,
  type SimulatorMode,
} from "@/lib/api-client";
import {
  filterAndFormatModels,
  formatModelName,
  type DisplayModel,
} from "@/lib/model-name";
import { cn } from "@/lib/utils";

interface SimulationPanelProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

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

const TONE_RING: Record<ScenarioOption["tone"], string> = {
  ok: "data-[state=checked]:ring-emerald-500/40 has-data-[state=checked]:border-emerald-500/40",
  warn: "data-[state=checked]:ring-amber-500/40 has-data-[state=checked]:border-amber-500/40",
  danger:
    "data-[state=checked]:ring-rose-500/40 has-data-[state=checked]:border-rose-500/40",
};

export function SimulationPanel({ open, onOpenChange }: SimulationPanelProps) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-full sm:max-w-md">
        <SheetHeader>
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-primary" />
            <SheetTitle>Painel de Simulação</SheetTitle>
          </div>
          <SheetDescription>
            Controle o modelo de inferência e o cenário do simulador em tempo
            real.
          </SheetDescription>
        </SheetHeader>

        <div className="flex flex-1 flex-col gap-6 overflow-y-auto px-6 py-4">
          <ModelSection />
          <ScenarioSection />
        </div>
      </SheetContent>
    </Sheet>
  );
}

// ── Modelo activo ────────────────────────────────────────────────────────────

function ModelSection() {
  const [data, setData] = useState<ModelsListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const fresh = await listModels();
      setData(fresh);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Falha ao carregar modelos",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const displayModels: DisplayModel[] = useMemo(() => {
    if (!data) return [];
    // Considera apenas os modelos com artefacto pronto — evita seleccionar
    // algo que o backend recusará no swap.
    const readyNames = data.models
      .filter((m) => m.artefact_ready)
      .map((m) => m.name);
    return filterAndFormatModels(readyNames);
  }, [data]);

  const handleChange = useCallback(
    async (rawName: string) => {
      if (!data || rawName === data.active_model) return;
      setPending(rawName);
      setError(null);
      try {
        await swapActiveModel(rawName);
        // Optimistic — refresh para confirmar o backend.
        await refresh();
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Falha ao trocar de modelo",
        );
      } finally {
        setPending(null);
      }
    },
    [data, refresh],
  );

  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-baseline justify-between">
        <Label htmlFor="model-select" className="text-sm">
          Modelo de IA
        </Label>
        {data && (
          <Badge variant="outline" className="font-mono text-[10px]">
            activo · {formatModelName(data.active_model)}
          </Badge>
        )}
      </div>

      <Select
        value={data?.active_model ?? ""}
        onValueChange={handleChange}
        disabled={loading || pending !== null || displayModels.length === 0}
      >
        <SelectTrigger id="model-select" aria-label="Modelo de inferência">
          <SelectValue
            placeholder={loading ? "Carregando..." : "Selecione um modelo"}
          />
        </SelectTrigger>
        <SelectContent>
          {displayModels.map(({ value, label }) => (
            <SelectItem key={value} value={value}>
              {label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {pending && (
        <p className="flex items-center gap-2 text-xs text-muted-foreground">
          <Loader2 className="h-3 w-3 animate-spin" />
          Trocando para {formatModelName(pending)}…
        </p>
      )}

      {error && (
        <p className="flex items-start gap-2 text-xs text-destructive">
          <CircleAlert className="h-3.5 w-3.5 shrink-0" />
          {error}
        </p>
      )}
    </section>
  );
}

// ── Cenário do simulador ─────────────────────────────────────────────────────

function ScenarioSection() {
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
        className="gap-2"
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
              className={cn(
                "flex cursor-pointer items-start gap-3 rounded-lg border border-border bg-background px-3 py-3 transition-colors",
                "hover:bg-muted/40",
                checked &&
                  "border-primary/40 bg-primary/5 ring-1 ring-primary/20",
                TONE_RING[scenario.tone],
              )}
            >
              <RadioGroupItem
                id={id}
                value={scenario.value}
                className="mt-0.5"
              />
              <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                <span className="flex items-center gap-2 text-sm font-medium text-foreground">
                  <Icon className="h-3.5 w-3.5" />
                  {scenario.label}
                </span>
                <span className="text-xs text-muted-foreground">
                  {scenario.description}
                </span>
              </div>
            </Label>
          );
        })}
      </RadioGroup>

      {pending && (
        <p className="flex items-center gap-2 text-xs text-muted-foreground">
          <Loader2 className="h-3 w-3 animate-spin" />
          Aplicando cenário {pending}…
        </p>
      )}

      {error && (
        <p className="flex items-start gap-2 text-xs text-destructive">
          <CircleAlert className="h-3.5 w-3.5 shrink-0" />
          {error}
        </p>
      )}
    </section>
  );
}
