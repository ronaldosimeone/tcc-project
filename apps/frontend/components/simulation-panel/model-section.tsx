"use client";

// ── Seção "Modelo activo" — RNF-58: extraído de simulation-panel.tsx ────────

import { useCallback, useEffect, useMemo, useState } from "react";
import { CircleAlert, Loader2 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  listModels,
  swapActiveModel,
  type ModelsListResponse,
} from "@/lib/api-client";
import {
  filterAndFormatModels,
  formatModelName,
  type DisplayModel,
} from "@/lib/model-name";

export function ModelSection() {
  const [data, setData] = useState<ModelsListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<string | null>(null);
  // Selecção controlada localmente — atualiza ANTES do fetch para que o
  // Select reflicta a escolha do utilizador instantaneamente (optimistic UI).
  // Sincronizada com `data.active_model` sempre que o backend confirma.
  const [selected, setSelected] = useState<string>("");

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const fresh = await listModels();
      setData(fresh);
      setSelected(fresh.active_model);
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
      if (!data || rawName === selected) return;
      const previous = selected;
      // Optimistic update — UI move já; backend confirma depois.
      setSelected(rawName);
      setPending(rawName);
      setError(null);
      try {
        await swapActiveModel(rawName);
        await refresh();
      } catch (err) {
        // Reverte para o estado anterior se o swap falhar.
        setSelected(previous);
        setError(
          err instanceof Error ? err.message : "Falha ao trocar de modelo",
        );
      } finally {
        setPending(null);
      }
    },
    [data, refresh, selected],
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
        value={selected}
        onValueChange={handleChange}
        disabled={loading || pending !== null || displayModels.length === 0}
      >
        <SelectTrigger id="model-select" aria-label="Modelo de inferência">
          <SelectValue
            placeholder={loading ? "Carregando..." : "Selecione um modelo"}
          />
        </SelectTrigger>
        <SelectContent position="popper" sideOffset={4} className="z-[100]">
          {displayModels.map(({ value, label }) => (
            <SelectItem key={value} value={value}>
              {label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {pending && (
        <p
          role="status"
          className="flex items-center gap-2 text-xs text-muted-foreground"
        >
          <Loader2 className="h-3 w-3 animate-spin" aria-hidden="true" />
          Trocando para {formatModelName(pending)}…
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
