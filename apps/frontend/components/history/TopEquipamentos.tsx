"use client";

/**
 * Ranking dinâmico dos ativos por volume de ocorrências.
 *
 * - Sem truncamento — mostra TODOS os equipamentos presentes nos eventos.
 * - Inclui também ativos da frota canónica (FLEET_ROSTER) que estejam zerados
 *   no recorte, para que a cobertura "monitoramos toda a frota" seja
 *   visualmente explícita.
 * - Cabeçalho fixo + corpo com overflow-y-auto: cresce na vertical sem
 *   estourar o card no mosaico.
 *
 * Tom da barra reflecte a severidade dominante:
 *   - alguma CRÍTICO  → vermelho
 *   - alguma ALERTA   → âmbar
 *   - só NORMAL/Diag  → esmeralda
 *   - 0 ocorrências   → slate (placeholder)
 */

import { useMemo } from "react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { HistoryEvent } from "@/lib/history-types";

type Tone = "red" | "amber" | "emerald" | "slate";

interface RankedRow {
  id: string;
  count: number;
  tone: Tone;
}

const BAR_BG: Record<Tone, string> = {
  red: "bg-red-500",
  amber: "bg-amber-500",
  emerald: "bg-emerald-500",
  slate: "bg-muted-foreground/30",
};

const TEXT_TONE: Record<Tone, string> = {
  red: "text-red-700 dark:text-red-400",
  amber: "text-amber-700 dark:text-amber-400",
  emerald: "text-emerald-700 dark:text-emerald-400",
  slate: "text-muted-foreground/70",
};

function rankEquipments(
  events: HistoryEvent[],
  roster: ReadonlyArray<string>,
): RankedRow[] {
  // Agrega contagens + flags de severidade por equipamento.
  const acc = new Map<
    string,
    { count: number; hasCritical: boolean; hasAlert: boolean }
  >();
  // Seed com a frota cadastrada (GET /v1/assets) — ativos sem eventos aparecem zerados.
  for (const id of roster) {
    acc.set(id, { count: 0, hasCritical: false, hasAlert: false });
  }
  for (const e of events) {
    const cur = acc.get(e.equipment) ?? {
      count: 0,
      hasCritical: false,
      hasAlert: false,
    };
    cur.count++;
    if (e.severity === "CRÍTICO") cur.hasCritical = true;
    else if (e.severity === "ALERTA") cur.hasAlert = true;
    acc.set(e.equipment, cur);
  }

  return Array.from(acc.entries())
    .map(
      ([id, v]): RankedRow => ({
        id,
        count: v.count,
        tone:
          v.count === 0
            ? "slate"
            : v.hasCritical
            ? "red"
            : v.hasAlert
            ? "amber"
            : "emerald",
      }),
    )
    .sort((a, b) => {
      // Maior contagem primeiro; em empate, alfabético estável.
      if (b.count !== a.count) return b.count - a.count;
      return a.id.localeCompare(b.id);
    });
}

interface TopEquipamentosProps {
  events: HistoryEvent[];
  /** Frota cadastrada (ids de GET /v1/assets). */
  roster: string[];
}

export default function TopEquipamentos({
  events,
  roster,
}: TopEquipamentosProps) {
  const rows = useMemo(() => rankEquipments(events, roster), [events, roster]);
  const maxCount = rows[0]?.count ?? 0;

  return (
    <Card className="flex h-full flex-col border border-border bg-card shadow-sm ring-0">
      {/* Cabeçalho fixo (shrink-0) — não rola junto com a lista. */}
      <CardHeader className="shrink-0 px-4 pb-2 pt-3">
        <CardTitle className="flex items-baseline justify-between text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          <span>Top equipamentos</span>
          <span className="font-mono text-[10px] normal-case tracking-normal text-muted-foreground/70">
            {rows.length} ativos
          </span>
        </CardTitle>
      </CardHeader>

      {/* Lista rolável: cresce dentro do espaço disponível do card no mosaico
          e ativa scroll apenas quando estoura. `min-h-0` é essencial dentro
          de um flex parent para que `overflow-y-auto` funcione. */}
      <CardContent className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto px-4 pb-3">
        {rows.length === 0 ? (
          <p className="my-auto text-center text-xs text-muted-foreground/70">
            Sem ativos no recorte.
          </p>
        ) : (
          rows.map((row) => {
            const pct =
              maxCount > 0 ? Math.round((row.count / maxCount) * 100) : 0;
            return (
              <div key={row.id} className="flex flex-col gap-0.5">
                <div className="flex items-baseline justify-between gap-2">
                  <span
                    className={cn(
                      "truncate font-mono text-[11px] font-semibold",
                      row.count === 0
                        ? "text-muted-foreground/70"
                        : "text-foreground/80",
                    )}
                  >
                    {row.id}
                  </span>
                  <span
                    className={cn(
                      "font-mono text-xs font-bold tabular-nums",
                      TEXT_TONE[row.tone],
                    )}
                  >
                    {row.count}
                  </span>
                </div>
                <div
                  className="h-1.5 w-full overflow-hidden rounded-full bg-muted"
                  role="presentation"
                >
                  <div
                    className={cn(
                      "h-full rounded-full transition-all",
                      BAR_BG[row.tone],
                    )}
                    // Mín. 4% quando count===0 e roster pede presença visual:
                    // ainda assim ficamos com 0 — a barra fica vazia (bg-muted).
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            );
          })
        )}
      </CardContent>
    </Card>
  );
}
