"use client";

/**
 * DevProfiler — instrumentação de render baseada em `React.Profiler` (RNF-39).
 *
 * Usada para medir, com dados reais (não análise estática), quantos renders
 * cada componente do Dashboard sofre durante o stream SSE e quanto tempo
 * cada um custa. Zero custo em produção: `ENABLED` é resolvido em build-time
 * (Next.js substitui `process.env.NODE_ENV` estaticamente), então o branch
 * `if (!ENABLED)` sempre bate a versão sem `Profiler` no bundle de prod.
 *
 * Uso (dev/profiling apenas):
 *   <DevProfiler id="FleetHealthTable"><FleetHealthTable ... /></DevProfiler>
 *
 * Leitura dos dados coletados no console do browser:
 *   window.__renderStats            // array de RenderStat
 *   summarizeRenderStats()          // agregado por id: contagem + duração total/média
 */

import { Profiler, type ProfilerOnRenderCallback, type ReactNode } from "react";

export interface RenderStat {
  id: string;
  phase: "mount" | "update" | "nested-update";
  actualDuration: number;
  baseDuration: number;
  startTime: number;
  commitTime: number;
}

declare global {
  interface Window {
    __renderStats?: RenderStat[];
  }
}

const ENABLED = process.env.NODE_ENV !== "production";

/** Limpa o buffer — chamar antes de iniciar uma nova janela de medição. */
export function resetRenderStats(): void {
  if (typeof window !== "undefined") window.__renderStats = [];
}

const onRender: ProfilerOnRenderCallback = (
  id,
  phase,
  actualDuration,
  baseDuration,
  startTime,
  commitTime,
) => {
  if (typeof window === "undefined") return;
  window.__renderStats ??= [];
  window.__renderStats.push({
    id,
    phase,
    actualDuration,
    baseDuration,
    startTime,
    commitTime,
  });
};

export interface DevProfilerProps {
  id: string;
  children: ReactNode;
}

export function DevProfiler({ id, children }: DevProfilerProps) {
  if (!ENABLED) return <>{children}</>;
  return (
    <Profiler id={id} onRender={onRender}>
      {children}
    </Profiler>
  );
}

/** Agregado legível por `id`: nº de renders, duração total/média (ms). */
export function summarizeRenderStats(): Record<
  string,
  { renders: number; totalMs: number; avgMs: number }
> {
  const stats = typeof window !== "undefined" ? window.__renderStats ?? [] : [];
  const byId: Record<string, { renders: number; totalMs: number }> = {};
  for (const s of stats) {
    byId[s.id] ??= { renders: 0, totalMs: 0 };
    byId[s.id].renders += 1;
    byId[s.id].totalMs += s.actualDuration;
  }
  const out: Record<
    string,
    { renders: number; totalMs: number; avgMs: number }
  > = {};
  for (const [id, v] of Object.entries(byId)) {
    out[id] = {
      renders: v.renders,
      totalMs: parseFloat(v.totalMs.toFixed(2)),
      avgMs: parseFloat((v.totalMs / v.renders).toFixed(3)),
    };
  }
  return out;
}
