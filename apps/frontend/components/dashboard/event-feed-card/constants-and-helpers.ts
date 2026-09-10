// ── Tipos/constantes/helpers — RNF-58: extraído de EventFeedCard ────────────

import type { ComponentType } from "react";
import { AlertTriangle, CheckCircle2, Info, Wrench } from "lucide-react";
import { RISK_THRESHOLDS } from "@/lib/risk-thresholds";

/** Tempo relativo em pt-BR — "agora", "há 2 min", "há 3 h". */
export function fromNow(ts: number, now: number = Date.now()): string {
  const diffSec = Math.max(0, Math.floor((now - ts) / 1000));
  if (diffSec < 30) return "agora";
  if (diffSec < 60) return `há ${diffSec}s`;
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `há ${diffMin} min`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `há ${diffHr}h`;
  const diffDay = Math.floor(diffHr / 24);
  return `há ${diffDay}d`;
}

export type EventTone = "warn" | "ok" | "info" | "wrench";

export interface FeedEvent {
  id: string;
  assetId: string;
  message: string;
  ts: number; // epoch ms
  tone: EventTone;
  /** Probabilidade de falha [0–1] — usada para gating de explicabilidade. */
  probability?: number;
}

// ── Explicabilidade da IA (Feature Importance mock) ─────────────────────────
// Para eventos críticos, simulamos a saída de SHAP/feature-importance do
// modelo. As 3 explicações alternam-se de forma determinística pelo hash
// do assetId, dando consistência por ativo entre re-renders.
const EXPLANATIONS: readonly string[] = [
  "Vibração elevada (3.2g) e Pressão atípica (8.1 bar).",
  "Temperatura do óleo acima do baseline (+12°C) e corrente irregular do motor.",
  "Ciclo de carga prolongado (>40s) combinado com queda na pressão DV.",
];

function hashAssetId(id: string): number {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) | 0;
  return Math.abs(h);
}

export function explanationFor(assetId: string): string {
  return EXPLANATIONS[hashAssetId(assetId) % EXPLANATIONS.length];
}

export function isCriticalEvent(evt: FeedEvent): boolean {
  if (
    evt.probability !== undefined &&
    evt.probability >= RISK_THRESHOLDS.CRITICAL
  )
    return true;
  return evt.tone === "warn";
}

export const TONE_ICON: Record<
  EventTone,
  ComponentType<{ className?: string }>
> = {
  warn: AlertTriangle,
  ok: CheckCircle2,
  info: Info,
  wrench: Wrench,
};

export const TONE_COLOR: Record<EventTone, string> = {
  warn: "text-amber-600",
  ok: "text-emerald-600",
  info: "text-slate-500",
  wrench: "text-blue-600",
};

export const TONE_BG: Record<EventTone, string> = {
  warn: "bg-amber-50",
  ok: "bg-emerald-50",
  info: "bg-slate-100",
  wrench: "bg-blue-50",
};

// Borda esquerda colorida — padrão visual SCADA para sinalizar severidade
// imediatamente no scan vertical. Eventos críticos têm a sua própria cor.
export const TONE_BORDER: Record<EventTone, string> = {
  warn: "border-l-amber-500",
  ok: "border-l-emerald-500",
  info: "border-l-slate-300",
  wrench: "border-l-blue-500",
};

export const CRITICAL_BORDER = "border-l-red-500";

// Eventos fixos para preencher o feed quando não há alertas vivos.
// Timestamps relativos ao "agora" para parecer vivo no demo.
export const DEMO_EVENTS: Array<Omit<FeedEvent, "ts"> & { ageMin: number }> = [
  {
    id: "demo-1",
    assetId: "APU-Trem-023",
    message: "Risco de anomalia elevado",
    tone: "warn",
    ageMin: 2,
  },
  {
    id: "demo-2",
    assetId: "APU-Trem-011",
    message: "Sinal estabilizado",
    tone: "ok",
    ageMin: 7,
  },
  {
    id: "demo-3",
    assetId: "APU-Trem-031",
    message: "Manutenção preventiva concluída",
    tone: "wrench",
    ageMin: 18,
  },
  {
    id: "demo-4",
    assetId: "APU-Trem-055",
    message: "TP2 abaixo do esperado",
    tone: "info",
    ageMin: 31,
  },
  {
    id: "demo-5",
    assetId: "APU-Trem-015",
    message: "Ciclo de carga/descarga normal",
    tone: "ok",
    ageMin: 44,
  },
];

// Defesa em profundidade: o hook já corta em QUEUE_MAX=5, mas mantemos
// um teto duro aqui caso o feed evolua para acumular histórico próprio.
export const MAX_FEED_EVENTS = 50;
// Quantos eventos efectivamente renderizamos na lista compacta.
// 10 linhas cabem confortavelmente dentro do h-[260px] com scroll suave.
export const VISIBLE_FEED_EVENTS = 10;

/** Horário curto (HH:mm:ss) — formato típico de log industrial / SCADA. */
export function formatClockTime(ts: number): string {
  return new Date(ts).toLocaleTimeString("pt-BR", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}
