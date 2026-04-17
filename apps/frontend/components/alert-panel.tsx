"use client";

/**
 * AlertPanel — RF-08 / RNF-14.
 *
 * RF-08:  Banner vermelho proeminente quando failure_probability >= 0.65 (CRÍTICO).
 * RNF-14: Histórico persistido via usePredictionHistory (localStorage, últimos 50).
 *
 * Estados visuais:
 *   NORMAL  — borda e fundo padrão do tema escuro.
 *   ALERTA  — borda âmbar sutil com fundo levemente âmbar.
 *   CRÍTICO — borda vermelha + banner de falha crítica no topo (RF-08).
 *
 * Posição: painel lateral direito, altura total, fixo enquanto o conteúdo
 * principal rola. A largura é gerenciada pelo flex layout do SensorMonitor.
 */

import {
  AlertTriangle,
  Bell,
  CheckCircle2,
  ClipboardList,
  Loader2,
  Trash2,
  XCircle,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import {
  usePredictionHistory,
  type PredictionHistoryEntry,
} from "@/hooks/use-prediction-history";
import type { PredictResponse } from "@/lib/api-client";
import type { RiskLevel } from "@/hooks/use-sensor-data";

// ── Props ─────────────────────────────────────────────────────────────────

interface AlertPanelProps {
  latest: PredictResponse | null;
  riskLevel: RiskLevel;
}

// ── Sub-componentes ───────────────────────────────────────────────────────

const RISK_CONFIG: Record<
  RiskLevel,
  {
    badgeClass: string;
    icon: React.ComponentType<{ className?: string }>;
  }
> = {
  NORMAL: {
    badgeClass: "border-green-500/40 bg-green-500/10 text-green-400",
    icon: CheckCircle2,
  },
  ALERTA: {
    badgeClass: "border-amber-500/40 bg-amber-500/10 text-amber-400",
    icon: AlertTriangle,
  },
  CRÍTICO: {
    badgeClass: "border-red-500/40 bg-red-500/10 text-red-400",
    icon: XCircle,
  },
};

function RiskBadge({ level }: { level: RiskLevel }) {
  const { icon: Icon, badgeClass } = RISK_CONFIG[level];
  return (
    <Badge
      variant="outline"
      className={cn("gap-1 px-2 py-0.5 text-[10px] font-bold", badgeClass)}
    >
      <Icon className="h-3 w-3" />
      {level}
    </Badge>
  );
}

function HistoryRow({ entry }: { entry: PredictionHistoryEntry }) {
  const pct = (entry.failure_probability * 100).toFixed(1);
  const time = new Date(entry.timestamp).toLocaleTimeString("pt-BR", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  const probColor =
    entry.riskLevel === "CRÍTICO"
      ? "text-red-400"
      : entry.riskLevel === "ALERTA"
      ? "text-amber-400"
      : "text-muted-foreground";

  return (
    <div
      className="flex items-center gap-2 border-b border-border/40 px-3 py-2 last:border-0"
      data-risk={entry.riskLevel}
    >
      <span className="w-16 shrink-0 font-mono text-[10px] text-muted-foreground/60">
        {time}
      </span>
      <RiskBadge level={entry.riskLevel} />
      <span
        className={cn(
          "ml-auto font-mono text-xs font-semibold tabular-nums",
          probColor,
        )}
      >
        {pct}%
      </span>
    </div>
  );
}

// ── Componente principal ───────────────────────────────────────────────────

export function AlertPanel({ latest, riskLevel }: AlertPanelProps) {
  const { history, isMounted, clearHistory } = usePredictionHistory(latest);

  const isCritical = riskLevel === "CRÍTICO";
  const isAlert = riskLevel === "ALERTA";

  // Probabilidade atual — clamped para evitar valores fora do range
  const rawProb = latest?.failure_probability ?? 0;
  const currentProb = Math.min(100, Math.max(0, rawProb * 100)).toFixed(1);

  const probColor = isCritical
    ? "text-red-400"
    : isAlert
    ? "text-amber-400"
    : "text-green-400";

  return (
    <aside
      className={cn(
        // RNF-22: 256px em 1024px (lg) libera espaço para o conteúdo principal;
        // 288px volta no breakpoint xl (1280px+) para o layout full-HD.
        "flex h-full w-64 shrink-0 flex-col border-l transition-colors duration-700 xl:w-72",
        isCritical
          ? "border-red-500/30 bg-red-500/0.02"
          : isAlert
          ? "border-amber-500/30 bg-amber-500/0.02"
          : "border-border bg-background/50",
      )}
      data-testid="alert-panel"
      data-risk={riskLevel}
      aria-label="Painel de Auditoria de Predições"
    >
      {/* ── RF-08: Banner de falha crítica ───────────────────────────────── */}
      {isCritical && (
        <div
          role="alert"
          data-testid="critical-banner"
          className="flex items-center gap-2 bg-destructive px-4 py-2.5 text-destructive-foreground"
        >
          <AlertTriangle className="h-4 w-4 shrink-0 animate-pulse" />
          <span className="text-xs font-bold tracking-wide">
            FALHA CRÍTICA DETECTADA
          </span>
        </div>
      )}

      {/* ── Cabeçalho ─────────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between border-b border-border/60 px-4 py-3">
        <div className="flex items-center gap-2">
          <Bell className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-xs font-semibold text-foreground/90">
            Auditoria
          </span>
          {isMounted && history.length > 0 && (
            <span className="flex h-4 min-w-4 items-center justify-center rounded-full bg-primary/20 px-1 text-[9px] font-bold text-primary">
              {history.length}
            </span>
          )}
        </div>

        {isMounted && history.length > 0 && (
          <Button
            variant="ghost"
            size="sm"
            onClick={clearHistory}
            aria-label="Limpar histórico"
            className="h-6 gap-1 px-2 text-[10px] text-muted-foreground hover:text-destructive"
          >
            <Trash2 className="h-3 w-3" />
            Limpar
          </Button>
        )}
      </div>

      {/* ── Status atual ──────────────────────────────────────────────────── */}
      <div className="border-b border-border/60 px-4 py-3">
        <p className="mb-2 text-[9px] font-semibold uppercase tracking-widest text-muted-foreground">
          Status Atual
        </p>
        <div className="flex items-center justify-between">
          <RiskBadge level={riskLevel} />
          <span
            className={cn(
              "font-mono text-lg font-bold tabular-nums",
              probColor,
            )}
            aria-label={`Probabilidade de falha: ${currentProb}%`}
          >
            {currentProb}%
          </span>
        </div>
      </div>

      {/* ── Histórico de eventos ──────────────────────────────────────────── */}
      <div className="flex min-h-0 flex-1 flex-col">
        <div className="px-4 py-2.5">
          <p className="text-[9px] font-semibold uppercase tracking-widest text-muted-foreground">
            Histórico de eventos
          </p>
        </div>

        {/* Estado: aguardando hidratação (SSR → cliente) */}
        {!isMounted ? (
          <div className="flex flex-1 items-center justify-center">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground/30" />
          </div>
        ) : history.length === 0 ? (
          /* Estado: sem eventos registrados */
          <div
            className="flex flex-1 flex-col items-center justify-center gap-2 px-4 text-center"
            data-testid="empty-state"
          >
            <ClipboardList className="h-8 w-8 text-muted-foreground/20" />
            <p className="text-[10px] text-muted-foreground/50">
              Sem eventos registrados
            </p>
          </div>
        ) : (
          /* Estado: lista de eventos */
          <ScrollArea className="flex-1">
            <div>
              {history.map((entry) => (
                <HistoryRow key={entry.id} entry={entry} />
              ))}
            </div>
          </ScrollArea>
        )}
      </div>
    </aside>
  );
}
