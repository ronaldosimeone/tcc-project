"use client";

// ── Log de Eventos — RNF-58: extraído de sensor-monitor.tsx ─────────────────

import { memo } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ClipboardList,
  Loader2,
  Trash2,
  XCircle,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { usePredictionHistory } from "@/hooks/use-prediction-history";
import type { RiskLevel } from "@/hooks/use-sensor-data";
import type { PredictResponse } from "@/lib/api-client";
import { cn } from "@/lib/utils";

export interface EventLogProps {
  latest: PredictResponse | null;
  riskLevel: RiskLevel;
}

const RISK_CFG: Record<
  RiskLevel,
  { cls: string; icon: React.ComponentType<{ className?: string }> }
> = {
  NORMAL: {
    cls: "border-emerald-400/40 bg-emerald-400/10 text-emerald-700",
    icon: CheckCircle2,
  },
  ALERTA: {
    cls: "border-amber-400/40 bg-amber-400/10 text-amber-700",
    icon: AlertTriangle,
  },
  CRÍTICO: {
    cls: "border-red-400/40 bg-red-400/10 text-red-700",
    icon: XCircle,
  },
};

export const EventLog = memo(function EventLog({
  latest,
  riskLevel,
}: EventLogProps) {
  const { history, isMounted, clearHistory } = usePredictionHistory(latest);

  return (
    <Card
      data-testid="prediction-history"
      className={cn(
        "flex flex-col border-slate-200 transition-colors duration-500",
        riskLevel === "CRÍTICO" && "border-red-300",
        riskLevel === "ALERTA" && "border-amber-300",
      )}
    >
      <CardHeader className="px-4 pb-2 pt-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-sm font-semibold text-foreground/90">
            <ClipboardList className="h-4 w-4 text-muted-foreground" />
            Histórico de Eventos
            {isMounted && history.length > 0 && (
              <span className="flex h-4 min-w-4 items-center justify-center rounded-full bg-primary/15 px-1 text-[9px] font-bold text-primary">
                {history.length}
              </span>
            )}
          </CardTitle>
          {isMounted && history.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={clearHistory}
              className="h-6 gap-1 px-2 text-[10px] text-muted-foreground hover:text-destructive"
            >
              <Trash2 className="h-3 w-3" />
              Limpar
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="flex min-h-0 flex-1 flex-col p-0">
        {!isMounted ? (
          <div className="flex flex-1 items-center justify-center py-8">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground/30" />
          </div>
        ) : history.length === 0 ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-2 py-8 text-center">
            <ClipboardList className="h-8 w-8 text-muted-foreground/20" />
            <p className="text-[10px] text-muted-foreground/50">
              Sem eventos registrados
            </p>
          </div>
        ) : (
          <ScrollArea className="h-[196px]">
            <div>
              {history.map((entry) => {
                const { cls, icon: Icon } = RISK_CFG[entry.riskLevel];
                const time = new Date(entry.timestamp).toLocaleTimeString(
                  "pt-BR",
                  { hour: "2-digit", minute: "2-digit", second: "2-digit" },
                );
                return (
                  <div
                    key={entry.id}
                    data-risk={entry.riskLevel}
                    className="flex items-center gap-2 border-b border-slate-100 px-4 py-2 last:border-0"
                  >
                    <span className="w-14 shrink-0 font-mono text-[9px] text-muted-foreground/60">
                      {time}
                    </span>
                    <Badge
                      variant="outline"
                      className={cn(
                        "gap-1 px-1.5 py-0 text-[9px] font-bold",
                        cls,
                      )}
                    >
                      <Icon className="h-2.5 w-2.5" />
                      {entry.riskLevel}
                    </Badge>
                    <span className="ml-auto font-mono text-xs font-semibold tabular-nums text-foreground">
                      {(entry.failure_probability * 100).toFixed(1)}%
                    </span>
                  </div>
                );
              })}
            </div>
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  );
});
