"use client";

/** Estados de carregamento / erro / vazio do Histórico (dados reais da API). */

import { AlertCircle, Inbox, RotateCw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

export const HistorySkeleton = () => (
  <div
    className="flex flex-col gap-6"
    aria-busy="true"
    aria-label="Carregando histórico"
  >
    {/* Mesma grade dos 6 KPIs (HistoryKPIs), sem saltos ao carregar. */}
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3 2xl:grid-cols-6">
      {Array.from({ length: 6 }).map((_, i) => (
        <Skeleton key={i} className="h-28" />
      ))}
    </div>
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
      <Skeleton className="h-72 lg:col-span-7" />
      <Skeleton className="h-72 lg:col-span-5" />
    </div>
  </div>
);

interface HistoryErrorStateProps {
  message: string;
  onRetry: () => void;
}

export const HistoryErrorState = ({
  message,
  onRetry,
}: HistoryErrorStateProps) => (
  <Card role="alert" className="border-destructive/40">
    <CardContent className="flex flex-col items-center gap-3 p-8 text-center">
      <AlertCircle className="h-8 w-8 text-destructive" aria-hidden="true" />
      <p className="text-sm font-semibold text-foreground">
        Não foi possível carregar o histórico
      </p>
      <p className="max-w-md text-xs text-muted-foreground">{message}</p>
      <Button variant="outline" size="sm" onClick={onRetry}>
        <RotateCw className="h-3.5 w-3.5" aria-hidden="true" />
        Tentar novamente
      </Button>
    </CardContent>
  </Card>
);

export const HistoryEmptyState = () => (
  <Card>
    <CardContent className="flex flex-col items-center gap-2 p-8 text-center">
      <Inbox className="h-8 w-8 text-muted-foreground" aria-hidden="true" />
      <p className="text-sm font-semibold text-foreground">
        Nenhum episódio de alarme no período
      </p>
      <p className="max-w-md text-xs text-muted-foreground">
        Eventos aparecem aqui quando o modelo mantém a probabilidade de falha
        acima do limiar por amostras consecutivas (RN-03). Ajuste o período ou
        acompanhe o monitoramento ao vivo.
      </p>
    </CardContent>
  </Card>
);
