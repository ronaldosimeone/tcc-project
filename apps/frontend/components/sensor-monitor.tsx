"use client";

// RNF-58: decomposto em `components/sensor-monitor/*` — cada sub-componente
// (Sparkline, SparkKpiCard, MainAreaChart, BooleanPanel, OperationalDonut,
// EventLog, PressureRadials, DegradedModeBanner, SensorMonitorHeader) mora em
// seu próprio arquivo; este componente é só o orquestrador (busca dados via
// hooks, decide o estado de risco, monta o layout). Nenhuma mudança de
// comportamento/DOM/classe CSS — puro reposicionamento de código já existente.

import {
  AlertTriangle,
  Gauge,
  ServerCrash,
  Thermometer,
  Zap,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { AlertToastQueue } from "@/components/alert-toast-queue";
import { useSensorData, getRiskLevel } from "@/hooks/use-sensor-data";
import { useAlertWebSocket } from "@/hooks/use-alert-websocket";
import { isCriticalProb } from "@/lib/risk-thresholds";
import { C } from "./sensor-monitor/constants";
import { SensorMonitorHeader } from "./sensor-monitor/header";
import { DegradedModeBanner } from "./sensor-monitor/degraded-mode-banner";
import { SparkKpiCard } from "./sensor-monitor/spark-kpi-card";
import { MainAreaChart } from "./sensor-monitor/main-area-chart";
import { BooleanPanel } from "./sensor-monitor/bool-signal";
import { OperationalDonut } from "./sensor-monitor/operational-donut";
import { EventLog } from "./sensor-monitor/event-log";
import { PressureRadials } from "./sensor-monitor/pressure-gauge";

export default function SensorMonitor() {
  const {
    history,
    latest,
    currentPayload,
    isLoading,
    error,
    sseStatus,
    sseReconnectAttempt,
  } = useSensorData();

  const { alerts, status: wsStatus, acknowledge } = useAlertWebSocket();

  // Bug-fix: NÃO misturar a fila de toasts com a telemetria.
  // A fila `alerts` é apenas histórico de notificações pendentes de
  // "Reconhecer" — se for usada como fallback/max aqui, o dashboard fica
  // "latched" no último pico até o operador fechar os toasts à mão.
  // A única fonte de verdade para o estado actual do compressor é o
  // último pacote SSE (`latest.failure_probability`).
  const effectiveProb = latest?.failure_probability ?? 0;
  const effectiveRiskLevel = getRiskLevel(effectiveProb);
  const anomalyScoreStr = (effectiveProb * 100).toFixed(1);

  const isOffline = error !== null && !isLoading;
  const isHardOffline = isOffline && history.length === 0;
  const isDegraded = sseStatus === "reconnecting" && history.length > 0;
  const isLive = sseStatus === "connected";

  // Modo "imersivo" 3-tier: cores estritamente por nível de risco oficial.
  // CRÍTICO → vermelho (reservado), ALERTA → âmbar, NORMAL → branco neutro.
  const isCriticalState = isCriticalProb(effectiveProb);
  const isAlertState = effectiveRiskLevel === "ALERTA";

  return (
    <div className="w-full h-full overflow-y-auto">
      <div className="flex flex-col gap-4 p-4">
        <SensorMonitorHeader
          riskLevel={effectiveRiskLevel}
          isCriticalState={isCriticalState}
          isAlertState={isAlertState}
          isLive={isLive}
          isLoading={isLoading}
          isOffline={isOffline}
          sseStatus={sseStatus}
          wsStatus={wsStatus}
        />

        {isDegraded && <DegradedModeBanner attempt={sseReconnectAttempt} />}

        {isHardOffline ? (
          /* ── ErrorState ─────────────────────────────────────── */
          <div
            role="alert"
            className="flex flex-1 flex-col items-center justify-center gap-4 rounded-xl border border-destructive/30 bg-destructive/5 px-6 py-12 text-center"
          >
            <ServerCrash className="h-12 w-12 text-destructive/60" />
            <div>
              <p className="text-sm font-semibold text-foreground">
                Sem conexão com o backend
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                Verifique a API em{" "}
                {process.env.NEXT_PUBLIC_API_URL ?? "localhost:8000"} e
                recarregue.
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => window.location.reload()}
            >
              Tentar novamente
            </Button>
          </div>
        ) : (
          <>
            {/* ── Seção 1 — Top KPIs (4 colunas) ──────────────── */}
            <div className="grid grid-cols-2 gap-4 xl:grid-cols-4">
              <SparkKpiCard
                title="TP3 — Pressão Painel"
                value={currentPayload.TP3.toFixed(2)}
                unit="bar"
                icon={Gauge}
                sparkData={history}
                sparkKey="TP3"
                sparkColor={C.tp3}
                isLoading={isLoading}
              />
              <SparkKpiCard
                title="Corrente Motor"
                value={currentPayload.Motor_current.toFixed(2)}
                unit="A"
                icon={Zap}
                sparkData={history}
                sparkKey="Motor_current"
                sparkColor={C.current}
                isLoading={isLoading}
              />
              <SparkKpiCard
                title="Temperatura Óleo"
                value={currentPayload.Oil_temperature.toFixed(1)}
                unit="°C"
                icon={Thermometer}
                sparkData={history}
                sparkKey="Oil_temperature"
                sparkColor={C.temp}
                isLoading={isLoading}
              />
              <SparkKpiCard
                title="Anomaly Score"
                value={anomalyScoreStr}
                unit="%"
                icon={AlertTriangle}
                sparkData={history}
                sparkKey="failure_probability"
                sparkColor={effectiveRiskLevel === "NORMAL" ? C.tp3 : C.anomaly}
                isLoading={isLoading}
                alertColor={effectiveRiskLevel === "CRÍTICO"}
              />
            </div>

            {/* ── Seção 2 — Middle: gráfico 75% + sinais 25% ───── */}
            <div className="grid grid-cols-1 gap-4 xl:grid-cols-[3fr_1fr]">
              <MainAreaChart
                data={history}
                isLive={isLive}
                riskLevel={effectiveRiskLevel}
              />
              <BooleanPanel
                COMP={currentPayload.COMP}
                DV_eletric={currentPayload.DV_eletric}
                Towers={currentPayload.Towers}
                MPG={currentPayload.MPG}
                LPS={0}
                Pressure_switch={0}
                Oil_level={currentPayload.Oil_level}
                Caudal_impulses={0}
              />
            </div>

            {/* ── Seção 3 — Bottom Row: donut | eventos | radiais ─ */}
            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
              <OperationalDonut history={history} isLoading={isLoading} />
              <EventLog latest={latest} riskLevel={effectiveRiskLevel} />
              <PressureRadials
                H1={currentPayload.H1}
                DV_pressure={currentPayload.DV_pressure}
                Reservoirs={currentPayload.Reservoirs}
                isLoading={isLoading}
              />
            </div>
          </>
        )}
      </div>

      {/* ── Fila de toasts ────────────────────────────────────── */}
      <AlertToastQueue
        alerts={alerts}
        status={wsStatus}
        onAcknowledge={acknowledge}
      />
    </div>
  );
}
