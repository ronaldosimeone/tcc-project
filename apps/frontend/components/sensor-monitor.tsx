"use client";

/**
 * SensorMonitor — componente de composição do dashboard.
 *
 * Orquestra os dois hooks de tempo real:
 *   • useSensorData   — SSE de sensores, polling de predições, WS de alertas do gráfico
 *   • useAlertWebSocket — WS de alertas para toasts (singleton, evita dupla conexão)
 *
 * O resultado de useAlertWebSocket é propagado como props para:
 *   • AlertToastQueue — toasts de alerta (RF-16 / RNF-34)
 *   • ConnectionStatus — indicadores de saúde SSE + WS (RF-17)
 */

import {
  AlertTriangle,
  CheckCircle2,
  Gauge,
  ServerCrash,
  Thermometer,
  Wind,
  WifiOff,
  XCircle,
  Zap,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { SensorChart } from "@/components/sensor-chart";
import { AlertPanel } from "@/components/alert-panel";
import {
  AlertToastQueue,
  ConnectionBanner,
} from "@/components/alert-toast-queue";
import { ConnectionStatus } from "@/components/connection-status";
import {
  useSensorData,
  getRiskLevel,
  type RiskLevel,
} from "@/hooks/use-sensor-data";
import { useAlertWebSocket } from "@/hooks/use-alert-websocket";
import { cn } from "@/lib/utils";

// ── Sub-componentes de apresentação ───────────────────────────────────────

interface GaugeProps {
  probability: number;
  riskLevel: RiskLevel;
}

function riskColor(level: RiskLevel): string {
  return level === "NORMAL"
    ? "hsl(142 71% 45%)"
    : level === "ALERTA"
    ? "hsl(38 92% 50%)"
    : "hsl(0 72% 51%)";
}

function FailureGauge({ probability, riskLevel }: GaugeProps) {
  const cx = 130;
  const cy = 110;
  const r = 88;
  const strokeW = 14;
  const color = riskColor(riskLevel);
  const pct = Math.min(1, Math.max(0, probability));

  function getPoint(angleDeg: number) {
    const rad = (angleDeg * Math.PI) / 180;
    return {
      x: cx + r * Math.cos(rad),
      y: cy - r * Math.sin(rad),
    };
  }

  const startAngle = 180;
  const endAngle = 180 - pct * 180;

  const p1 = getPoint(startAngle);
  const p2 = getPoint(endAngle);
  const p3 = getPoint(0);

  const valuePath =
    pct > 0 ? `M ${p1.x} ${p1.y} A ${r} ${r} 0 0 1 ${p2.x} ${p2.y}` : "";
  const bgPath =
    pct < 1 ? `M ${p2.x} ${p2.y} A ${r} ${r} 0 0 1 ${p3.x} ${p3.y}` : "";

  const percentText = (pct * 100).toFixed(1);

  return (
    <svg
      viewBox={`0 0 ${cx * 2} ${cy + 20}`}
      className="w-full max-w-[260px]"
      aria-label={`Probabilidade de quebra: ${percentText}%`}
    >
      {bgPath && (
        <path
          d={bgPath}
          fill="none"
          stroke="hsl(var(--muted))"
          strokeWidth={strokeW}
          strokeLinecap="round"
        />
      )}
      {valuePath && (
        <path
          d={valuePath}
          fill="none"
          stroke={color}
          strokeWidth={strokeW}
          strokeLinecap="round"
          style={{
            transition: "stroke-dashoffset 0.6s ease, stroke 0.5s ease",
          }}
        />
      )}
      <text
        x={cx}
        y={cy - 10}
        textAnchor="middle"
        fill="white"
        fontSize="28"
        fontWeight="700"
        fontFamily="var(--font-geist-sans, sans-serif)"
      >
        {percentText}%
      </text>
      <text
        x={cx}
        y={cy + 14}
        textAnchor="middle"
        fill="hsl(var(--muted-foreground))"
        fontSize="11"
        fontFamily="var(--font-geist-sans, sans-serif)"
      >
        Probabilidade de Quebra
      </text>
    </svg>
  );
}

interface KpiCardProps {
  title: string;
  value: string;
  unit: string;
  icon: React.ComponentType<{ className?: string }>;
  isLoading?: boolean;
}

function KpiCard({ title, value, unit, icon: Icon, isLoading }: KpiCardProps) {
  return (
    <Card className="border-border bg-card">
      <CardContent className="p-3 lg:p-4">
        <div className="flex items-start gap-2">
          <div className="min-w-0 flex-1">
            <p className="text-[10px] font-medium uppercase leading-tight tracking-wider text-muted-foreground lg:text-xs">
              {title}
            </p>
            {isLoading ? (
              <Skeleton className="mt-2 h-7 w-20" data-testid="kpi-skeleton" />
            ) : (
              <p className="mt-1.5 text-2xl font-bold tabular-nums text-foreground">
                {value}
                <span className="ml-1 text-sm font-normal text-muted-foreground">
                  {unit}
                </span>
              </p>
            )}
          </div>
          <div className="shrink-0 rounded-lg bg-primary/10 p-2">
            <Icon className="h-4 w-4 text-primary" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function StatusBadge({ level }: { level: RiskLevel }) {
  const config: Record<
    RiskLevel,
    { icon: React.ComponentType<{ className?: string }>; className: string }
  > = {
    NORMAL: {
      icon: CheckCircle2,
      className: "border-green-500/40 bg-green-500/10 text-green-400",
    },
    ALERTA: {
      icon: AlertTriangle,
      className: "border-amber-500/40 bg-amber-500/10 text-amber-400",
    },
    CRÍTICO: {
      icon: XCircle,
      className: "border-red-500/40 bg-red-500/10 text-red-400",
    },
  };
  const { icon: Icon, className } = config[level];
  return (
    <Badge
      variant="outline"
      className={`gap-1.5 px-3 py-1 text-sm font-semibold ${className}`}
    >
      <Icon className="h-4 w-4" />
      {level}
    </Badge>
  );
}

interface SensorChipProps {
  label: string;
  value: string;
  unit: string;
  isLoading?: boolean;
}

function SensorChip({ label, value, unit, isLoading }: SensorChipProps) {
  return (
    <div className="flex flex-col gap-0.5 rounded-md border border-border bg-card/60 px-3 py-2">
      <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
      {isLoading ? (
        <Skeleton className="mt-1 h-4 w-10" data-testid="chip-skeleton" />
      ) : (
        <span className="text-sm font-bold tabular-nums text-foreground">
          {value}
          <span className="ml-0.5 text-xs font-normal text-muted-foreground">
            {unit}
          </span>
        </span>
      )}
    </div>
  );
}

// ── Banner de modo degradado (RNF-35) ─────────────────────────────────────

interface DegradedBannerProps {
  reconnectAttempt: number;
}

function DegradedModeBanner({ reconnectAttempt }: DegradedBannerProps) {
  return (
    <div
      role="status"
      aria-live="polite"
      data-testid="degraded-mode-banner"
      className={cn(
        "flex items-center gap-3 rounded-xl border border-amber-500/30",
        "bg-amber-500/5 px-4 py-3 text-amber-400",
      )}
    >
      <WifiOff className="h-4 w-4 shrink-0" />
      <div className="min-w-0 flex-1">
        <p className="text-base font-bold text-amber-500">Modo Degradado</p>
        <p className="text-xs text-amber-400">
          Exibindo últimos dados conhecidos · Reconexão {reconnectAttempt} em
          curso…
        </p>
      </div>
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-400" />
    </div>
  );
}

// ── Componente principal ───────────────────────────────────────────────────

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

  // Singleton do WebSocket de alertas — passado como props para evitar
  // múltiplas conexões ao mesmo endpoint.
  const { alerts, status: wsStatus, acknowledge } = useAlertWebSocket();

  const probability = latest?.failure_probability ?? 0;

  // Unifica a fonte de risco: toma o max entre a predição ML e os alertas WS
  // não reconhecidos. Sem isso, o poll de 5s redefine o riskLevel para NORMAL
  // mesmo com toasts CRÍTICO ativos.
  const alertProb = alerts.reduce((max, a) => Math.max(max, a.probability), 0);
  const effectiveProb = Math.max(probability, alertProb);
  const effectiveRiskLevel = getRiskLevel(effectiveProb);
  const effectiveIsAnomaly = effectiveRiskLevel !== "NORMAL";

  const isOffline = error !== null && !isLoading;
  const isHardOffline = isOffline && history.length === 0;

  // RNF-35: modo degradado — SSE caiu após dados serem carregados
  const isDegradedMode = sseStatus === "reconnecting" && history.length > 0;

  return (
    <div className="flex h-full overflow-hidden">
      {/* ── Área de conteúdo principal (rolável) ── */}
      <div className="flex-1 overflow-y-auto">
        <div className="flex flex-col gap-6 p-6">
          {/* ── Cabeçalho ── */}
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h1 className="text-xl font-bold tracking-tight text-foreground">
                Monitoramento em Tempo Real
              </h1>
              <p className="text-sm text-muted-foreground">
                Compressor de ar · MetroPT-3 ·{" "}
                {sseStatus === "connected"
                  ? "Streaming em tempo real"
                  : sseStatus === "reconnecting"
                  ? "Reconectando…"
                  : "Conectando…"}
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              {/* RF-17: Indicadores de status de conexão */}
              <ConnectionStatus sseStatus={sseStatus} wsStatus={wsStatus} />

              {isOffline && (
                <Badge
                  variant="outline"
                  className="gap-1.5 border-destructive/40 bg-destructive/10 text-destructive"
                >
                  <WifiOff className="h-3 w-3" />
                  Backend offline
                </Badge>
              )}
              {!isLoading && <StatusBadge level={effectiveRiskLevel} />}
            </div>
          </div>

          {/* RNF-35: Banner persistente de modo degradado */}
          {isDegradedMode && (
            <DegradedModeBanner reconnectAttempt={sseReconnectAttempt} />
          )}

          {isHardOffline ? (
            /* ── RNF-21: ErrorState — sem histórico e sem conexão ── */
            <div
              role="alert"
              data-testid="error-state"
              className="flex flex-1 flex-col items-center justify-center gap-4 rounded-xl border border-destructive/30 bg-destructive/5 px-6 py-12 text-center"
            >
              <ServerCrash className="h-12 w-12 text-destructive/60" />
              <div>
                <p className="text-sm font-semibold text-foreground">
                  Sem conexão com o backend
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Verifique se a API está rodando em{" "}
                  {process.env.NEXT_PUBLIC_API_URL ?? "localhost:8000"} e
                  recarregue a página.
                </p>
              </div>
              <Button
                variant="outline"
                size="sm"
                className="mt-1"
                onClick={() => window.location.reload()}
              >
                Tentar novamente
              </Button>
            </div>
          ) : (
            <>
              {/* ── KPI Cards ── */}
              <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
                <KpiCard
                  title="Pressão TP2"
                  value={currentPayload.TP2.toFixed(2)}
                  unit="bar"
                  icon={Gauge}
                  isLoading={isLoading}
                />
                <KpiCard
                  title="Temperatura Óleo"
                  value={currentPayload.Oil_temperature.toFixed(1)}
                  unit="°C"
                  icon={Thermometer}
                  isLoading={isLoading}
                />
                <KpiCard
                  title="Corrente Motor"
                  value={currentPayload.Motor_current.toFixed(2)}
                  unit="A"
                  icon={Zap}
                  isLoading={isLoading}
                />
                <KpiCard
                  title="Reservatório"
                  value={currentPayload.Reservoirs.toFixed(2)}
                  unit="bar"
                  icon={Wind}
                  isLoading={isLoading}
                />
              </div>

              {/* ── Gráficos + Gauge ── */}
              <div className="grid gap-4 lg:grid-cols-5">
                <div
                  className={cn(
                    "lg:col-span-3 rounded-xl transition-colors duration-500",
                    effectiveRiskLevel === "ALERTA" && "bg-yellow-400/10",
                    effectiveRiskLevel === "CRÍTICO" && "bg-red-500/15",
                  )}
                >
                  <SensorChart
                    history={history}
                    isAnomaly={effectiveIsAnomaly}
                    riskLevel={effectiveRiskLevel}
                    isLive={sseStatus === "connected"}
                  />
                </div>

                <Card className="border-border bg-card lg:col-span-2">
                  <div className="flex h-full flex-col items-center gap-4 p-5">
                    <p className="self-start text-sm font-semibold text-foreground/90">
                      Risco de Falha
                    </p>
                    <p className="self-start text-xs text-muted-foreground">
                      Predição do modelo RandomForest
                    </p>

                    {isLoading ? (
                      <div
                        className="flex w-full flex-col items-center gap-3"
                        aria-label="Carregando predição"
                      >
                        <Skeleton
                          className="h-[110px] w-[220px] rounded-t-full"
                          data-testid="gauge-skeleton"
                        />
                        <Skeleton className="h-6 w-24" />
                      </div>
                    ) : (
                      <>
                        <FailureGauge
                          probability={effectiveProb}
                          riskLevel={effectiveRiskLevel}
                        />
                        <StatusBadge level={effectiveRiskLevel} />
                        {latest && (
                          <p className="text-[10px] text-muted-foreground">
                            Classe {latest.predicted_class} ·{" "}
                            {new Date(latest.timestamp).toLocaleTimeString(
                              "pt-BR",
                            )}
                          </p>
                        )}
                        {wsStatus !== "open" && (
                          <div className="mt-auto self-end">
                            <ConnectionBanner status={wsStatus} />
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </Card>
              </div>

              {/* ── Grade de sensores secundários ── */}
              <Card className="border-border bg-card">
                <div className="px-5 pb-5 pt-4">
                  <p className="mb-3 text-sm font-semibold text-foreground/90">
                    Painel de Sensores
                  </p>
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-4 xl:grid-cols-8">
                    <SensorChip
                      label="TP3"
                      value={currentPayload.TP3.toFixed(2)}
                      unit="bar"
                      isLoading={isLoading}
                    />
                    <SensorChip
                      label="H1"
                      value={currentPayload.H1.toFixed(2)}
                      unit="bar"
                      isLoading={isLoading}
                    />
                    <SensorChip
                      label="DV Press."
                      value={currentPayload.DV_pressure.toFixed(2)}
                      unit="bar"
                      isLoading={isLoading}
                    />
                    <SensorChip
                      label="Reserv."
                      value={currentPayload.Reservoirs.toFixed(2)}
                      unit="bar"
                      isLoading={isLoading}
                    />
                    <SensorChip
                      label="COMP"
                      value={currentPayload.COMP === 1 ? "ON" : "OFF"}
                      unit=""
                      isLoading={isLoading}
                    />
                    <SensorChip
                      label="DV Elétric"
                      value={currentPayload.DV_eletric === 1 ? "ON" : "OFF"}
                      unit=""
                      isLoading={isLoading}
                    />
                    <SensorChip
                      label="Towers"
                      value={currentPayload.Towers === 1 ? "ON" : "OFF"}
                      unit=""
                      isLoading={isLoading}
                    />
                    <SensorChip
                      label="Oil Level"
                      value={currentPayload.Oil_level === 1 ? "OK" : "LOW"}
                      unit=""
                      isLoading={isLoading}
                    />
                  </div>
                </div>
              </Card>
            </>
          )}
        </div>
      </div>

      {/* ── Painel lateral de auditoria (RF-08 / RNF-14) ── */}
      <AlertPanel latest={latest} riskLevel={effectiveRiskLevel} />

      {/* ── Fila de toasts (RF-16 / RNF-34) — dados do hook singleton ── */}
      <AlertToastQueue
        alerts={alerts}
        status={wsStatus}
        onAcknowledge={acknowledge}
      />
    </div>
  );
}
