"use client";

/**
 * SensorMonitor — componente de composição do dashboard.
 *
 * Toda lógica de estado (polling, histórico, isAnomaly) foi extraída para
 * `useSensorData` (RF-06 / RF-07).  Este componente é responsável apenas
 * por layout e apresentação.
 */

import {
  AlertTriangle,
  CheckCircle2,
  Gauge,
  Thermometer,
  Wind,
  WifiOff,
  XCircle,
  Zap,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { SensorChart } from "@/components/sensor-chart";
import { AlertPanel } from "@/components/alert-panel";
import {
  useSensorData,
  type RiskLevel,
  POLL_INTERVAL_MS,
} from "@/hooks/use-sensor-data";

// ── Sub-componentes de apresentação ───────────────────────────────────────

interface GaugeProps {
  probability: number;
  riskLevel: RiskLevel;
}

// Lógica que garante as cores corretas para cada estado
function riskColor(level: RiskLevel): string {
  return level === "NORMAL"
    ? "hsl(142 71% 45%)" // Verde
    : level === "ALERTA"
    ? "hsl(38 92% 50%)" // Amarelo/Âmbar
    : "hsl(0 72% 51%)"; // Vermelho
}

function FailureGauge({ probability, riskLevel }: GaugeProps) {
  const cx = 130;
  const cy = 110;
  const r = 88;
  const strokeW = 14;
  const color = riskColor(riskLevel);
  // Garante que o valor fique sempre entre 0 e 1 (0% a 100%)
  const pct = Math.min(1, Math.max(0, probability));

  // Função auxiliar para encontrar as coordenadas X e Y exatas no arco
  function getPoint(angleDeg: number) {
    const rad = (angleDeg * Math.PI) / 180;
    return {
      x: cx + r * Math.cos(rad),
      y: cy - r * Math.sin(rad), // Subtrai porque o Y inverte no SVG
    };
  }

  // O velocímetro vai da esquerda (180°) para a direita (0°)
  const startAngle = 180;
  const endAngle = 180 - pct * 180;

  const p1 = getPoint(startAngle); // Início da barra
  const p2 = getPoint(endAngle); // Onde o valor (cor) termina e o fundo cinza começa
  const p3 = getPoint(0); // Fim da barra (lado direito)

  // Caminho do Valor (Colorido) - Vai da esquerda até o valor atual
  // large-arc é sempre 0 porque o ângulo nunca passa de 180°
  const valuePath =
    pct > 0 ? `M ${p1.x} ${p1.y} A ${r} ${r} 0 0 1 ${p2.x} ${p2.y}` : "";

  // Caminho do Fundo (Cinza) - Começa exatamente onde o valor parou e vai até o final
  const bgPath =
    pct < 1 ? `M ${p2.x} ${p2.y} A ${r} ${r} 0 0 1 ${p3.x} ${p3.y}` : "";

  const percentText = (pct * 100).toFixed(1);

  return (
    <svg
      viewBox={`0 0 ${cx * 2} ${cy + 20}`}
      className="w-full max-w-[260px]"
      aria-label={`Probabilidade de quebra: ${percentText}%`}
    >
      {/* Desenha a trilha cinza contígua (apenas a parte vazia) */}
      {bgPath && (
        <path
          d={bgPath}
          fill="none"
          stroke="hsl(217 18% 18%)"
          strokeWidth={strokeW}
          strokeLinecap="round"
        />
      )}

      {/* Desenha a trilha de cor (apenas a parte cheia) */}
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
        fill="hsl(215 15% 50%)"
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
}

function KpiCard({ title, value, unit, icon: Icon }: KpiCardProps) {
  return (
    <Card className="border-border bg-card">
      <CardContent className="p-4">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              {title}
            </p>
            <p className="mt-1.5 text-2xl font-bold tabular-nums text-foreground">
              {value}
              <span className="ml-1 text-sm font-normal text-muted-foreground">
                {unit}
              </span>
            </p>
          </div>
          <div className="rounded-lg bg-primary/10 p-2">
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
}

function SensorChip({ label, value, unit }: SensorChipProps) {
  return (
    <div className="flex flex-col gap-0.5 rounded-md border border-border bg-card/60 px-3 py-2">
      <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
      <span className="text-sm font-bold tabular-nums text-foreground">
        {value}
        <span className="ml-0.5 text-xs font-normal text-muted-foreground">
          {unit}
        </span>
      </span>
    </div>
  );
}

// ── Componente principal ───────────────────────────────────────────────────

export default function SensorMonitor() {
  // Toda lógica de dados vem do hook (RF-06 polling 5s, RF-07 isAnomaly)
  const {
    history,
    latest,
    currentPayload,
    isLoading,
    isAnomaly,
    riskLevel,
    error,
  } = useSensorData();

  const probability = latest?.failure_probability ?? 0;
  const isOffline = error !== null && !isLoading;

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
                Compressor de ar · MetroPT-3 · Polling a cada{" "}
                {POLL_INTERVAL_MS / 1000}s
              </p>
            </div>

            <div className="flex items-center gap-3">
              {isOffline && (
                <Badge
                  variant="outline"
                  className="gap-1.5 border-destructive/40 bg-destructive/10 text-destructive"
                >
                  <WifiOff className="h-3 w-3" />
                  Backend offline
                </Badge>
              )}
              {!isLoading && <StatusBadge level={riskLevel} />}
            </div>
          </div>

          {/* ── KPI Cards ── */}
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <KpiCard
              title="Pressão TP2"
              value={currentPayload.TP2.toFixed(2)}
              unit="bar"
              icon={Gauge}
            />
            <KpiCard
              title="Temperatura Óleo"
              value={currentPayload.Oil_temperature.toFixed(1)}
              unit="°C"
              icon={Thermometer}
            />
            <KpiCard
              title="Corrente Motor"
              value={currentPayload.Motor_current.toFixed(2)}
              unit="A"
              icon={Zap}
            />
            <KpiCard
              title="Reservatório"
              value={currentPayload.Reservoirs.toFixed(2)}
              unit="bar"
              icon={Wind}
            />
          </div>

          {/* ── Gráficos (RF-06 / RF-07) + Gauge ── */}
          <div className="grid gap-4 lg:grid-cols-5">
            {/* SensorChart: 4 séries, destaque de anomalia */}
            <SensorChart
              history={history}
              isAnomaly={isAnomaly}
              riskLevel={riskLevel}
              className="lg:col-span-3"
            />

            {/* Gauge de probabilidade de falha */}
            <Card className="border-border bg-card lg:col-span-2">
              <div className="flex flex-col items-center gap-4 p-5">
                <p className="self-start text-sm font-semibold text-foreground/90">
                  Risco de Falha
                </p>
                <p className="self-start text-xs text-muted-foreground">
                  Predição do modelo RandomForest
                </p>

                {isLoading ? (
                  <div className="flex h-[140px] w-full items-center justify-center text-sm text-muted-foreground">
                    Aguardando modelo…
                  </div>
                ) : (
                  <>
                    <FailureGauge
                      probability={probability}
                      riskLevel={riskLevel}
                    />
                    <StatusBadge level={riskLevel} />
                    {latest && (
                      <p className="text-[10px] text-muted-foreground">
                        Classe {latest.predicted_class} ·{" "}
                        {new Date(latest.timestamp).toLocaleTimeString("pt-BR")}
                      </p>
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
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-8">
                <SensorChip
                  label="TP3"
                  value={currentPayload.TP3.toFixed(2)}
                  unit="bar"
                />
                <SensorChip
                  label="H1"
                  value={currentPayload.H1.toFixed(2)}
                  unit="bar"
                />
                <SensorChip
                  label="DV Press."
                  value={currentPayload.DV_pressure.toFixed(2)}
                  unit="bar"
                />
                <SensorChip
                  label="Reserv."
                  value={currentPayload.Reservoirs.toFixed(2)}
                  unit="bar"
                />
                <SensorChip
                  label="COMP"
                  value={currentPayload.COMP === 1 ? "ON" : "OFF"}
                  unit=""
                />
                <SensorChip
                  label="DV Elétric"
                  value={currentPayload.DV_eletric === 1 ? "ON" : "OFF"}
                  unit=""
                />
                <SensorChip
                  label="Towers"
                  value={currentPayload.Towers === 1 ? "ON" : "OFF"}
                  unit=""
                />
                <SensorChip
                  label="Oil Level"
                  value={currentPayload.Oil_level === 1 ? "OK" : "LOW"}
                  unit=""
                />
              </div>
            </div>
          </Card>
        </div>
        {/* fim .flex-col.gap-6.p-6 */}
      </div>
      {/* fim .flex-1.overflow-y-auto */}

      {/* ── Painel lateral de auditoria (RF-08 / RNF-14) ── */}
      <AlertPanel latest={latest} riskLevel={riskLevel} />
    </div>
  );
}
