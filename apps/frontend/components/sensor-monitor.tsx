"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts";
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
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import {
  predict,
  type PredictPayload,
  type PredictResponse,
} from "@/lib/api-client";

// ── Constantes ────────────────────────────────────────────────────────────────

const POLL_INTERVAL_MS = 3_000;
const HISTORY_MAX_POINTS = 24;

const NOMINAL: PredictPayload = {
  TP2: 5.5,
  TP3: 9.2,
  H1: 8.8,
  DV_pressure: 2.1,
  Reservoirs: 8.7,
  Motor_current: 4.2,
  Oil_temperature: 68.5,
  COMP: 1,
  DV_eletric: 0,
  Towers: 1,
  MPG: 1,
  Oil_level: 1,
};

// ── Tipos internos ─────────────────────────────────────────────────────────

interface PressurePoint {
  time: string;
  tp2: number;
  tp3: number;
}

type RiskLevel = "NORMAL" | "ALERTA" | "CRÍTICO";

function getRiskLevel(prob: number): RiskLevel {
  if (prob < 0.3) return "NORMAL";
  if (prob < 0.65) return "ALERTA";
  return "CRÍTICO";
}

function getRiskColor(level: RiskLevel): string {
  return level === "NORMAL"
    ? "hsl(142 71% 45%)"
    : level === "ALERTA"
    ? "hsl(38 92% 50%)"
    : "hsl(0 72% 51%)";
}

// ── Simulação de leituras ──────────────────────────────────────────────────

function jitter(value: number, range: number): number {
  return Math.max(0, value + (Math.random() - 0.5) * range);
}

function simulateReading(base: PredictPayload): PredictPayload {
  return {
    ...base,
    TP2: jitter(base.TP2, 0.35),
    TP3: jitter(base.TP3, 0.25),
    H1: jitter(base.H1, 0.2),
    Motor_current: jitter(base.Motor_current, 0.5),
    Oil_temperature: jitter(base.Oil_temperature, 2.0),
    Reservoirs: jitter(base.Reservoirs, 0.15),
  };
}

function nowLabel(): string {
  return new Date().toLocaleTimeString("pt-BR", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

// ── Gauge SVG (semicírculo) ────────────────────────────────────────────────

interface GaugeProps {
  probability: number;
  riskLevel: RiskLevel;
}

function FailureGauge({ probability, riskLevel }: GaugeProps) {
  const cx = 130;
  const cy = 110;
  const r = 88;
  const strokeW = 14;
  const color = getRiskColor(riskLevel);
  const pct = Math.min(1, Math.max(0, probability));

  // Background arc
  const bgPath = `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`;

  // Value arc (de 180° até 180° - pct*180°)
  function arcPoint(angleDeg: number): [number, number] {
    const rad = (angleDeg * Math.PI) / 180;
    return [cx + r * Math.cos(rad), cy - r * Math.sin(rad)];
  }
  const [startX, startY] = arcPoint(180);
  const endAngle = 180 - pct * 180;
  const [endX, endY] = arcPoint(endAngle);
  const largeArc = pct > 0.5 ? 1 : 0;
  const valuePath =
    pct <= 0.001
      ? ""
      : `M ${startX} ${startY} A ${r} ${r} 0 ${largeArc} 1 ${endX} ${endY}`;

  const percentText = (pct * 100).toFixed(1);

  return (
    <svg
      viewBox={`0 0 ${cx * 2} ${cy + 20}`}
      className="w-full max-w-[260px]"
      aria-label={`Probabilidade de quebra: ${percentText}%`}
    >
      {/* Track */}
      <path
        d={bgPath}
        fill="none"
        stroke="hsl(217 18% 18%)"
        strokeWidth={strokeW}
        strokeLinecap="round"
      />
      {/* Progresso */}
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
      {/* Percentual central */}
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
      {/* Label abaixo */}
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

// ── KPI Card ───────────────────────────────────────────────────────────────

interface KpiCardProps {
  title: string;
  value: string;
  unit: string;
  icon: React.ComponentType<{ className?: string }>;
  trend?: "up" | "down" | "stable";
}

function KpiCard({ title, value, unit, icon: Icon }: KpiCardProps) {
  return (
    <Card className="bg-card border-border">
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

// ── Status Badge ──────────────────────────────────────────────────────────

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

// ── Sensor Chip ───────────────────────────────────────────────────────────

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

const chartConfig = {
  tp2: { label: "TP2 (bar)", color: "hsl(217 91% 60%)" },
  tp3: { label: "TP3 (bar)", color: "hsl(142 71% 45%)" },
} satisfies ChartConfig;

export default function SensorMonitor() {
  const [reading, setReading] = useState<PredictPayload>(NOMINAL);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [history, setHistory] = useState<PressurePoint[]>([]);
  const [isConnected, setIsConnected] = useState(true);
  const [isLoading, setIsLoading] = useState(true);
  const baseRef = useRef<PredictPayload>(NOMINAL);

  const tick = useCallback(async () => {
    const newReading = simulateReading(baseRef.current);
    baseRef.current = newReading;
    setReading(newReading);

    const point: PressurePoint = {
      time: nowLabel(),
      tp2: parseFloat(newReading.TP2.toFixed(2)),
      tp3: parseFloat(newReading.TP3.toFixed(2)),
    };

    setHistory((prev) => {
      const next = [...prev, point];
      return next.length > HISTORY_MAX_POINTS
        ? next.slice(next.length - HISTORY_MAX_POINTS)
        : next;
    });

    try {
      const response = await predict(newReading);
      setResult(response);
      setIsConnected(true);
    } catch {
      setIsConnected(false);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void tick();
    const id = setInterval(() => void tick(), POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [tick]);

  const riskLevel = getRiskLevel(result?.failure_probability ?? 0);
  const probability = result?.failure_probability ?? 0;

  return (
    <div className="flex flex-col gap-6 p-6">
      {/* ── Cabeçalho da seção ── */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-foreground">
            Monitoramento em Tempo Real
          </h1>
          <p className="text-sm text-muted-foreground">
            Compressor de ar · MetroPT-3 · Atualização a cada{" "}
            {POLL_INTERVAL_MS / 1000}s
          </p>
        </div>

        <div className="flex items-center gap-3">
          {!isConnected && (
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
          value={reading.TP2.toFixed(2)}
          unit="bar"
          icon={Gauge}
        />
        <KpiCard
          title="Temperatura Óleo"
          value={reading.Oil_temperature.toFixed(1)}
          unit="°C"
          icon={Thermometer}
        />
        <KpiCard
          title="Corrente Motor"
          value={reading.Motor_current.toFixed(2)}
          unit="A"
          icon={Zap}
        />
        <KpiCard
          title="Reservatório"
          value={reading.Reservoirs.toFixed(2)}
          unit="bar"
          icon={Wind}
        />
      </div>

      {/* ── Linha principal: Gráfico + Gauge ── */}
      <div className="grid gap-4 lg:grid-cols-5">
        {/* Gráfico de área — TP2 / TP3 */}
        <Card className="lg:col-span-3 border-border bg-card">
          <CardHeader className="pb-2 pt-4 px-5">
            <CardTitle className="text-sm font-semibold text-foreground/90">
              Histórico de Pressão
            </CardTitle>
            <p className="text-xs text-muted-foreground">
              Últimas {HISTORY_MAX_POINTS} leituras · TP2 e TP3 (bar)
            </p>
          </CardHeader>
          <CardContent className="px-2 pb-4">
            {history.length < 2 ? (
              <div className="flex h-[200px] items-center justify-center text-sm text-muted-foreground">
                Coletando dados…
              </div>
            ) : (
              <ChartContainer config={chartConfig} className="h-[200px] w-full">
                <AreaChart
                  data={history}
                  margin={{ top: 4, right: 8, bottom: 0, left: -8 }}
                >
                  <defs>
                    <linearGradient id="gradTP2" x1="0" y1="0" x2="0" y2="1">
                      <stop
                        offset="5%"
                        stopColor="hsl(217 91% 60%)"
                        stopOpacity={0.25}
                      />
                      <stop
                        offset="95%"
                        stopColor="hsl(217 91% 60%)"
                        stopOpacity={0}
                      />
                    </linearGradient>
                    <linearGradient id="gradTP3" x1="0" y1="0" x2="0" y2="1">
                      <stop
                        offset="5%"
                        stopColor="hsl(142 71% 45%)"
                        stopOpacity={0.2}
                      />
                      <stop
                        offset="95%"
                        stopColor="hsl(142 71% 45%)"
                        stopOpacity={0}
                      />
                    </linearGradient>
                  </defs>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="hsl(217 18% 18%)"
                    vertical={false}
                  />
                  <XAxis
                    dataKey="time"
                    tick={{ fontSize: 10, fill: "hsl(215 15% 50%)" }}
                    tickLine={false}
                    axisLine={false}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    domain={["auto", "auto"]}
                    tick={{ fontSize: 10, fill: "hsl(215 15% 50%)" }}
                    tickLine={false}
                    axisLine={false}
                    width={32}
                  />
                  <ChartTooltip
                    content={
                      <ChartTooltipContent className="border-border bg-card text-foreground shadow-xl" />
                    }
                  />
                  <Area
                    type="monotone"
                    dataKey="tp2"
                    stroke="hsl(217 91% 60%)"
                    fill="url(#gradTP2)"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4, fill: "hsl(217 91% 60%)" }}
                  />
                  <Area
                    type="monotone"
                    dataKey="tp3"
                    stroke="hsl(142 71% 45%)"
                    fill="url(#gradTP3)"
                    strokeWidth={1.5}
                    dot={false}
                    activeDot={{ r: 3, fill: "hsl(142 71% 45%)" }}
                  />
                </AreaChart>
              </ChartContainer>
            )}
          </CardContent>
        </Card>

        {/* Gauge de falha */}
        <Card className="lg:col-span-2 border-border bg-card">
          <CardHeader className="pb-0 pt-4 px-5">
            <CardTitle className="text-sm font-semibold text-foreground/90">
              Risco de Falha
            </CardTitle>
            <p className="text-xs text-muted-foreground">
              Predição do modelo RandomForest
            </p>
          </CardHeader>
          <CardContent className="flex flex-col items-center gap-4 pb-5 pt-3">
            {isLoading ? (
              <div className="flex h-[140px] w-full items-center justify-center text-sm text-muted-foreground">
                Aguardando modelo…
              </div>
            ) : (
              <>
                <FailureGauge probability={probability} riskLevel={riskLevel} />
                <StatusBadge level={riskLevel} />
                {result && (
                  <p className="text-[10px] text-muted-foreground">
                    Classe {result.predicted_class} ·{" "}
                    {new Date(result.timestamp).toLocaleTimeString("pt-BR")}
                  </p>
                )}
              </>
            )}
          </CardContent>
        </Card>
      </div>

      {/* ── Grade de sensores secundários ── */}
      <Card className="border-border bg-card">
        <CardHeader className="pb-2 pt-4 px-5">
          <CardTitle className="text-sm font-semibold text-foreground/90">
            Painel de Sensores
          </CardTitle>
        </CardHeader>
        <CardContent className="px-5 pb-5">
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
            <SensorChip label="TP3" value={reading.TP3.toFixed(2)} unit="bar" />
            <SensorChip label="H1" value={reading.H1.toFixed(2)} unit="bar" />
            <SensorChip
              label="DV Pressure"
              value={reading.DV_pressure.toFixed(2)}
              unit="bar"
            />
            <SensorChip
              label="COMP"
              value={reading.COMP === 1 ? "ON" : "OFF"}
              unit=""
            />
            <SensorChip
              label="DV Elétric"
              value={reading.DV_eletric === 1 ? "ON" : "OFF"}
              unit=""
            />
            <SensorChip
              label="Towers"
              value={reading.Towers === 1 ? "ON" : "OFF"}
              unit=""
            />
            <SensorChip
              label="MPG"
              value={reading.MPG === 1 ? "ON" : "OFF"}
              unit=""
            />
            <SensorChip
              label="Oil Level"
              value={reading.Oil_level === 1 ? "OK" : "LOW"}
              unit=""
            />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
