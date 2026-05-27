"use client";

import { memo, useMemo } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  ClipboardList,
  Gauge,
  Loader2,
  ServerCrash,
  Thermometer,
  Trash2,
  WifiOff,
  XCircle,
  Zap,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertToastQueue } from "@/components/alert-toast-queue";
import { ConnectionStatus } from "@/components/connection-status";
import {
  useSensorData,
  getRiskLevel,
  type RiskLevel,
  type SensorDataPoint,
} from "@/hooks/use-sensor-data";
import { useAlertWebSocket } from "@/hooks/use-alert-websocket";
import { usePredictionHistory } from "@/hooks/use-prediction-history";
import type { PredictResponse } from "@/lib/api-client";
import { isCriticalProb } from "@/lib/risk-thresholds";
import { cn } from "@/lib/utils";

// ── Paleta de cores (light-mode) ──────────────────────────────────────────

const C = {
  tp2: "#3b82f6",
  tp3: "#10b981",
  current: "#8b5cf6",
  temp: "#f97316",
  anomaly: "#ef4444",
  h1: "#3b82f6",
  dvp: "#8b5cf6",
  res: "#10b981",
  off: "#cbd5e1",
  noload: "#93c5fd",
  load: "#6ee7b7",
  start: "#fcd34d",
} as const;

const GRID_STROKE = "hsl(var(--border))";
const AXIS_TICK = {
  fontSize: 10,
  fill: "hsl(var(--muted-foreground))",
} as const;

// ── Helpers ────────────────────────────────────────────────────────────────

function riskColor(level: RiskLevel): string {
  return level === "NORMAL"
    ? "hsl(142 71% 45%)"
    : level === "ALERTA"
    ? "hsl(38 92% 50%)"
    : "hsl(0 72% 51%)";
}

/** Constrói caminhos SVG para um arco semi-circular de gauges. */
function buildArcPaths(
  cx: number,
  cy: number,
  r: number,
  pct: number,
): { bg: string; fg: string } {
  const bg = `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`;
  if (pct <= 0) return { bg, fg: "" };
  if (pct >= 1) return { bg, fg: bg };
  const endDeg = 180 * (1 - pct);
  const rad = (endDeg * Math.PI) / 180;
  const ex = cx + r * Math.cos(rad);
  const ey = cy - r * Math.sin(rad);
  return { bg, fg: `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${ex} ${ey}` };
}

// ── Sparkline ─────────────────────────────────────────────────────────────

interface SparklineProps {
  data: SensorDataPoint[];
  dataKey: keyof SensorDataPoint;
  color: string;
}

const Sparkline = memo(function Sparkline({
  data,
  dataKey,
  color,
}: SparklineProps) {
  return (
    <ResponsiveContainer width="100%" height={36}>
      <LineChart data={data} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
        <Line
          type="monotone"
          dataKey={dataKey as string}
          stroke={color}
          strokeWidth={1.5}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
});

// ── KPI Card com sparkline ─────────────────────────────────────────────────

interface SparkKpiCardProps {
  title: string;
  value: string;
  unit: string;
  icon: React.ComponentType<{
    className?: string;
    style?: React.CSSProperties;
  }>;
  sparkData: SensorDataPoint[];
  sparkKey: keyof SensorDataPoint;
  sparkColor: string;
  isLoading?: boolean;
  alertColor?: boolean;
}

const SparkKpiCard = memo(function SparkKpiCard({
  title,
  value,
  unit,
  icon: Icon,
  sparkData,
  sparkKey,
  sparkColor,
  isLoading,
  alertColor,
}: SparkKpiCardProps) {
  return (
    <Card className="border-slate-200 bg-card">
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              {title}
            </p>
            {isLoading ? (
              <Skeleton className="mt-2 h-7 w-20" />
            ) : (
              <p
                className={cn(
                  "mt-1 text-2xl font-bold tabular-nums",
                  alertColor ? "text-destructive" : "text-foreground",
                )}
              >
                {value}
                <span className="ml-1 text-sm font-normal text-muted-foreground">
                  {unit}
                </span>
              </p>
            )}
          </div>
          <div
            className="shrink-0 rounded-lg p-2"
            style={{ background: `${sparkColor}18` }}
          >
            <Icon className="h-4 w-4" style={{ color: sparkColor }} />
          </div>
        </div>
        {!isLoading && sparkData.length > 1 && (
          <div className="mt-2">
            <Sparkline data={sparkData} dataKey={sparkKey} color={sparkColor} />
          </div>
        )}
      </CardContent>
    </Card>
  );
});

// ── Gráfico principal — AreaChart TP2 + TP3 ────────────────────────────────

interface MainAreaChartProps {
  data: SensorDataPoint[];
  isLive: boolean;
  riskLevel: RiskLevel;
}

interface AreaTooltipEntry {
  name: string;
  value: number;
  color: string;
}

interface AreaTooltipProps {
  active?: boolean;
  label?: string;
  payload?: AreaTooltipEntry[];
}

const AreaChartTooltip = memo(function AreaChartTooltip({
  active,
  label,
  payload,
}: AreaTooltipProps) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 shadow-md">
      <p className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
        {label}
      </p>
      {payload.map((e) => (
        <div key={e.name} className="flex items-center gap-2 text-xs">
          <span
            className="h-2 w-2 rounded-full"
            style={{ background: e.color }}
          />
          <span className="text-muted-foreground">{e.name}</span>
          <span className="ml-auto font-bold tabular-nums text-foreground">
            {e.value} bar
          </span>
        </div>
      ))}
    </div>
  );
});

const AREA_TOOLTIP = <AreaChartTooltip />;

const MainAreaChart = memo(function MainAreaChart({
  data,
  isLive,
  riskLevel,
}: MainAreaChartProps) {
  const isEmpty = data.length < 2;

  return (
    <Card
      className={cn(
        "border-slate-200 transition-colors duration-700",
        riskLevel === "ALERTA" && "border-amber-300",
        riskLevel === "CRÍTICO" && "border-red-300",
      )}
    >
      <CardHeader className="px-5 pb-2 pt-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-sm font-semibold text-foreground/90">
            <Activity className="h-4 w-4" style={{ color: C.tp2 }} />
            Pressão em Tempo Real — TP2 &amp; TP3
            {isLive && (
              <span className="flex items-center gap-1">
                <span className="relative flex h-2 w-2">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-400 opacity-75" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-green-500" />
                </span>
                <span className="text-[10px] font-semibold uppercase tracking-wider text-green-600">
                  Live
                </span>
              </span>
            )}
          </CardTitle>
          <div className="flex items-center gap-3">
            {(["TP2", "TP3"] as const).map((k) => (
              <div key={k} className="flex items-center gap-1.5">
                <span
                  className="inline-block h-2 w-4 rounded-full"
                  style={{ background: k === "TP2" ? C.tp2 : C.tp3 }}
                />
                <span className="text-[10px] text-muted-foreground">
                  {k} (bar)
                </span>
              </div>
            ))}
          </div>
        </div>
      </CardHeader>
      <CardContent className="px-4 pb-4">
        {isEmpty ? (
          <div className="flex h-[180px] items-center justify-center text-sm text-muted-foreground">
            Coletando dados…
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart
              data={data}
              margin={{ top: 4, right: 8, bottom: 0, left: -10 }}
            >
              <defs>
                <linearGradient id="gradTP2" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={C.tp2} stopOpacity={0.25} />
                  <stop offset="100%" stopColor={C.tp2} stopOpacity={0} />
                </linearGradient>
                <linearGradient id="gradTP3" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={C.tp3} stopOpacity={0.25} />
                  <stop offset="100%" stopColor={C.tp3} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke={GRID_STROKE}
                vertical={false}
              />
              <XAxis
                dataKey="time"
                tick={AXIS_TICK}
                tickLine={false}
                axisLine={false}
                interval="preserveStartEnd"
              />
              <YAxis
                domain={[0, 12]}
                tick={AXIS_TICK}
                tickLine={false}
                axisLine={false}
                width={28}
              />
              <RechartsTooltip content={AREA_TOOLTIP} />
              <Area
                type="monotone"
                dataKey="TP2"
                name="TP2"
                stroke={C.tp2}
                strokeWidth={2}
                fill="url(#gradTP2)"
                dot={false}
                activeDot={{ r: 4, fill: C.tp2 }}
                isAnimationActive={false}
              />
              <Area
                type="monotone"
                dataKey="TP3"
                name="TP3"
                stroke={C.tp3}
                strokeWidth={2}
                fill="url(#gradTP3)"
                dot={false}
                activeDot={{ r: 4, fill: C.tp3 }}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
});

// ── Painel de Sinais Booleanos ─────────────────────────────────────────────

// mode:
//   "normal"      — verde quando ON, cinza quando OFF
//   "alertWhenOn" — vermelho quando ON (ex: LPS, Pressure Switch)
//   "alertWhenOff"— vermelho quando OFF (ex: Oil Level)
type BoolSignalMode = "normal" | "alertWhenOn" | "alertWhenOff";

interface BoolSignalProps {
  label: string;
  value: number;
  mode?: BoolSignalMode;
  isCount?: boolean; // para Caudal Impulses
}

function BoolSignal({
  label,
  value,
  mode = "normal",
  isCount = false,
}: BoolSignalProps) {
  const isOn = isCount ? value > 0 : value === 1;
  const isAlert =
    mode === "alertWhenOn" ? isOn : mode === "alertWhenOff" ? !isOn : false;
  const isSuccess = !isAlert && isOn;

  const dotCls = isAlert
    ? "bg-red-500 shadow-sm shadow-red-400/40 animate-pulse"
    : isSuccess
    ? "bg-emerald-500 shadow-sm shadow-emerald-400/30"
    : "bg-slate-300";

  const valueCls = isAlert
    ? "text-red-700 font-extrabold"
    : isSuccess
    ? "text-emerald-600"
    : "text-slate-400";

  const displayValue = isCount
    ? value > 0
      ? String(value)
      : "—"
    : isOn
    ? "ON"
    : "OFF";

  return (
    <div
      className={cn(
        "flex items-center justify-between rounded-md border p-2 transition-colors duration-200",
        isAlert ? "border-red-300 bg-red-50" : "border-slate-200 bg-white",
      )}
    >
      <span
        className={cn(
          "truncate text-[10px] font-semibold uppercase tracking-wide",
          isAlert ? "text-red-700" : "text-slate-500",
        )}
      >
        {label}
      </span>
      <div className="flex shrink-0 items-center gap-1">
        <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", dotCls)} />
        <span className={cn("text-[10px] font-bold tabular-nums", valueCls)}>
          {displayValue}
        </span>
      </div>
    </div>
  );
}

interface BooleanPanelProps {
  COMP: number;
  DV_eletric: number;
  Towers: number;
  MPG: number;
  LPS: number;
  Pressure_switch: number;
  Oil_level: number;
  Caudal_impulses: number;
}

const BOOL_SIGNALS: Array<{
  key: keyof BooleanPanelProps;
  label: string;
  mode: BoolSignalMode;
  isCount?: boolean;
}> = [
  { key: "COMP", label: "COMP", mode: "normal" },
  { key: "DV_eletric", label: "DV Elec", mode: "normal" },
  { key: "Towers", label: "TOWERS", mode: "normal" },
  { key: "MPG", label: "MPG", mode: "normal" },
  { key: "LPS", label: "LPS", mode: "alertWhenOn" },
  { key: "Pressure_switch", label: "Press. SW", mode: "alertWhenOn" },
  { key: "Oil_level", label: "Oil Level", mode: "alertWhenOff" },
  {
    key: "Caudal_impulses",
    label: "Caudal Imp",
    mode: "normal",
    isCount: true,
  },
];

const BooleanPanel = memo(function BooleanPanel(props: BooleanPanelProps) {
  return (
    <Card className="flex h-full flex-col border-slate-200">
      <CardHeader className="shrink-0 px-4 pb-2 pt-4">
        <CardTitle className="text-sm font-semibold text-foreground/90">
          Sinais Digitais
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col p-3 pt-0">
        <div className="grid h-full grid-cols-2 grid-rows-4 gap-2">
          {BOOL_SIGNALS.map(({ key, label, mode, isCount }) => (
            <BoolSignal
              key={key}
              label={label}
              value={props[key]}
              mode={mode}
              isCount={isCount}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  );
});

// ── Donut — Estado Operacional ─────────────────────────────────────────────

// Limites calibrados com dados reais do MetroPT-3:
//   0.04A → Desligado  |  3.76A → Sem Carga  |  7A → Com Carga  |  9A → Partida
const OP_THRESHOLDS = { off: 1.0, noLoad: 5.5, load: 8.5 } as const;
const OP_STATES = [
  { name: "Desligado", color: C.off },
  { name: "Sem Carga", color: C.noload },
  { name: "Com Carga", color: C.load },
  { name: "Partida", color: C.start },
] as const;

interface OpSlice {
  name: string;
  color: string;
  value: number;
}

function computeOpState(history: SensorDataPoint[]): OpSlice[] {
  const counts: [number, number, number, number] = [0, 0, 0, 0];
  for (const p of history) {
    const c = Number(p.Motor_current);
    if (!isFinite(c)) continue; // descarta leituras corrompidas
    if (c < OP_THRESHOLDS.off) counts[0]++;
    else if (c < OP_THRESHOLDS.noLoad) counts[1]++;
    else if (c < OP_THRESHOLDS.load) counts[2]++;
    else counts[3]++;
  }
  return OP_STATES.map((s, i): OpSlice => ({ ...s, value: counts[i] })).filter(
    (d) => d.value > 0,
  );
}

// Formatter seguro para o tooltip do PieChart.
// Recebe `value` (número de pontos) e `total` via closure.
function makeDonutFormatter(total: number) {
  return (
    raw: string | number | readonly (string | number)[] | undefined,
    name: string | number | undefined,
  ): [string, string] => {
    const n =
      raw === undefined
        ? 0
        : Array.isArray(raw)
        ? Number(raw[0]) || 0
        : Number(raw) || 0;
    const pct = total > 0 ? ((n / total) * 100).toFixed(1) : "0.0";
    const label = String(name ?? "");
    return [`${n} pts · ${pct}%`, label];
  };
}

interface OperationalDonutProps {
  history: SensorDataPoint[];
  isLoading: boolean;
}

const OperationalDonut = memo(function OperationalDonut({
  history,
  isLoading,
}: OperationalDonutProps) {
  const data = useMemo(() => computeOpState(history), [history]);
  const total = useMemo(() => data.reduce((s, d) => s + d.value, 0), [data]);
  const donutFormatter = useMemo(() => makeDonutFormatter(total), [total]);

  return (
    <Card className="border-slate-200">
      <CardHeader className="px-4 pb-2 pt-4">
        <CardTitle className="text-sm font-semibold text-foreground/90">
          Estado Operacional
        </CardTitle>
        <p className="text-[10px] text-muted-foreground">
          Derivado da corrente do motor
        </p>
      </CardHeader>
      <CardContent className="flex flex-col items-center gap-3 px-4 pb-4">
        {isLoading || data.length === 0 ? (
          <div className="flex h-[140px] items-center justify-center">
            {isLoading ? (
              <Skeleton className="h-[120px] w-[120px] rounded-full" />
            ) : (
              <p className="text-xs text-muted-foreground">Sem dados</p>
            )}
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={140}>
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={58}
                paddingAngle={3}
                dataKey="value"
                isAnimationActive={false}
              >
                {data.map((entry, i) => (
                  <Cell key={`cell-${i}`} fill={entry.color} />
                ))}
              </Pie>
              <RechartsTooltip
                formatter={donutFormatter}
                contentStyle={{
                  backgroundColor: "#ffffff",
                  color: "#0f172a",
                  border: "1px solid #e2e8f0",
                  borderRadius: "8px",
                  fontSize: "12px",
                  fontWeight: "500",
                }}
                labelStyle={{ color: "#0f172a", fontWeight: "600" }}
                itemStyle={{ color: "#334155" }}
              />
            </PieChart>
          </ResponsiveContainer>
        )}
        <div className="grid w-full grid-cols-2 gap-x-3 gap-y-1">
          {OP_STATES.map((s) => (
            <div key={s.name} className="flex items-center gap-1.5">
              <span
                className="h-2 w-2 shrink-0 rounded-full"
                style={{ background: s.color }}
              />
              <span className="truncate text-[10px] text-muted-foreground">
                {s.name}
              </span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
});

// ── Log de Eventos ─────────────────────────────────────────────────────────

interface EventLogProps {
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

const EventLog = memo(function EventLog({ latest, riskLevel }: EventLogProps) {
  const { history, isMounted, clearHistory } = usePredictionHistory(latest);

  return (
    <Card
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

// ── Manômetros de Pressão (gauge SVG semi-circular) ────────────────────────

interface PressureGaugeProps {
  label: string;
  value: number;
  max: number;
  unit: string;
  color: string;
}

function PressureGauge({ label, value, max, unit, color }: PressureGaugeProps) {
  const pct = Math.min(1, Math.max(0, value / max));
  const cx = 50;
  const cy = 46;
  const r = 34;
  const sw = 7;
  const { bg, fg } = buildArcPaths(cx, cy, r, pct);

  return (
    <div className="flex flex-col items-center gap-1">
      {/* viewBox 100×72 dá folga absoluta para descida das letras
          ("bar" tem descender visual mesmo sem 'g'/'p'/'q'); somado a
          overflow-visible no <svg>, garante que nada é cortado mesmo
          quando o navegador aplica antialiasing agressivo. */}
      <svg
        viewBox="0 0 100 72"
        className="w-full max-w-[96px] overflow-visible"
      >
        <path
          d={bg}
          fill="none"
          stroke="rgb(226 232 240)"
          strokeWidth={sw}
          strokeLinecap="round"
        />
        {fg && (
          <path
            d={fg}
            fill="none"
            stroke={color}
            strokeWidth={sw}
            strokeLinecap="round"
          />
        )}
        <text
          x={cx}
          y={cy - 4}
          textAnchor="middle"
          fill="rgb(15 23 42)"
          fontSize="11"
          fontWeight="700"
          fontFamily="var(--font-geist-sans, sans-serif)"
        >
          {value.toFixed(1)}
        </text>
        <text
          x={cx}
          y={cy + 14}
          textAnchor="middle"
          dominantBaseline="middle"
          fill="rgb(100 116 139)"
          fontSize="9"
          fontFamily="var(--font-geist-sans, sans-serif)"
        >
          {unit}
        </text>
      </svg>
      <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
      <div className="w-full max-w-[80px]">
        <Progress
          value={pct * 100}
          className="h-1"
          style={
            {
              "--progress-bg": color,
            } as React.CSSProperties
          }
        />
      </div>
    </div>
  );
}

interface PressureRadialsProps {
  H1: number;
  DV_pressure: number;
  Reservoirs: number;
  isLoading: boolean;
}

const PressureRadials = memo(function PressureRadials({
  H1,
  DV_pressure,
  Reservoirs,
  isLoading,
}: PressureRadialsProps) {
  return (
    <Card className="border-slate-200">
      <CardHeader className="px-4 pb-2 pt-4">
        <CardTitle className="text-sm font-semibold text-foreground/90">
          Pressões Secundárias
        </CardTitle>
        <p className="text-[10px] text-muted-foreground">
          H1 · DV Pressure · Reservatório
        </p>
      </CardHeader>
      <CardContent className="px-4 pb-6">
        {isLoading ? (
          <div className="flex items-center justify-around gap-2">
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-[100px] w-[80px] rounded-lg" />
            ))}
          </div>
        ) : (
          <div className="flex items-center justify-around gap-2">
            <PressureGauge
              label="H1"
              value={H1}
              max={11}
              unit="bar"
              color={C.h1}
            />
            <PressureGauge
              label="DV Press"
              value={DV_pressure}
              max={4}
              unit="bar"
              color={C.dvp}
            />
            <PressureGauge
              label="Reserv."
              value={Reservoirs}
              max={12}
              unit="bar"
              color={C.res}
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
});

// ── Banner modo degradado ──────────────────────────────────────────────────

function DegradedModeBanner({ attempt }: { attempt: number }) {
  return (
    <div
      role="status"
      aria-live="polite"
      className="flex items-center gap-3 rounded-xl border border-amber-400/40 bg-amber-400/8 px-4 py-3 text-amber-700"
    >
      <WifiOff className="h-4 w-4 shrink-0" />
      <div className="min-w-0 flex-1">
        <p className="text-sm font-bold">Modo Degradado</p>
        <p className="text-xs text-amber-600">
          Exibindo últimos dados conhecidos · Reconexão {attempt} em curso…
        </p>
      </div>
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-500" />
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
        {/* ── Cabeçalho ────────────────────────────────────────────
            CRÍTICO  → fundo vermelho suave + borda vermelha (atenção máxima)
            ALERTA   → fundo âmbar suave + borda âmbar (atenção intermediária)
            NORMAL   → branco neutro */}
        <div
          className={cn(
            "flex flex-wrap items-center justify-between gap-3 rounded-lg border px-4 py-3 transition-colors",
            isCriticalState
              ? "border-red-500 bg-red-50"
              : isAlertState
              ? "border-amber-500 bg-amber-50"
              : "border-slate-200 bg-white",
          )}
        >
          <div>
            <h1
              className={cn(
                "text-xl font-bold tracking-tight",
                isCriticalState
                  ? "text-red-900"
                  : isAlertState
                  ? "text-amber-900"
                  : "text-slate-900",
              )}
            >
              APU-Trem-042
            </h1>
            <p
              className={cn(
                "text-sm",
                isCriticalState
                  ? "text-red-700"
                  : isAlertState
                  ? "text-amber-700"
                  : "text-slate-500",
              )}
            >
              Compressor MetroPT-3 ·{" "}
              {isLive
                ? "Streaming em tempo real"
                : sseStatus === "reconnecting"
                ? "Reconectando…"
                : "Conectando…"}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
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
            {!isLoading && (
              <Badge
                variant="outline"
                className={cn(
                  "gap-1.5 font-semibold",
                  isCriticalState
                    ? "animate-pulse border-red-500 bg-red-600 text-white"
                    : isAlertState
                    ? "animate-pulse border-amber-400 bg-amber-100 text-amber-800"
                    : "border-emerald-400/40 bg-emerald-50 text-emerald-700",
                )}
              >
                {isCriticalState ? (
                  <>
                    <span className="relative flex h-2 w-2">
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-white opacity-75" />
                      <span className="relative inline-flex h-2 w-2 rounded-full bg-white" />
                    </span>
                    FALHA CRÍTICA
                  </>
                ) : isAlertState ? (
                  <>
                    <AlertTriangle className="h-3.5 w-3.5" />
                    ALERTA
                  </>
                ) : (
                  <>
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    OPERACIONAL
                  </>
                )}
              </Badge>
            )}
          </div>
        </div>

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
