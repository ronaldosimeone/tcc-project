"use client";
/* eslint-disable react-hooks/set-state-in-effect --
   O effect que mantém `latencyHistory` é o caso canónico de "sincronizar
   estado com fonte externa" (telemetria SSE chega via prop). Sem setState
   no effect, o histórico nunca seria actualizado. */

/**
 * KPIs do cockpit operacional — 4 cards industriais com sparkline.
 *
 * Cada card mantém: título + valor grande + ícone tonal + sparkline 40–50px
 * (sem eixos, sem grid). A cor da sparkline reflecte a saúde do KPI:
 * verde = bom, âmbar = atenção, rosa = crítico, slate = neutro.
 *
 * Card 4 ("Modelo de IA Ativo") lê o estado real via listModels(); a
 * sparkline aí é neutra (confiança do modelo num histórico mock).
 */

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ComponentType,
} from "react";
import { AlertOctagon, Gauge, HeartPulse, ShieldAlert } from "lucide-react";
import { Area, AreaChart, ResponsiveContainer } from "recharts";

import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { getRiskLevel, type RiskLevel } from "@/lib/risk-thresholds";
import { cn } from "@/lib/utils";

/**
 * Telemetria de latência por frame WS. Recebemos um **objeto** (e não um
 * primitivo) propositadamente: cada novo frame WS — mesmo com latência
 * numericamente idêntica ao anterior (ex.: `42 → 42`) — gera uma referência
 * nova, o que faz o `useEffect` em FleetKPIs disparar. Se passássemos apenas
 * `liveLatency: number`, o React deduplicaria via Object.is e perderíamos
 * leituras consecutivas com o mesmo valor.
 */
export interface LatencyTelemetry {
  /** ID único do frame WS — chave de invalidação do effect. */
  messageId: string;
  /** Latência da inferência em ms. */
  latencyMs: number;
}

interface FleetKPIsProps {
  /** Probabilidade de falha do ativo ao vivo (0–1). */
  liveProbability: number;
  /** Nível de risco efectivo do ativo ao vivo. */
  effectiveRiskLevel: RiskLevel;
  /** Última leitura de latência reportada pelo backend. `null` → ainda nenhuma. */
  latencyTelemetry: LatencyTelemetry | null;
  isLoading: boolean;
}

/** Tamanho da janela deslizante da sparkline de latência. */
const LATENCY_HISTORY_SIZE = 24;

// ── Sparkline genérica ──────────────────────────────────────────────────────

type SparkTone = "ok" | "warn" | "danger" | "neutral";

const SPARK_COLOR: Record<SparkTone, string> = {
  ok: "#10b981",
  warn: "#f59e0b",
  danger: "#f43f5e",
  neutral: "#64748b",
};

interface KpiSparklineProps {
  data: ReadonlyArray<number>;
  tone: SparkTone;
}

/** Mini AreaChart — 44px, sem eixos, sem grid. Decorativo. */
function KpiSparkline({ data, tone }: KpiSparklineProps) {
  const color = SPARK_COLOR[tone];
  const gradId = `kpi-spark-${tone}`;
  const chartData = useMemo(() => data.map((v, i) => ({ i, v })), [data]);

  return (
    <ResponsiveContainer width="100%" height={44}>
      <AreaChart
        data={chartData}
        margin={{ top: 4, right: 0, left: 0, bottom: 0 }}
      >
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.32} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <Area
          type="monotone"
          dataKey="v"
          stroke={color}
          strokeWidth={1.75}
          fill={`url(#${gradId})`}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

// ── Shell de card ───────────────────────────────────────────────────────────

interface KpiShellProps {
  title: string;
  value: React.ReactNode;
  /** Linha de contexto opcional abaixo do valor (ex.: percentil). */
  subtitle?: string;
  icon: ComponentType<{ className?: string }>;
  iconTone: SparkTone;
  /** Série da sparkline. Omitir para cards que renderizam um visual customizado. */
  spark?: ReadonlyArray<number>;
  sparkTone?: SparkTone;
  /**
   * Visual alternativo no rodapé do card (ex.: Andon Board). Quando provido,
   * substitui a sparkline. `spark` e `children` são mutuamente exclusivos —
   * `children` vence se ambos forem passados.
   */
  children?: React.ReactNode;
  isLoading?: boolean;
}

const TONE_BG: Record<SparkTone, string> = {
  neutral: "bg-slate-100 text-slate-600",
  ok: "bg-emerald-50 text-emerald-600",
  warn: "bg-amber-50 text-amber-600",
  danger: "bg-rose-50 text-rose-600",
};

function KpiShell({
  title,
  value,
  subtitle,
  icon: Icon,
  iconTone,
  spark,
  sparkTone,
  children,
  isLoading,
}: KpiShellProps) {
  // children > sparkline. Cards podem optar por nenhum visual de rodapé
  // simplesmente omitindo ambos.
  const footerVisual: React.ReactNode = children ? (
    <div className="mt-3 h-11">{children}</div>
  ) : spark ? (
    <div className="mt-2 -mb-1 h-11">
      <KpiSparkline data={spark} tone={sparkTone ?? "neutral"} />
    </div>
  ) : null;
  return (
    <Card className="border border-slate-200 bg-white shadow-sm ring-0">
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-500">
              {title}
            </p>
            {isLoading ? (
              <Skeleton className="mt-2 h-7 w-20" />
            ) : (
              <>
                <div className="mt-1 font-mono text-2xl font-bold tracking-tight tabular-nums text-slate-900">
                  {value}
                </div>
                {subtitle && (
                  <p className="mt-0.5 text-[10px] text-slate-500">
                    {subtitle}
                  </p>
                )}
              </>
            )}
          </div>
          <div className={cn("shrink-0 rounded-lg p-2", TONE_BG[iconTone])}>
            <Icon className="h-4 w-4" />
          </div>
        </div>
        {footerVisual}
      </CardContent>
    </Card>
  );
}

// ── Séries mock para as 4 sparklines (24 pontos = 1 ponto/hora) ─────────────

const SPARK_HEALTH = [
  91, 92, 92, 93, 91, 90, 91, 92, 93, 94, 94, 93, 94, 95, 94, 93, 92, 93, 94,
  94, 95, 95, 94, 94,
];

const SPARK_ANOMALY = [
  12, 14, 11, 13, 18, 20, 24, 22, 27, 31, 28, 33, 36, 34, 38, 41, 39, 42, 44,
  43, 45, 44, 46, 46,
];

// ── Componente ──────────────────────────────────────────────────────────────

export default function FleetKPIs({
  liveProbability,
  effectiveRiskLevel,
  latencyTelemetry,
  isLoading,
}: FleetKPIsProps) {
  // ── KPI 1: Saúde Global ──
  const liveHealth = Math.round((1 - liveProbability) * 100);
  const fleetHealth = Math.round((liveHealth + 95 + 92 + 96 + 88) / 5);

  // ── KPI 4: Latência da inferência ──
  // Histórico inicia VAZIO — só passa a renderizar valor numérico quando o
  // backend reporta a primeira medição real. Evita o "40 ms" fantasma.
  const [latencyHistory, setLatencyHistory] = useState<number[]>([]);
  // De-dupe explícito por messageId: garante que cada frame WS gere
  // exactamente uma actualização do histórico, mesmo em StrictMode (que
  // executa effects duas vezes em dev).
  const lastMessageId = useRef<string | null>(null);

  useEffect(() => {
    if (!latencyTelemetry) return;
    if (latencyTelemetry.messageId === lastMessageId.current) return;
    if (!Number.isFinite(latencyTelemetry.latencyMs)) return;

    lastMessageId.current = latencyTelemetry.messageId;
    const sample = Math.max(0, Math.round(latencyTelemetry.latencyMs));
    setLatencyHistory((prev) => {
      const next = [...prev, sample];
      // Mantém a janela deslizante: descarta o mais antigo se passou do tecto.
      return next.length > LATENCY_HISTORY_SIZE
        ? next.slice(next.length - LATENCY_HISTORY_SIZE)
        : next;
    });
  }, [latencyTelemetry]);

  const hasLatency = latencyHistory.length > 0;
  const currentLatencyMs = hasLatency
    ? latencyHistory[latencyHistory.length - 1]
    : null;

  // Mapeia a faixa oficial (NORMAL/ALERTA/CRÍTICO) para o tom visual usado
  // tanto no ícone como na sparkline. Único ponto de verdade — qualquer
  // mudança nos limiares (lib/risk-thresholds) propaga automaticamente.
  const anomalyTone: SparkTone =
    getRiskLevel(liveProbability) === "CRÍTICO"
      ? "danger"
      : getRiskLevel(liveProbability) === "ALERTA"
      ? "warn"
      : "ok";

  // ── KPI 2: Ativos em Alerta ──
  const liveIsCritical = effectiveRiskLevel === "CRÍTICO";
  const liveIsAlert = effectiveRiskLevel === "ALERTA";

  // Matriz Andon: 5 blocos representando o estado de cada compressor.
  // O primeiro (APU-Trem-042) é reativo ao stream; os 4 demais seguem o
  // mesmo mock da FleetHealthTable para coerência visual entre componentes.
  const fleetAndon: ReadonlyArray<{ id: string; status: RiskLevel }> = useMemo(
    () => [
      { id: "APU-Trem-042", status: effectiveRiskLevel },
      { id: "APU-Trem-015", status: "NORMAL" },
      { id: "APU-Trem-023", status: "ALERTA" },
      { id: "APU-Trem-031", status: "NORMAL" },
      { id: "APU-Trem-055", status: "NORMAL" },
    ],
    [effectiveRiskLevel],
  );

  const inAlert = fleetAndon.filter((a) => a.status !== "NORMAL").length;

  // Cor do bloco por nível — pulse só em CRÍTICO para evitar ruído visual.
  const ANDON_COLOR: Record<RiskLevel, string> = {
    NORMAL: "bg-emerald-500",
    ALERTA: "bg-amber-500",
    CRÍTICO: "bg-red-500 animate-pulse",
  };

  // ── KPI 3: Anomalia Máxima ──
  const maxAnomalyPct = (liveProbability * 100).toFixed(1);

  // Sparkline da anomalia — extende a série mock com o valor actual no final
  // para que o gráfico reaja à leitura ao vivo do APU-Trem-042.
  const anomalySpark = useMemo(
    () => [...SPARK_ANOMALY.slice(0, -1), Math.round(liveProbability * 100)],
    [liveProbability],
  );

  return (
    <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
      <KpiShell
        title="Saúde Global da Frota"
        value={
          <span>
            {fleetHealth}
            <span className="ml-0.5 text-base font-medium text-slate-400">
              %
            </span>
          </span>
        }
        icon={HeartPulse}
        iconTone="ok"
        spark={SPARK_HEALTH}
        sparkTone="ok"
        isLoading={isLoading}
      />

      <KpiShell
        title="Ativos em Alerta"
        value={
          <span className="flex items-baseline gap-2">
            <span>{inAlert}</span>
            <span className="text-sm font-medium text-slate-400">/ 5</span>
          </span>
        }
        icon={ShieldAlert}
        iconTone={liveIsCritical ? "danger" : liveIsAlert ? "warn" : "ok"}
        isLoading={isLoading}
      >
        {/* Andon Board: cada bloco = 1 compressor. Status em scan rápido,
            sem precisar interpretar um número ou ler um label. */}
        <div
          className="flex h-10 w-full gap-1.5"
          role="group"
          aria-label="Estado da frota — matriz Andon"
        >
          {fleetAndon.map((asset) => (
            <div
              key={asset.id}
              title={`${asset.id}: ${asset.status}`}
              aria-label={`${asset.id}: ${asset.status}`}
              className={cn(
                "flex-1 rounded-sm border border-black/5 transition-colors",
                ANDON_COLOR[asset.status],
              )}
            />
          ))}
        </div>
      </KpiShell>

      <KpiShell
        title="Anomalia Máxima Atual"
        value={
          <span>
            {maxAnomalyPct}
            <span className="ml-0.5 text-base font-medium text-slate-400">
              %
            </span>
          </span>
        }
        icon={AlertOctagon}
        iconTone={anomalyTone}
        spark={anomalySpark}
        sparkTone={anomalyTone}
        isLoading={isLoading}
      />

      <KpiShell
        title="Latência de Inferência"
        value={
          currentLatencyMs !== null ? (
            <span>
              {currentLatencyMs}
              <span className="ml-0.5 text-base font-medium text-slate-400">
                ms
              </span>
            </span>
          ) : (
            // Sem leitura ainda — skeleton honesto em vez de "40 ms" fantasma.
            <Skeleton className="h-7 w-20" />
          )
        }
        subtitle={
          hasLatency
            ? `ao vivo · ${latencyHistory.length}/${LATENCY_HISTORY_SIZE} amostras`
            : "aguardando primeira inferência…"
        }
        icon={Gauge}
        iconTone="neutral"
        spark={latencyHistory}
        sparkTone="neutral"
        isLoading={isLoading}
      />
    </div>
  );
}
