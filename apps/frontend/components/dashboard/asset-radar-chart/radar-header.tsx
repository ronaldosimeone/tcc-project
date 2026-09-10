// ── Cabeçalho (título + badges + tiras de valores brutos) — RNF-58:
// extraído de AssetRadarChart ────────────────────────────────────────────────

import { Activity } from "lucide-react";
import { CardHeader, CardTitle } from "@/components/ui/card";
import type { PredictPayload } from "@/lib/api-client";
import { ANOMALY_STYLE, toAnomalyLevel } from "./helpers";
import { RawValue } from "./raw-value";

interface RadarHeaderProps {
  sensorData: PredictPayload;
  isLive: boolean;
  assetId: string;
  anomalyScore: number;
  isLoading?: boolean;
}

export function RadarHeader({
  sensorData,
  isLive,
  assetId,
  anomalyScore,
  isLoading,
}: RadarHeaderProps) {
  const level = toAnomalyLevel(anomalyScore);

  return (
    <CardHeader className="px-5 pb-2 pt-4">
      {/* ── Title row with status badges ─────────────────────────── */}
      <div className="flex items-start justify-between gap-2">
        <CardTitle className="flex items-center gap-2 text-sm font-semibold text-foreground/90">
          <Activity className="h-4 w-4 text-primary" />
          Perfil Operacional
        </CardTitle>

        <div className="flex shrink-0 items-center gap-1.5">
          {isLive ? (
            <span className="flex items-center gap-1.5 rounded-full border border-green-500/30 bg-green-500/10 px-2 py-0.5 text-[10px] font-semibold text-green-400">
              <span
                className="h-1.5 w-1.5 animate-pulse rounded-full bg-green-400"
                aria-hidden="true"
              />
              LIVE
            </span>
          ) : (
            <span className="rounded-full border border-border bg-muted/30 px-2 py-0.5 text-[10px] font-medium text-muted-foreground">
              IDLE
            </span>
          )}

          {/* Anomaly score badge — only shown in live mode */}
          {isLive && (
            <span
              className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${ANOMALY_STYLE[level]}`}
              aria-label={`Score de anomalia: ${level} ${(
                anomalyScore * 100
              ).toFixed(1)}%`}
            >
              {level} {(anomalyScore * 100).toFixed(1)}%
            </span>
          )}
        </div>
      </div>

      <p className="text-[11px] text-muted-foreground">
        {assetId} · {isLive ? "Ótimo vs Atual" : "Ótimo vs Estático"}{" "}
        (normalizado 0–100)
      </p>

      {/* ── Raw instantaneous values strip — live mode only ──────── */}
      {isLive && !isLoading && (
        <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-border/50 pt-1.5">
          <RawValue label="TP2" value={sensorData.TP2} unit="bar" />
          <span className="text-border/60" aria-hidden="true">
            ·
          </span>
          <RawValue
            label="Temp"
            value={sensorData.Oil_temperature}
            decimals={1}
            unit="°C"
          />
          <span className="text-border/60" aria-hidden="true">
            ·
          </span>
          <RawValue label="Motor" value={sensorData.Motor_current} unit="A" />
        </div>
      )}
    </CardHeader>
  );
}
