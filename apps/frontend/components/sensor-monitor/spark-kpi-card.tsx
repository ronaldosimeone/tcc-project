"use client";

// ── KPI Card com sparkline — RNF-58: extraído de sensor-monitor.tsx ─────────

import { memo } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import type { SensorDataPoint } from "@/hooks/use-sensor-data";
import { cn } from "@/lib/utils";
import { Sparkline } from "./sparkline";

export interface SparkKpiCardProps {
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

export const SparkKpiCard = memo(function SparkKpiCard({
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
