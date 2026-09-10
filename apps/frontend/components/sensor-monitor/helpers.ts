// ── Helpers puros — RNF-58: extraído de sensor-monitor.tsx ─────────────────

import type { RiskLevel, SensorDataPoint } from "@/hooks/use-sensor-data";
import { OP_STATES, OP_THRESHOLDS } from "./constants";

export function riskColor(level: RiskLevel): string {
  return level === "NORMAL"
    ? "hsl(142 71% 45%)"
    : level === "ALERTA"
    ? "hsl(38 92% 50%)"
    : "hsl(0 72% 51%)";
}

/** Constrói caminhos SVG para um arco semi-circular de gauges. */
export function buildArcPaths(
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

export interface OpSlice {
  name: string;
  color: string;
  value: number;
}

export function computeOpState(history: SensorDataPoint[]): OpSlice[] {
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
export function makeDonutFormatter(total: number) {
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
