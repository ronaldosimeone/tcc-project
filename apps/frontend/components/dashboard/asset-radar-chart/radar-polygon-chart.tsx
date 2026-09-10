// ── Gráfico radar (Ótimo vs Atual) — RNF-58: extraído de AssetRadarChart ────

import {
  Legend,
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { ANGLE_TICK, GRID_STROKE } from "./helpers";
import { TOOLTIP_CONTENT } from "./radar-tooltip";

export interface RadarDatum {
  subject: string;
  ótimo: number;
  atual: number;
}

interface RadarPolygonChartProps {
  data: RadarDatum[];
}

export function RadarPolygonChart({ data }: RadarPolygonChartProps) {
  return (
    <ResponsiveContainer width="100%" height={200}>
      <RadarChart
        data={data}
        outerRadius={62}
        margin={{ top: 8, right: 28, bottom: 0, left: 28 }}
      >
        <PolarGrid stroke={GRID_STROKE} radialLines />
        <PolarAngleAxis dataKey="subject" tick={ANGLE_TICK} />
        <Tooltip content={TOOLTIP_CONTENT} />
        <Radar
          name="Ótimo"
          dataKey="ótimo"
          stroke="#4ade80"
          fill="#4ade80"
          fillOpacity={0.08}
          strokeWidth={1.5}
          strokeDasharray="4 2"
        />
        <Radar
          name="Atual"
          dataKey="atual"
          stroke="#60a5fa"
          fill="#60a5fa"
          fillOpacity={0.22}
          strokeWidth={2}
        />
        <Legend
          iconSize={8}
          iconType="circle"
          verticalAlign="bottom"
          wrapperStyle={{
            fontSize: 11,
            color: "var(--muted-foreground)",
            paddingTop: 12,
          }}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
}
