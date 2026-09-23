"use client";

/**
 * RNF-74 — extraído de `ModelStatusCard.tsx` para poder ser carregado via
 * `next/dynamic` (ver uso em ModelStatusCard.tsx): é a ÚNICA parte do card
 * que depende de `recharts` (~100KB gzip no bundle analisado, o maior
 * contribuinte individual do First Load JS do Dashboard — ver
 * RELATORIO-RNF-74-RNF-75.md). O resto do card (título, badge de
 * confiança, legenda) não precisa de recharts e continua estático.
 *
 * Mantém EXATAMENTE o mesmo markup/props que existia inline em
 * ModelStatusCard antes desta task — nenhuma mudança visual.
 */

import { Cell, Pie, PieChart, ResponsiveContainer } from "recharts";

export interface DonutSlice {
  name: string;
  value: number;
  color: string;
}

interface ModelStatusDonutProps {
  pieData: DonutSlice[];
}

export function ModelStatusDonut({ pieData }: ModelStatusDonutProps) {
  return (
    <div aria-hidden="true" className="contents">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart accessibilityLayer={false}>
          <Pie
            data={pieData}
            innerRadius={28}
            outerRadius={44}
            paddingAngle={2}
            dataKey="value"
            stroke="none"
            isAnimationActive={false}
          >
            {pieData.map((entry, i) => (
              <Cell key={i} fill={entry.color} />
            ))}
          </Pie>
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
