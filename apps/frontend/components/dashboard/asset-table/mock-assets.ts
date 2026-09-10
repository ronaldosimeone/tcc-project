// ── Mock data — RNF-58: extraído de AssetTable ──────────────────────────────
// NOTA: cópia independente da de `fleet-health-table/mock-assets.ts` — os
// dois componentes já mantinham dados mockados próprios antes desta
// decomposição (não consolidados, para evitar divergência sutil de risco).

import type { RiskLevel } from "@/hooks/use-sensor-data";

export interface MockAsset {
  id: string;
  riskLevel: RiskLevel;
  prob: number;
  tp2: number;
  tp3: number;
  motorCurrent: number;
  oilTemp: number;
  lastSeen: string;
}

export const MOCK_ASSETS: MockAsset[] = [
  {
    id: "APU-Trem-015",
    riskLevel: "NORMAL",
    prob: 0.123,
    tp2: 8.2,
    tp3: 7.9,
    motorCurrent: 5.1,
    oilTemp: 72.0,
    lastSeen: "2 min",
  },
  {
    id: "APU-Trem-023",
    riskLevel: "ALERTA",
    prob: 0.456,
    tp2: 7.8,
    tp3: 7.2,
    motorCurrent: 7.8,
    oilTemp: 81.3,
    lastSeen: "1 min",
  },
  {
    id: "APU-Trem-031",
    riskLevel: "NORMAL",
    prob: 0.089,
    tp2: 8.4,
    tp3: 8.1,
    motorCurrent: 4.2,
    oilTemp: 68.5,
    lastSeen: "3 min",
  },
  {
    id: "APU-Trem-055",
    riskLevel: "NORMAL",
    prob: 0.221,
    tp2: 8.1,
    tp3: 7.8,
    motorCurrent: 6.3,
    oilTemp: 74.1,
    lastSeen: "4 min",
  },
];
