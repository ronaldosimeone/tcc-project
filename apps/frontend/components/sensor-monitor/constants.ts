// ── Paleta de cores (light-mode) — RNF-58: extraído de sensor-monitor.tsx ──

export const C = {
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

// NOTA: `var(--x)` direto, sem `hsl(...)` — os tokens do tema já são
// `oklch(...)` completos; `hsl(oklch(...))` é CSS inválido e renderiza
// transparente (mesmo bug corrigido no @theme inline de app/globals.css).
export const GRID_STROKE = "var(--border)";
export const AXIS_TICK = {
  fontSize: 10,
  fill: "var(--muted-foreground)",
} as const;

// Limites calibrados com dados reais do MetroPT-3:
//   0.04A → Desligado  |  3.76A → Sem Carga  |  7A → Com Carga  |  9A → Partida
export const OP_THRESHOLDS = { off: 1.0, noLoad: 5.5, load: 8.5 } as const;

export const OP_STATES = [
  { name: "Desligado", color: C.off },
  { name: "Sem Carga", color: C.noload },
  { name: "Com Carga", color: C.load },
  { name: "Partida", color: C.start },
] as const;

export type BoolSignalMode = "normal" | "alertWhenOn" | "alertWhenOff";

export interface BooleanPanelProps {
  COMP: number;
  DV_eletric: number;
  Towers: number;
  MPG: number;
  LPS: number;
  Pressure_switch: number;
  Oil_level: number;
  Caudal_impulses: number;
}

export const BOOL_SIGNALS: Array<{
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
