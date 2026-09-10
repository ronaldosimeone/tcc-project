// ── Paleta/estilos compartilhados — RNF-58: extraído de sensor-chart.tsx ────

export const LINE_COLORS = {
  TP2: "#60a5fa", // blue-400
  TP3: "#4ade80", // green-400
  Motor_current: "#c084fc", // purple-400
  Oil_temperature: "#fb923c", // orange-400
} as const;

// Estilos compartilhados dos eixos (constantes de módulo — refs estáveis).
// NOTA: `var(--x)` direto, sem envolver em `hsl(...)` — as variáveis do tema
// (app/globals.css) já são `oklch(...)` completos; `hsl(oklch(...))` é CSS
// inválido e renderiza transparente (mesmo bug corrigido no @theme inline).
export const AXIS_TICK_STYLE = {
  fontSize: 10,
  fill: "var(--muted-foreground)",
} as const;

export const GRID_STROKE = "var(--border)";

export const LEGEND_ITEMS: Array<{
  key: keyof typeof LINE_COLORS;
  label: string;
}> = [
  { key: "TP2", label: "TP2 (bar)" },
  { key: "TP3", label: "TP3 (bar)" },
  { key: "Motor_current", label: "Corrente (A)" },
  { key: "Oil_temperature", label: "Temperatura (°C)" },
];
