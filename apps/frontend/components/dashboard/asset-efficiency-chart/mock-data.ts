// ── Dados mock por ativo — RNF-58: extraído de AssetEfficiencyChart ─────────

export interface EfficiencyDay {
  day: string;
  carga: number;
  ocioso: number;
}

export const EFFICIENCY_BY_ASSET: Record<string, EfficiencyDay[]> = {
  "APU-Trem-042": [
    { day: "Seg", carga: 22, ocioso: 2 },
    { day: "Ter", carga: 21, ocioso: 3 },
    { day: "Qua", carga: 23, ocioso: 1 },
    { day: "Qui", carga: 20, ocioso: 4 },
    { day: "Sex", carga: 22, ocioso: 2 },
    { day: "Sáb", carga: 18, ocioso: 6 },
    { day: "Dom", carga: 16, ocioso: 8 },
  ],
  "APU-Trem-015": [
    { day: "Seg", carga: 20, ocioso: 4 },
    { day: "Ter", carga: 20, ocioso: 4 },
    { day: "Qua", carga: 22, ocioso: 2 },
    { day: "Qui", carga: 19, ocioso: 5 },
    { day: "Sex", carga: 21, ocioso: 3 },
    { day: "Sáb", carga: 16, ocioso: 8 },
    { day: "Dom", carga: 15, ocioso: 9 },
  ],
  "APU-Trem-023": [
    { day: "Seg", carga: 17, ocioso: 7 },
    { day: "Ter", carga: 16, ocioso: 8 },
    { day: "Qua", carga: 18, ocioso: 6 },
    { day: "Qui", carga: 15, ocioso: 9 },
    { day: "Sex", carga: 17, ocioso: 7 },
    { day: "Sáb", carga: 12, ocioso: 12 },
    { day: "Dom", carga: 10, ocioso: 14 },
  ],
  "APU-Trem-031": [
    { day: "Seg", carga: 23, ocioso: 1 },
    { day: "Ter", carga: 22, ocioso: 2 },
    { day: "Qua", carga: 23, ocioso: 1 },
    { day: "Qui", carga: 21, ocioso: 3 },
    { day: "Sex", carga: 23, ocioso: 1 },
    { day: "Sáb", carga: 20, ocioso: 4 },
    { day: "Dom", carga: 19, ocioso: 5 },
  ],
  "APU-Trem-055": [
    { day: "Seg", carga: 19, ocioso: 5 },
    { day: "Ter", carga: 21, ocioso: 3 },
    { day: "Qua", carga: 20, ocioso: 4 },
    { day: "Qui", carga: 18, ocioso: 6 },
    { day: "Sex", carga: 19, ocioso: 5 },
    { day: "Sáb", carga: 14, ocioso: 10 },
    { day: "Dom", carga: 13, ocioso: 11 },
  ],
};

export const DEFAULT_EFFICIENCY = EFFICIENCY_BY_ASSET["APU-Trem-042"];

// ── Estilos ───────────────────────────────────────────────────────────────
// NOTA: `var(--x)` direto, sem `hsl(...)` — os tokens do tema já são
// `oklch(...)` completos; `hsl(oklch(...))` é CSS inválido e renderiza
// transparente (mesmo bug corrigido no @theme inline de app/globals.css).

export const AXIS_TICK = {
  fontSize: 10,
  fill: "var(--muted-foreground)",
} as const;
export const GRID_STROKE = "var(--border)";
