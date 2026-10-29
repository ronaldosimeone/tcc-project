// ── Seção "Limite de alerta" — RNF-58: extraído de alert-settings-form.tsx ──
//
// O valor é um NÍVEL DE RISCO relativo ao limiar calibrado do modelo ativo
// (mesma escala do painel e de apps/backend/src/core/risk.py): 65% = o
// modelo decide "falha". Uma probabilidade bruta fixa não serve para todos
// os modelos — a MLP decide falha com P ≥ 0,06 e nunca chegaria a 85%.

import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import {
  MAX_THRESHOLD,
  MIN_THRESHOLD,
  STEP,
  toPercentLabel,
} from "@/hooks/use-alert-settings-form";
import { CRITICAL_RISK } from "@/lib/risk-thresholds";

export interface ThresholdSectionProps {
  threshold: number;
  isSaving: boolean;
  onSliderChange: (values: number[]) => void;
}

export function ThresholdSection({
  threshold,
  isSaving,
  onSliderChange,
}: ThresholdSectionProps) {
  const markerLeft =
    ((CRITICAL_RISK - MIN_THRESHOLD) / (MAX_THRESHOLD - MIN_THRESHOLD)) * 100;

  return (
    <section className="flex flex-col gap-4">
      <div className="flex items-baseline justify-between">
        <Label htmlFor="alert-threshold-slider" className="text-sm font-medium">
          Notificar quando o índice de risco chegar a
        </Label>
        <span
          data-testid="alert-threshold-value"
          className="font-mono text-lg font-semibold text-foreground"
        >
          {toPercentLabel(threshold)}
        </span>
      </div>

      <Slider
        id="alert-threshold-slider"
        aria-label="Índice de risco para notificação"
        min={MIN_THRESHOLD}
        max={MAX_THRESHOLD}
        step={STEP}
        value={[threshold]}
        onValueChange={onSliderChange}
        disabled={isSaving}
      />
      <div className="relative flex justify-between text-xs text-muted-foreground">
        <span>{toPercentLabel(MIN_THRESHOLD)}</span>
        <span
          className="absolute -translate-x-1/2 font-medium text-foreground"
          style={{ left: `${markerLeft}%` }}
        >
          {toPercentLabel(CRITICAL_RISK)} · início do CRÍTICO
        </span>
        <span>{toPercentLabel(MAX_THRESHOLD)}</span>
      </div>
    </section>
  );
}
