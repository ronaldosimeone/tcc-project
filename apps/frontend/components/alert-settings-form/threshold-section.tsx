// ── Seção "Limite de alerta" — RNF-58: extraído de alert-settings-form.tsx ──

import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import {
  MAX_THRESHOLD,
  MIN_THRESHOLD,
  STEP,
  toPercentLabel,
} from "@/hooks/use-alert-settings-form";

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
  return (
    <section className="flex flex-col gap-4">
      <div className="flex items-baseline justify-between">
        <Label htmlFor="alert-threshold-slider" className="text-sm font-medium">
          Limite de alerta crítico
        </Label>
        <span
          data-testid="alert-threshold-value"
          className="font-mono text-lg font-semibold text-slate-900"
        >
          {toPercentLabel(threshold)}
        </span>
      </div>

      <Slider
        id="alert-threshold-slider"
        aria-label="Limite de alerta crítico"
        min={MIN_THRESHOLD}
        max={MAX_THRESHOLD}
        step={STEP}
        value={[threshold]}
        onValueChange={onSliderChange}
        disabled={isSaving}
      />
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{toPercentLabel(MIN_THRESHOLD)}</span>
        <span>{toPercentLabel(MAX_THRESHOLD)}</span>
      </div>

      <p className="text-xs text-muted-foreground">
        Alertas críticos serão disparados quando a probabilidade de falha
        ultrapassar {toPercentLabel(threshold)}.
      </p>
    </section>
  );
}
