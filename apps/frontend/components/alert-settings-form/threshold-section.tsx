// ── Seção "Limite de alerta" — RNF-58: extraído de alert-settings-form.tsx ──
//
// O valor é um NÍVEL DE RISCO relativo ao limiar calibrado do modelo ativo
// (mesma escala do painel e de apps/backend/src/core/risk.py): 65% = o
// modelo decide "falha". Uma probabilidade bruta fixa não serve para todos
// os modelos — a MLP decide falha com P ≥ 0,06 e nunca chegaria a 85%.

import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import type { ActiveModelInfo } from "@/hooks/use-active-model-info";
import {
  MAX_THRESHOLD,
  MIN_THRESHOLD,
  STEP,
  toPercentLabel,
} from "@/hooks/use-alert-settings-form";
import { CRITICAL_RISK, probabilityForRisk } from "@/lib/risk-thresholds";

export interface ThresholdSectionProps {
  threshold: number;
  isSaving: boolean;
  onSliderChange: (values: number[]) => void;
  /** Modelo ativo e seu limiar — `null` enquanto desconhecido. */
  activeModel?: ActiveModelInfo | null;
}

const relation = (threshold: number): string => {
  if (Math.abs(threshold - CRITICAL_RISK) < 1e-6) {
    return "exatamente quando o modelo decide “falha” (o painel mostra CRÍTICO)";
  }
  return threshold > CRITICAL_RISK
    ? "só com evidência mais forte do que a decisão do modelo"
    : "ainda na faixa de ALERTA, antes de o modelo decidir “falha”";
};

export function ThresholdSection({
  threshold,
  isSaving,
  onSliderChange,
  activeModel = null,
}: ThresholdSectionProps) {
  const equivalent = activeModel
    ? probabilityForRisk(threshold, activeModel.threshold)
    : null;
  const markerLeft =
    ((CRITICAL_RISK - MIN_THRESHOLD) / (MAX_THRESHOLD - MIN_THRESHOLD)) * 100;

  return (
    <section className="flex flex-col gap-4">
      <div className="flex items-baseline justify-between">
        <Label htmlFor="alert-threshold-slider" className="text-sm font-medium">
          Nível de risco para alerta crítico
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
        aria-label="Nível de risco para alerta crítico"
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
          {toPercentLabel(CRITICAL_RISK)} · decisão do modelo
        </span>
        <span>{toPercentLabel(MAX_THRESHOLD)}</span>
      </div>

      <p className="text-xs text-muted-foreground">
        O nível é relativo ao limiar calibrado do modelo ativo: com{" "}
        {toPercentLabel(threshold)}, o alerta crítico (Telegram / e-mail) é
        enviado {relation(threshold)}.
      </p>
      {activeModel && equivalent !== null && (
        <p
          data-testid="alert-threshold-equivalent"
          className="text-xs text-muted-foreground"
        >
          Com o modelo ativo <strong>{activeModel.label}</strong> (limiar{" "}
          {activeModel.threshold.toFixed(3)}), isso corresponde a probabilidade
          de falha ≥ <strong>{toPercentLabel(equivalent)}</strong>.
        </p>
      )}
    </section>
  );
}
