"use client";

// ── Formulário de disparo — RNF-58: extraído de maintenance-assistant.tsx ───

import { type FormEvent, useId, useState } from "react";
import { Loader2, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import type { MaintenanceSuggestionPayload } from "@/lib/api-client";
import { CRITICAL_RISK } from "@/lib/risk-thresholds";

export interface SuggestionFormProps {
  disabled: boolean;
  onSubmit: (input: MaintenanceSuggestionPayload) => void;
}

export function SuggestionForm({ disabled, onSubmit }: SuggestionFormProps) {
  const riskId = useId();
  const [equipmentName, setEquipmentName] = useState(
    "Compressor de ar industrial",
  );
  const [symptomDescription, setSymptomDescription] = useState("");
  const [riskPct, setRiskPct] = useState(String(CRITICAL_RISK * 100));

  const handleSubmit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    const risk = Number(riskPct) / 100;
    if (Number.isNaN(risk)) return;
    // Índice de risco = probabilidade num modelo de limiar 0,65 (escala de
    // lib/risk-thresholds), então o backend aciona com risco ≥ 65%.
    onSubmit({
      failure_probability: Math.min(Math.max(risk, 0), 1),
      decision_threshold: CRITICAL_RISK,
      equipment_name: equipmentName.trim() || "Equipamento",
      symptom_description: symptomDescription.trim() || undefined,
    });
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-3">
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="maintenance-equipment-name">Equipamento</Label>
        <Input
          id="maintenance-equipment-name"
          value={equipmentName}
          onChange={(e) => setEquipmentName(e.target.value)}
          disabled={disabled}
          maxLength={200}
        />
      </div>

      <div className="flex flex-col gap-1.5">
        <Label htmlFor="maintenance-symptom">Sintoma observado</Label>
        <Textarea
          id="maintenance-symptom"
          value={symptomDescription}
          onChange={(e) => setSymptomDescription(e.target.value)}
          disabled={disabled}
          maxLength={500}
          placeholder="ex.: vazamento de óleo, ruído excessivo na sucção…"
        />
      </div>

      <div className="flex flex-col gap-1.5">
        <Label htmlFor={riskId}>Índice de risco (%)</Label>
        <Input
          id={riskId}
          type="number"
          min={0}
          max={100}
          step={1}
          value={riskPct}
          onChange={(e) => setRiskPct(e.target.value)}
          disabled={disabled}
        />
        <p className="text-xs text-muted-foreground">
          A sugestão é gerada a partir de 65% (CRÍTICO), como nos eventos reais.
        </p>
      </div>

      <Button
        type="submit"
        disabled={disabled}
        size="sm"
        className="self-start"
      >
        {disabled ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
        ) : (
          <Sparkles className="h-3.5 w-3.5" />
        )}
        Gerar sugestão
      </Button>
    </form>
  );
}
