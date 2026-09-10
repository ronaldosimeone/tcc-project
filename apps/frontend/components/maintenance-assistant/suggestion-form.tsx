"use client";

// ── Formulário de disparo — RNF-58: extraído de maintenance-assistant.tsx ───

import { type FormEvent, useId, useState } from "react";
import { Loader2, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";

export interface SuggestionFormProps {
  disabled: boolean;
  onSubmit: (input: {
    failure_probability: number;
    equipment_name: string;
    symptom_description?: string;
  }) => void;
}

export function SuggestionForm({ disabled, onSubmit }: SuggestionFormProps) {
  const probabilityId = useId();
  const [equipmentName, setEquipmentName] = useState(
    "Compressor de ar industrial",
  );
  const [symptomDescription, setSymptomDescription] = useState("");
  const [failureProbability, setFailureProbability] = useState("0.85");

  const handleSubmit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    const probability = Number(failureProbability);
    if (Number.isNaN(probability)) return;
    onSubmit({
      failure_probability: probability,
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
        <Label htmlFor={probabilityId}>Probabilidade de falha</Label>
        <Input
          id={probabilityId}
          type="number"
          min={0}
          max={1}
          step={0.01}
          value={failureProbability}
          onChange={(e) => setFailureProbability(e.target.value)}
          disabled={disabled}
        />
        <p className="text-xs text-muted-foreground">
          Sugestão automática só é gerada quando a probabilidade excede 0.7
          (RF-22).
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
