"use client";

/**
 * Painel de simulação MLOps.
 *
 * Aberto a partir do botão "Simulação" da Sidebar. Permite:
 *  - Trocar o modelo activo de inferência (RF-11)
 *  - Trocar o cenário de dados do simulador (RNF-29)
 *
 * Regra crítica: o `value` enviado para a API é SEMPRE o nome bruto
 * (`random_forest_v2`), enquanto o utilizador vê apenas a label limpa
 * (`Random Forest`). Toda a lógica de filtragem/formatação vive em
 * `lib/model-name.ts` e é puramente sincrona.
 *
 * RNF-58: decomposto em `components/simulation-panel/*` — `ModelSection`
 * e `ScenarioSection`. Este arquivo é só o wrapper Sheet.
 */

import { Sparkles } from "lucide-react";

import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { ModelSection } from "./simulation-panel/model-section";
import { ScenarioSection } from "./simulation-panel/scenario-section";

interface SimulationPanelProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function SimulationPanel({ open, onOpenChange }: SimulationPanelProps) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-full sm:max-w-md">
        <SheetHeader>
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-primary" />
            <SheetTitle>Painel de Simulação</SheetTitle>
          </div>
          <SheetDescription>
            Controle o modelo de inferência e o cenário do simulador em tempo
            real.
          </SheetDescription>
        </SheetHeader>

        <div className="flex flex-1 flex-col gap-6 overflow-y-auto px-6 py-4">
          <ModelSection />
          <ScenarioSection />
        </div>
      </SheetContent>
    </Sheet>
  );
}
