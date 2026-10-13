// ── Causa raiz da IA — RNF-58: extraído de RootCauseDrawer ──────────────────

import { Sparkles } from "lucide-react";

export function AiDiagnosis() {
  return (
    <section
      className="rounded-lg border border-border border-l-4 border-l-indigo-500 bg-indigo-50/50 p-4"
      aria-label="Diagnóstico da IA"
    >
      <div className="flex items-start gap-3">
        <span className="mt-0.5 shrink-0 rounded-md bg-indigo-500/15 p-1.5 text-indigo-600 dark:text-indigo-400">
          <Sparkles className="h-3.5 w-3.5" />
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-[10px] font-bold uppercase tracking-wider text-indigo-700 dark:text-indigo-400">
            Análise Preditiva
          </p>
          <p className="mt-1 text-sm leading-relaxed text-foreground/80">
            A anomalia foi precedida por{" "}
            <span className="font-semibold text-foreground">
              vibração anômala no rolamento principal
            </span>{" "}
            cerca de{" "}
            <span className="font-semibold text-foreground">45 minutos</span>{" "}
            antes do evento crítico. Padrão compatível com{" "}
            <span className="font-semibold text-foreground">
              desgaste de material
            </span>
            .
          </p>
        </div>
      </div>
    </section>
  );
}
