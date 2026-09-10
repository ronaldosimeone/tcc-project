// ── Referências (texto, nunca link inventado — RF-23 §7) — RNF-58: extraído
// de maintenance-assistant.tsx ───────────────────────────────────────────────

import { FileText } from "lucide-react";
import type { ManualReference } from "@/lib/api-client";

export function ReferencesList({
  references,
}: {
  references: ManualReference[];
}) {
  if (references.length === 0) {
    return (
      <p
        data-testid="maintenance-references"
        className="text-xs text-muted-foreground"
      >
        Nenhum manual foi citado como referência para este plano.
      </p>
    );
  }
  return (
    <ul data-testid="maintenance-references" className="flex flex-col gap-1.5">
      {references.map((ref, i) => (
        <li
          key={`${ref.file_name}-${ref.page}-${ref.chunk_index}-${i}`}
          className="flex items-start gap-2 text-xs text-muted-foreground"
        >
          <FileText className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          {/* Sem URL segura conhecida para o PDF de origem (nenhum endpoint
              de arquivo estático existe hoje) — mostrado como texto, nunca
              como link inventado (RF-23 §7). */}
          <span>
            <span className="font-mono">{ref.file_name}</span> — página{" "}
            {ref.page}
            <span className="ml-1 text-[10px] opacity-70">
              (score {ref.score.toFixed(2)})
            </span>
          </span>
        </li>
      ))}
    </ul>
  );
}
