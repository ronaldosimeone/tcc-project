// ── Estado vazio — RNF-58: extraído de EventLogTable ────────────────────────

import { FileSearch } from "lucide-react";

export function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-3 py-16 text-center">
      <FileSearch className="h-10 w-10 text-muted-foreground/25" />
      <p className="text-sm font-medium text-muted-foreground">
        Nenhum evento encontrado
      </p>
      <p className="text-xs text-muted-foreground/60">
        Ajuste os filtros para ampliar a busca
      </p>
    </div>
  );
}
