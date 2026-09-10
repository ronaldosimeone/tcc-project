// ── Paginação — RNF-58: extraído de EventLogTable ───────────────────────────

import { Button } from "@/components/ui/button";

interface EventLogPaginationProps {
  safePage: number;
  totalPages: number;
  pageNumbers: (number | "…")[];
  onPrev: () => void;
  onNext: () => void;
  onPageSelect: (page: number) => void;
}

export function EventLogPagination({
  safePage,
  totalPages,
  pageNumbers,
  onPrev,
  onNext,
  onPageSelect,
}: EventLogPaginationProps) {
  return (
    <div className="mt-5 flex items-center justify-between border-t border-border pt-4">
      <Button
        variant="outline"
        size="sm"
        className="h-8 text-xs"
        onClick={onPrev}
        disabled={safePage === 1}
      >
        Anterior
      </Button>

      <div className="flex items-center gap-1">
        {pageNumbers.map((p, i) =>
          p === "…" ? (
            <span
              key={`ellipsis-${i}`}
              className="px-1 text-xs text-muted-foreground"
            >
              …
            </span>
          ) : (
            <Button
              key={p}
              variant={Number(safePage) === Number(p) ? "outline" : "ghost"}
              className={
                Number(safePage) === Number(p)
                  ? "h-8 w-8 rounded-full border-2 border-primary font-bold text-primary"
                  : "h-8 w-8 rounded-full text-muted-foreground"
              }
              onClick={() => onPageSelect(p as number)}
            >
              {p}
            </Button>
          ),
        )}
      </div>

      <Button
        variant="outline"
        size="sm"
        className="h-8 text-xs"
        onClick={onNext}
        disabled={safePage === totalPages}
      >
        Próxima
      </Button>
    </div>
  );
}
