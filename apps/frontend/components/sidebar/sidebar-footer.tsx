// ── Rodapé (tags de status) — RNF-58: extraído de sidebar.tsx ───────────────

import { Activity } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface SidebarFooterProps {
  isOpen: boolean;
}

export function SidebarFooter({ isOpen }: SidebarFooterProps) {
  return (
    <div
      className={cn(
        "overflow-hidden transition-[max-height,opacity] duration-300",
        isOpen ? "max-h-24 opacity-100" : "max-h-0 opacity-0",
      )}
    >
      <div className="border-t border-slate-300 p-3">
        <p className="truncate text-xs text-muted-foreground">
          MetroPT-3 · Compressor Industrial
        </p>
        <Badge
          variant="outline"
          className="mt-2 gap-1.5 border-primary/40 bg-primary/10 text-primary"
        >
          <Activity className="h-3 w-3 animate-pulse" />
          AO VIVO
        </Badge>
      </div>
    </div>
  );
}
