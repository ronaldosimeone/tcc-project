// ── Topo (logo + toggle) — RNF-58: extraído de sidebar.tsx ──────────────────

import Link from "next/link";
import { Cpu, PanelLeftClose, PanelLeftOpen } from "lucide-react";
import { Button } from "@/components/ui/button";

interface SidebarHeaderProps {
  isOpen: boolean;
  onToggle: (open: boolean) => void;
}

export function SidebarHeader({ isOpen, onToggle }: SidebarHeaderProps) {
  return (
    <div className="flex h-14 shrink-0 items-center border-b border-slate-300 px-3">
      {isOpen ? (
        <div className="flex w-full items-center justify-between gap-2">
          <Link
            href="/"
            className="flex min-w-0 items-center gap-2 transition-opacity hover:opacity-80"
          >
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 ring-1 ring-primary/30">
              <Cpu className="h-4 w-4 text-primary" />
            </div>
            <div className="flex min-w-0 flex-col leading-none">
              <span className="truncate text-sm font-semibold tracking-tight text-sidebar-foreground">
                PredictIQ
              </span>
              <span className="truncate text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
                Manutenção Preditiva
              </span>
            </div>
          </Link>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 shrink-0 text-muted-foreground hover:text-sidebar-foreground"
            onClick={() => onToggle(false)}
            aria-label="Fechar barra lateral"
          >
            <PanelLeftClose className="h-4 w-4" />
          </Button>
        </div>
      ) : (
        <div className="group relative flex w-full items-center justify-center">
          {/* Ícone da logo — desaparece no hover */}
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10 ring-1 ring-primary/30 transition-opacity duration-150 group-hover:opacity-0">
            <Cpu className="h-4 w-4 text-primary" />
          </div>
          {/* Botão de abrir — aparece no hover */}
          <Button
            variant="ghost"
            size="icon"
            className="absolute inset-0 m-auto h-8 w-8 text-muted-foreground opacity-0 transition-opacity duration-150 group-hover:opacity-100 hover:text-sidebar-foreground"
            onClick={() => onToggle(true)}
            aria-label="Abrir barra lateral"
          >
            <PanelLeftOpen className="h-4 w-4" />
          </Button>
        </div>
      )}
    </div>
  );
}
