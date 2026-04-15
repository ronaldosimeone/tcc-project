import { Activity, Cpu } from "lucide-react";
import { Badge } from "@/components/ui/badge";

export default function Header() {
  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b border-border bg-card px-6">
      {/* Marca */}
      <div className="flex items-center gap-3">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10 ring-1 ring-primary/30">
          <Cpu className="h-4 w-4 text-primary" />
        </div>
        <div className="flex flex-col leading-none">
          <span className="text-sm font-semibold tracking-tight text-foreground">
            PredictIQ
          </span>
          <span className="text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
            Manutenção Preditiva
          </span>
        </div>
      </div>

      {/* Status do sistema */}
      <div className="flex items-center gap-4">
        <div className="hidden items-center gap-2 text-xs text-muted-foreground sm:flex">
          <span>MetroPT-3 · Compressor Industrial</span>
        </div>

        <Badge
          variant="outline"
          className="gap-1.5 border-primary/40 bg-primary/10 text-primary"
        >
          <Activity className="h-3 w-3 animate-pulse" />
          AO VIVO
        </Badge>
      </div>
    </header>
  );
}
