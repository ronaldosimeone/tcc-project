"use client";

import { useState } from "react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import dynamic from "next/dynamic";
import {
  Activity,
  BarChart3,
  Cpu,
  FlaskConical,
  Gauge,
  LayoutDashboard,
  PanelLeftClose,
  PanelLeftOpen,
  Settings,
  Sparkles,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
// Code splitting (RNF-40): o painel só é montado quando o usuário clica em
// "Simulação" — nunca faz parte do primeiro paint do Dashboard. Sem isso,
// seu JS (formulário + radio group) entrava no bundle inicial de toda
// visita a "/" mesmo para quem nunca abre o painel. `ssr:false` é seguro
// aqui porque o conteúdo só existe dentro de um <Sheet> fechado por padrão.
const SimulationPanel = dynamic(
  () => import("@/components/simulation-panel").then((m) => m.SimulationPanel),
  { ssr: false },
);
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

type NavAction = "simulation";

interface NavItem {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  href?: string;
  action?: NavAction;
  /** Item exibido mas inativo — indica feature em roadmap (Coming Soon). */
  disabled?: boolean;
  /** Texto opcional do tooltip extra (ex.: "Em breve"). */
  hint?: string;
}

// Navegação única — itens `disabled` (Assistente de IA, Configurações)
// convivem com os ativos na mesma lista, mantendo a estética uniforme.
// "Configurações" fica por último como ponto de acesso final convencional.
const NAV_ITEMS: NavItem[] = [
  { icon: LayoutDashboard, label: "Dashboard", href: "/" },
  { icon: Gauge, label: "Sensores", href: "/sensors/APU-Trem-042" },
  { icon: BarChart3, label: "Histórico", href: "/history" },
  { icon: FlaskConical, label: "Simulação", action: "simulation" },
  {
    icon: Sparkles,
    label: "Assistente de IA",
    disabled: true,
    hint: "Em breve",
  },
  { icon: Settings, label: "Configurações", disabled: true, hint: "Em breve" },
];

function isActive(href: string | undefined, pathname: string): boolean {
  if (!href) return false;
  if (href === "/") return pathname === "/";
  return pathname.startsWith(href);
}

interface NavLinkProps {
  item: NavItem;
  active: boolean;
  isOpen: boolean;
  onAction?: (action: NavAction) => void;
}

function NavLink({ item, active, isOpen, onAction }: NavLinkProps) {
  const Icon = item.icon;

  const cls = cn(
    "flex h-10 w-full items-center rounded-md text-sm font-medium transition-colors",
    isOpen ? "gap-3 px-3" : "justify-center px-0",
    item.disabled
      ? "cursor-not-allowed text-muted-foreground opacity-50"
      : active
      ? "bg-primary/10 text-primary ring-1 ring-inset ring-primary/20"
      : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
  );

  const inner = (
    <>
      <Icon className="h-4 w-4 shrink-0" />
      <span
        className={cn(
          "overflow-hidden truncate transition-[opacity,width] duration-200",
          isOpen ? "w-auto opacity-100" : "w-0 opacity-0",
        )}
      >
        {item.label}
      </span>
    </>
  );

  // Disabled: sem Link, sem action — apenas um botão inerte com aria correto.
  // Hover/active variants neutralizados via classes acima.
  const element = item.disabled ? (
    <button
      type="button"
      disabled
      aria-disabled="true"
      title={item.hint}
      className={cls}
    >
      {inner}
    </button>
  ) : item.href ? (
    <Link href={item.href} className={cls}>
      {inner}
    </Link>
  ) : (
    <button
      type="button"
      className={cls}
      onClick={item.action ? () => onAction?.(item.action!) : undefined}
    >
      {inner}
    </button>
  );

  if (!isOpen) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>{element}</TooltipTrigger>
        <TooltipContent side="right">
          {item.label}
          {item.hint && (
            <span className="ml-1 text-[10px] text-slate-400">
              · {item.hint}
            </span>
          )}
        </TooltipContent>
      </Tooltip>
    );
  }

  return element;
}

export default function Sidebar() {
  const [isOpen, setIsOpen] = useState(true);
  const [simulationOpen, setSimulationOpen] = useState(false);
  const pathname = usePathname();

  const handleAction = (action: NavAction) => {
    if (action === "simulation") setSimulationOpen(true);
  };

  return (
    <TooltipProvider delayDuration={0}>
      <SimulationPanel open={simulationOpen} onOpenChange={setSimulationOpen} />
      <aside
        className={cn(
          "flex shrink-0 flex-col border-r border-slate-300 bg-sidebar",
          "overflow-hidden transition-[width] duration-300 ease-in-out",
          isOpen ? "w-64" : "w-16",
        )}
      >
        {/* ── TOPO ──────────────────────────────────────────────── */}
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
                onClick={() => setIsOpen(false)}
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
                onClick={() => setIsOpen(true)}
                aria-label="Abrir barra lateral"
              >
                <PanelLeftOpen className="h-4 w-4" />
              </Button>
            </div>
          )}
        </div>

        {/* ── NAVEGAÇÃO ────────────────────────────────────────────
            Lista única; itens em roadmap (disabled) convivem com os
            ativos preservando a estética. `flex-1` empurra o rodapé
            de status para a base. */}
        <nav className="flex flex-1 flex-col gap-1 p-2 py-4">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.label}
              item={item}
              active={isActive(item.href, pathname)}
              isOpen={isOpen}
              onAction={handleAction}
            />
          ))}
        </nav>

        {/* ── RODAPÉ — Tags de status ────────────────────────────── */}
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
      </aside>
    </TooltipProvider>
  );
}
