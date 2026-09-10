"use client";

/**
 * RNF-58: decomposto em `components/sidebar/*` — constants
 * (NAV_ITEMS/isActive), NavLink, SidebarHeader, SidebarFooter. Este
 * arquivo mantém só o estado (isOpen/painéis) e a orquestração do
 * `<aside>`. Nenhuma mudança de comportamento/DOM.
 */

import { useState } from "react";
import { usePathname } from "next/navigation";
import dynamic from "next/dynamic";

// Code splitting (RNF-40): o painel só é montado quando o usuário clica em
// "Simulação" — nunca faz parte do primeiro paint do Dashboard. Sem isso,
// seu JS (formulário + radio group) entrava no bundle inicial de toda
// visita a "/" mesmo para quem nunca abre o painel. `ssr:false` é seguro
// aqui porque o conteúdo só existe dentro de um <Sheet> fechado por padrão.
const SimulationPanel = dynamic(
  () => import("@/components/simulation-panel").then((m) => m.SimulationPanel),
  { ssr: false },
);
// RF-23: mesmo motivo de code-splitting do SimulationPanel acima — o painel
// (react-markdown + formulário + hook de streaming) só entra no bundle
// quando o usuário clica em "Assistente de IA", nunca no first paint.
const MaintenanceAssistant = dynamic(
  () =>
    import("@/components/maintenance-assistant").then(
      (m) => m.MaintenanceAssistant,
    ),
  { ssr: false },
);

import { TooltipProvider } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { isActive, NAV_ITEMS, type NavAction } from "./sidebar/constants";
import { NavLink } from "./sidebar/nav-link";
import { SidebarFooter } from "./sidebar/sidebar-footer";
import { SidebarHeader } from "./sidebar/sidebar-header";

export default function Sidebar() {
  const [isOpen, setIsOpen] = useState(true);
  const [simulationOpen, setSimulationOpen] = useState(false);
  const [maintenanceAssistantOpen, setMaintenanceAssistantOpen] =
    useState(false);
  const pathname = usePathname();

  const handleAction = (action: NavAction) => {
    if (action === "simulation") setSimulationOpen(true);
    if (action === "maintenance-assistant") setMaintenanceAssistantOpen(true);
  };

  return (
    <TooltipProvider delayDuration={0}>
      <SimulationPanel open={simulationOpen} onOpenChange={setSimulationOpen} />
      <MaintenanceAssistant
        open={maintenanceAssistantOpen}
        onOpenChange={setMaintenanceAssistantOpen}
      />
      <aside
        className={cn(
          "flex shrink-0 flex-col border-r border-slate-300 bg-sidebar",
          "overflow-hidden transition-[width] duration-300 ease-in-out",
          isOpen ? "w-64" : "w-16",
        )}
      >
        <SidebarHeader isOpen={isOpen} onToggle={setIsOpen} />

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

        <SidebarFooter isOpen={isOpen} />
      </aside>
    </TooltipProvider>
  );
}
