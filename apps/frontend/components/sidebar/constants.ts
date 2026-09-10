// ── Itens de navegação/helpers — RNF-58: extraído de sidebar.tsx ────────────

import {
  BarChart3,
  FlaskConical,
  Gauge,
  LayoutDashboard,
  Settings,
  Sparkles,
} from "lucide-react";

export type NavAction = "simulation" | "maintenance-assistant";

export interface NavItem {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  href?: string;
  action?: NavAction;
  /** Item exibido mas inativo — indica feature em roadmap (Coming Soon). */
  disabled?: boolean;
  /** Texto opcional do tooltip extra (ex.: "Em breve"). */
  hint?: string;
}

// Navegação única. "Assistente de IA" saiu do roadmap na RF-23 (action +
// Sheet). "Configurações" saiu do roadmap na RF-25 — página real
// (`/settings/alerts`), mesmo padrão de rota das demais páginas
// (`/history`, `/sensors/[id]`), não um Sheet. Fica por último como ponto
// de acesso final convencional.
export const NAV_ITEMS: NavItem[] = [
  { icon: LayoutDashboard, label: "Dashboard", href: "/" },
  { icon: Gauge, label: "Sensores", href: "/sensors/APU-Trem-042" },
  { icon: BarChart3, label: "Histórico", href: "/history" },
  { icon: FlaskConical, label: "Simulação", action: "simulation" },
  {
    icon: Sparkles,
    label: "Assistente de IA",
    action: "maintenance-assistant",
  },
  { icon: Settings, label: "Configurações", href: "/settings/alerts" },
];

export function isActive(href: string | undefined, pathname: string): boolean {
  if (!href) return false;
  if (href === "/") return pathname === "/";
  return pathname.startsWith(href);
}
