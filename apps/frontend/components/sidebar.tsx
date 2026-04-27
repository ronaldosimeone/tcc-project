"use client";

import { BarChart3, Gauge, LayoutDashboard, Settings } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

interface NavItem {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  href?: string;
}

const NAV_ITEMS: NavItem[] = [
  { icon: LayoutDashboard, label: "Dashboard", href: "/" },
  { icon: Gauge, label: "Sensores", href: "/sensors/APU-Trem-042" },
  { icon: BarChart3, label: "Histórico", href: "/history" },
  { icon: Settings, label: "Configurações" },
];

function isActive(href: string | undefined, pathname: string): boolean {
  if (!href) return false;
  if (href === "/") return pathname === "/";
  return pathname.startsWith(href);
}

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex w-16 flex-col items-center gap-1 border-r border-border bg-card py-4 lg:w-56 lg:items-start lg:px-3">
      <nav className="flex w-full flex-col gap-1">
        {NAV_ITEMS.map(({ icon: Icon, label, href }) => {
          const active = isActive(href, pathname);
          const className = [
            "flex h-10 w-full items-center gap-3 rounded-md px-3 text-sm font-medium transition-colors",
            active
              ? "bg-primary/10 text-primary ring-1 ring-inset ring-primary/20"
              : "text-muted-foreground hover:bg-accent hover:text-foreground",
          ].join(" ");

          if (href) {
            return (
              <Link key={label} href={href} className={className}>
                <Icon className="h-4 w-4 shrink-0" />
                <span className="hidden lg:inline">{label}</span>
              </Link>
            );
          }

          return (
            <button key={label} className={className}>
              <Icon className="h-4 w-4 shrink-0" />
              <span className="hidden lg:inline">{label}</span>
            </button>
          );
        })}
      </nav>

      <div className="mt-auto hidden w-full rounded-md border border-border bg-muted/30 p-3 lg:block">
        <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
          Modelo
        </p>
        <p className="mt-0.5 truncate text-xs text-foreground/80">
          Random Forest · v1.0
        </p>
        <p className="mt-0.5 text-[10px] text-muted-foreground">
          12 sensores · anomaly
        </p>
      </div>
    </aside>
  );
}
