// ── Item de navegação — RNF-58: extraído de sidebar.tsx ─────────────────────

import Link from "next/link";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type { NavAction, NavItem } from "./constants";

interface NavLinkProps {
  item: NavItem;
  active: boolean;
  isOpen: boolean;
  onAction?: (action: NavAction) => void;
}

export function NavLink({ item, active, isOpen, onAction }: NavLinkProps) {
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
