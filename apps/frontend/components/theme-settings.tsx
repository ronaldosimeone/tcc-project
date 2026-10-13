"use client";

/**
 * Seção "Aparência" das Configurações — alterna entre tema claro, escuro
 * e o do sistema operacional. A preferência fica salva no navegador.
 */

import { Monitor, Moon, Sun } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { useTheme } from "@/hooks/use-theme";
import type { ThemePreference } from "@/lib/theme";

interface ThemeOption {
  value: ThemePreference;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}

const THEME_OPTIONS: ThemeOption[] = [
  { value: "light", label: "Claro", icon: Sun },
  { value: "dark", label: "Escuro", icon: Moon },
  { value: "system", label: "Sistema", icon: Monitor },
];

export function ThemeSettings() {
  const { preference, setPreference } = useTheme();

  return (
    <section
      aria-labelledby="theme-settings-title"
      className="flex flex-col gap-4 border-t border-border pt-6"
    >
      <div className="flex flex-col gap-1">
        <Label id="theme-settings-title" className="text-sm font-medium">
          Aparência
        </Label>
        <p className="text-xs text-muted-foreground">
          Escolha o tema da interface. A preferência fica salva neste navegador.
        </p>
      </div>

      <div
        role="group"
        aria-labelledby="theme-settings-title"
        className="flex gap-2"
      >
        {THEME_OPTIONS.map(({ value, label, icon: Icon }) => (
          <Button
            key={value}
            type="button"
            variant={preference === value ? "default" : "outline"}
            aria-pressed={preference === value}
            onClick={() => setPreference(value)}
          >
            <Icon className="h-4 w-4" aria-hidden="true" />
            {label}
          </Button>
        ))}
      </div>
    </section>
  );
}
