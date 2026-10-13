/**
 * Tema claro/escuro — preferência persistida em localStorage.
 *
 * O CSS já define as variáveis do tema escuro em `.dark` (app/globals.css);
 * este módulo só decide QUANDO a classe `dark` é aplicada em <html>.
 * O padrão é "light" para preservar o visual original do produto.
 */

export type ThemePreference = "light" | "dark" | "system";
export type ResolvedTheme = "light" | "dark";

export const THEME_STORAGE_KEY = "predictiq-theme";
export const DEFAULT_THEME: ThemePreference = "light";

export const isThemePreference = (value: unknown): value is ThemePreference =>
  value === "light" || value === "dark" || value === "system";

export const resolveTheme = ({
  preference,
  systemPrefersDark,
}: {
  preference: ThemePreference;
  systemPrefersDark: boolean;
}): ResolvedTheme => {
  if (preference === "system") return systemPrefersDark ? "dark" : "light";
  return preference;
};

export const readStoredTheme = (): ThemePreference => {
  try {
    const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
    return isThemePreference(stored) ? stored : DEFAULT_THEME;
  } catch {
    return DEFAULT_THEME;
  }
};

export const applyTheme = (theme: ResolvedTheme): void => {
  const root = document.documentElement;
  root.classList.toggle("dark", theme === "dark");
  root.style.colorScheme = theme;
};

/**
 * Script inline executado antes da hidratação (em <head>) para aplicar a
 * classe `dark` sem "flash" de tema claro. Mantém a mesma regra de
 * `resolveTheme`/`readStoredTheme` acima.
 */
export const THEME_INIT_SCRIPT = `(function(){try{var p=localStorage.getItem("${THEME_STORAGE_KEY}");var d=p==="dark"||(p==="system"&&window.matchMedia("(prefers-color-scheme: dark)").matches);var r=document.documentElement;r.classList.toggle("dark",d);r.style.colorScheme=d?"dark":"light";}catch(e){}})();`;
