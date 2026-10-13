"use client";

/**
 * useTheme — preferência de tema (claro / escuro / sistema).
 *
 * A aplicação inicial da classe `dark` é feita pelo script inline do
 * layout (sem flash). Este hook só:
 *  - expõe a preferência persistida (useSyncExternalStore: SSR devolve o
 *    padrão, o cliente lê o localStorage sem divergência de hidratação);
 *  - aplica a mudança quando o usuário escolhe uma opção;
 *  - acompanha `prefers-color-scheme` enquanto a opção for "sistema".
 */

import { useCallback, useEffect, useSyncExternalStore } from "react";

import {
  DEFAULT_THEME,
  THEME_STORAGE_KEY,
  applyTheme,
  readStoredTheme,
  resolveTheme,
  type ThemePreference,
} from "@/lib/theme";

const DARK_QUERY = "(prefers-color-scheme: dark)";

const listeners = new Set<() => void>();

const subscribe = (callback: () => void): (() => void) => {
  listeners.add(callback);
  window.addEventListener("storage", callback);
  return () => {
    listeners.delete(callback);
    window.removeEventListener("storage", callback);
  };
};

const getServerSnapshot = (): ThemePreference => DEFAULT_THEME;

interface UseThemeReturn {
  preference: ThemePreference;
  setPreference: (preference: ThemePreference) => void;
}

export function useTheme(): UseThemeReturn {
  const preference = useSyncExternalStore(
    subscribe,
    readStoredTheme,
    getServerSnapshot,
  );

  const setPreference = useCallback((next: ThemePreference): void => {
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, next);
    } catch {
      // armazenamento indisponível — a escolha vale só para esta sessão
    }
    applyTheme(
      resolveTheme({
        preference: next,
        systemPrefersDark: window.matchMedia(DARK_QUERY).matches,
      }),
    );
    listeners.forEach((listener) => listener());
  }, []);

  useEffect(() => {
    if (preference !== "system") return;
    const media = window.matchMedia(DARK_QUERY);
    const sync = (): void =>
      applyTheme(
        resolveTheme({ preference, systemPrefersDark: media.matches }),
      );
    sync();
    media.addEventListener("change", sync);
    return () => media.removeEventListener("change", sync);
  }, [preference]);

  return { preference, setPreference };
}
