"use client";

import { useSyncExternalStore } from "react";

const supported = (): boolean =>
  typeof window !== "undefined" && typeof window.matchMedia === "function";

/**
 * `true` enquanto a media query casa. No servidor, na hidratação e onde não
 * há `matchMedia` (jsdom) devolve `serverValue` — o layout de desktop por
 * padrão — e o cliente corrige logo em seguida, sem erro de hidratação.
 */
export const useMediaQuery = (query: string, serverValue = true): boolean =>
  useSyncExternalStore(
    (onChange) => {
      if (!supported()) return () => {};
      const mql = window.matchMedia(query);
      mql.addEventListener("change", onChange);
      return () => mql.removeEventListener("change", onChange);
    },
    () => (supported() ? window.matchMedia(query).matches : serverValue),
    () => serverValue,
  );
