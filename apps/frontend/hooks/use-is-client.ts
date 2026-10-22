"use client";

import { useSyncExternalStore } from "react";

const subscribe = (): (() => void) => () => {};

/**
 * `false` no servidor e na hidratação, `true` depois. Para conteúdo que
 * depende do relógio ou do fuso do navegador (o servidor roda em UTC): sem
 * isso o HTML do servidor ficava na tela e os horários apareciam 3 h à frente.
 */
export const useIsClient = (): boolean =>
  useSyncExternalStore(
    subscribe,
    () => true,
    () => false,
  );
