"use client";

/**
 * Limiar de decisão calibrado do modelo ativo — store compartilhado, sem
 * requisição própria. Quem já recebe o valor do backend o publica:
 *   - `use-sensor-data`, a cada poll de GET /v1/predictions (`active_threshold`);
 *   - o painel de modelos, logo após uma troca (GET /models).
 * `null` enquanto desconhecido — a UI cai nas faixas fixas.
 */

import { useSyncExternalStore } from "react";

let current: number | null = null;
const listeners = new Set<() => void>();

export const publishActiveThreshold = (value: number | null): void => {
  if (value === current) return;
  current = value;
  listeners.forEach((notify) => notify());
};

const subscribe = (notify: () => void): (() => void) => {
  listeners.add(notify);
  return () => listeners.delete(notify);
};

const snapshot = (): number | null => current;
const serverSnapshot = (): number | null => null;

export const useActiveThreshold = (): number | null =>
  useSyncExternalStore(subscribe, snapshot, serverSnapshot);
