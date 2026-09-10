import tailwindcss from "@tailwindcss/vite";
import type { StorybookConfig } from "@storybook/nextjs-vite";

// RNF-58: histórias vivem ao lado dos componentes que documentam
// (`components/**/*.stories.tsx`), não numa pasta `stories/` separada —
// mais fácil de manter em sincronia com o componente real.
const config: StorybookConfig = {
  stories: ["../components/**/*.stories.@(ts|tsx)"],
  addons: ["@storybook/addon-a11y", "@storybook/addon-docs"],
  framework: "@storybook/nextjs-vite",
  staticDirs: ["../public"],
  // Sem isso, o `@theme inline { --color-primary: ...; ... }` de
  // app/globals.css (Tailwind v4) não é compilado no pipeline Vite do
  // Storybook — a build "funciona" mas TODAS as cores semânticas (primary,
  // destructive, muted, etc.) saem transparentes/erradas (confirmado via
  // inspeção do CSS gerado). `@tailwindcss/postcss` (usado pelo Next.js)
  // não é acionado pelo builder Vite do Storybook; o plugin oficial
  // `@tailwindcss/vite` é o caminho documentado para Vite puro.
  async viteFinal(viteConfig) {
    viteConfig.plugins = viteConfig.plugins ?? [];
    viteConfig.plugins.push(tailwindcss());
    return viteConfig;
  },
};
export default config;
