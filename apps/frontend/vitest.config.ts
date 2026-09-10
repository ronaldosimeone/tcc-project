import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./vitest.setup.ts"],
    css: false,
    include: ["__tests__/**/*.{test,spec}.{ts,tsx}"],
    coverage: {
      provider: "v8",
      reporter: ["text", "lcov"],
      include: ["components/**/*.tsx", "lib/**/*.ts"],
      // components/ui/**: shadcn/ui vendorizado, não é código autoral.
      // *.stories.tsx (RNF-58): exercitadas pelo Storybook, não pelo
      // Vitest — sem isso, ficam 0% e derrubam artificialmente a métrica.
      exclude: ["components/ui/**", "**/*.stories.tsx"],
      // RNF-59: sem isso, o v8 provider SILENCIOSAMENTE pula o relatório de
      // cobertura sempre que qualquer teste falha — inclusive falhas
      // pré-existentes e não relacionadas (ver PENDENCIAS.md). A cobertura
      // real precisa ser medível mesmo com essas falhas presentes.
      reportOnFailure: true,
      // RNF-59: piso de 70% — mesmo padrão do backend
      // (apps/backend/pyproject.toml → [tool.coverage.report] fail_under).
      // `vitest run --coverage` sai com código != 0 se qualquer métrica
      // ficar abaixo disto, sem precisar de um step de CI separado só para
      // checar o número. Cobertura real medida no fechamento da RNF-59:
      // ~88% stmts / ~76% branch / ~88% funcs / ~91% lines — 70 dá margem
      // sem travar PRs por flutuações normais.
      thresholds: {
        statements: 70,
        branches: 70,
        functions: 70,
        lines: 70,
      },
    },
  },
  resolve: {
    alias: {
      "@": resolve(__dirname, "."),
    },
  },
});
