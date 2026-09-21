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
      // RNF-66/RNF-67: piso elevado de 70% -> 75% (meta de qualidade da
      // auditoria de acessibilidade). `vitest run --coverage` sai com
      // código != 0 se qualquer métrica ficar abaixo disto. Cobertura real
      // no fechamento da RNF-66/67: ~88% stmts / ~76% branch / ~88% funcs /
      // ~91% lines — branch fica com pouca margem (~1pp) sobre o piso; se
      // flutuar abaixo de 75% num PR futuro sem regressão real de teste,
      // considere lift do piso após adicionar cobertura, não relaxar aqui.
      thresholds: {
        statements: 75,
        branches: 75,
        functions: 75,
        lines: 75,
      },
    },
  },
  resolve: {
    alias: {
      "@": resolve(__dirname, "."),
    },
  },
});
