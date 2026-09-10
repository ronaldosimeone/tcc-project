// For more info, see https://github.com/storybookjs/eslint-plugin-storybook#configuration-flat-config-format
import storybook from "eslint-plugin-storybook";

import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // RNF-58: guarda de complexidade ciclomática — complementar ao script
  // `scripts/check-component-size.mjs` (limite de 200 linhas/arquivo).
  // São checagens independentes: um arquivo pode ficar sob 200 linhas e
  // ainda assim acumular complexidade excessiva (muitos if/else/ternários
  // aninhados), e vice-versa. Threshold 20 = default do próprio ESLint para
  // esta regra — validado contra o código real do projeto (incl. parsers
  // como `lib/maintenance-stream.ts::parseSseBlock`, complexidade 19,
  // pré-existente e fora do escopo desta task): um threshold menor (ex. 15)
  // teria disparado falso positivo em lógica legítima já testada, sem
  // relação com o objetivo desta regra (evitar NOVOS componentes
  // excessivamente ramificados).
  {
    rules: {
      complexity: ["error", 20],
    },
  },
  // components/ui/** é shadcn/ui vendorizado (mesmo critério já usado em
  // vitest.config.ts → coverage.exclude e no script RNF-58) — não é código
  // autoral, não faz sentido aplicar a mesma régua de complexidade.
  {
    files: ["components/ui/**"],
    rules: {
      complexity: "off",
    },
  },
  // Bloqueio global de arquivos e pastas para o Linter
  globalIgnores([
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    "node_modules/**",
    // Pastas de relatórios e traces do Playwright
    "playwright-report/**",
    "test-results/**",
    "e2e-results/**",
    // Relatório HTML gerado por `vitest --coverage` (RNF-59)
    "coverage/**",
    // Build estático do Storybook (RNF-58) — gerado, não é código autoral.
    "storybook-static/**",
    // Arquivo gerado pelo MSW
    "public/mockServiceWorker.js",
  ]),
  ...storybook.configs["flat/recommended"],
]);

export default eslintConfig;
