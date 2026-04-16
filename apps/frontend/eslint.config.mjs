import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
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
    // Arquivo gerado pelo MSW
    "public/mockServiceWorker.js",
  ]),
]);

export default eslintConfig;
