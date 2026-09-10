/**
 * check-component-size.mjs — RNF-58: nenhum componente de produção pode
 * exceder 200 linhas.
 *
 * "Componente" = arquivo `.tsx` que exporta um componente React, sob
 * `app/` ou `components/`. Excluídos (não são código de produção):
 *   - testes (`__tests__/`, `*.test.tsx`, `*.spec.tsx`)
 *   - Storybook (`*.stories.tsx`)
 *   - mocks/fixtures (`mocks/`, `__mocks__/`, `fixtures/`)
 *   - `components/ui/**` — shadcn/ui vendorizado (mesmo critério já usado
 *     em `vitest.config.ts` → `coverage.exclude`), não é código autoral.
 *   - arquivos gerados (`*.generated.tsx`)
 *
 * Uso:
 *   node scripts/check-component-size.mjs
 *
 * Saída: lista os arquivos violadores (caminho + contagem de linhas) e
 * sai com código 1. Sem violações → código 0 e mensagem de sucesso.
 * Usado localmente e no CI (.github/workflows/ci.yml).
 */
import { readFileSync, readdirSync } from "node:fs";
import { dirname, join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";

const MAX_LINES = 200;
const ROOT = dirname(dirname(fileURLToPath(import.meta.url))); // apps/frontend
const SCAN_DIRS = ["app", "components"];

const EXCLUDE_SEGMENTS = ["__tests__", "__mocks__", "mocks", "fixtures", "ui"];

function isExcluded(relPath) {
  const segments = relPath.split(sep);
  if (segments.some((s) => EXCLUDE_SEGMENTS.includes(s))) return true;
  const base = segments[segments.length - 1];
  if (base.endsWith(".test.tsx")) return true;
  if (base.endsWith(".spec.tsx")) return true;
  if (base.endsWith(".stories.tsx")) return true;
  if (base.endsWith(".generated.tsx")) return true;
  return false;
}

function countLines(filePath) {
  const content = readFileSync(filePath, "utf8");
  if (content.length === 0) return 0;
  // Conta quebras de linha; +1 se o arquivo não terminar em "\n" (última
  // linha sem newline final ainda conta como linha).
  const newlines = (content.match(/\n/g) ?? []).length;
  return content.endsWith("\n") ? newlines : newlines + 1;
}

function collectTsxFiles(dir) {
  let entries;
  try {
    entries = readdirSync(dir, { withFileTypes: true, recursive: true });
  } catch {
    return [];
  }
  const files = [];
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith(".tsx")) continue;
    // Node <20.something reports `entry.path`/`parentPath` inconsistently
    // across versions with `recursive: true`; resolve via readdirSync's own
    // base + name to stay version-agnostic.
    const parentPath = entry.parentPath ?? entry.path ?? dir;
    files.push(join(parentPath, entry.name));
  }
  return files;
}

const violations = [];

for (const dir of SCAN_DIRS) {
  const absDir = join(ROOT, dir);
  for (const filePath of collectTsxFiles(absDir)) {
    const relPath = relative(ROOT, filePath);
    if (isExcluded(relPath)) continue;
    const lines = countLines(filePath);
    if (lines > MAX_LINES) {
      violations.push({ relPath, lines });
    }
  }
}

if (violations.length > 0) {
  violations.sort((a, b) => b.lines - a.lines);
  console.error(
    `RNF-58: ${violations.length} componente(s) excedem o limite de ${MAX_LINES} linhas:\n`,
  );
  for (const { relPath, lines } of violations) {
    console.error(`  ${lines.toString().padStart(4)} linhas  ${relPath}`);
  }
  console.error(
    "\nDecomponha os arquivos acima (extraia sub-componentes/hooks/helpers) — ver CLAUDE.md.",
  );
  process.exit(1);
}

console.log(`RNF-58: OK — nenhum componente excede ${MAX_LINES} linhas.`);
process.exit(0);
