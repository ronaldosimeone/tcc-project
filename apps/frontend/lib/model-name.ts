/**
 * Regra de negócio da exibição dos modelos no painel de Simulação.
 *
 * 1. O backend devolve uma lista de modelos brutos (ex.: `random_forest`,
 *    `random_forest_v2`, `xgboost`, `xgboost_v2`, `bilstm`).
 * 2. Para cada modelo só a maior versão sobrevive (`_v3` > `_v2` > sem sufixo).
 * 3. O `value` enviado para a API mantém-se EXATAMENTE igual ao nome bruto.
 *    Apenas o texto exibido é higienizado por `formatModelName`.
 *
 * Tudo aqui é puro/sincrono para facilitar testes unitários.
 */

/** Sufixo de versão da nomenclatura do backend (`_v2`, `_v3`, …). */
const VERSION_RE = /_v(\d+)$/;

/** Versão numérica do nome bruto; o nome sem sufixo é a versão 0 (legado). */
export const versionOf = (rawName: string): number => {
  const match = VERSION_RE.exec(rawName);
  return match ? Number(match[1]) : 0;
};

/**
 * Acrónimos preservados na sua forma idiomática.
 * Mantemos sempre `lowercase` como chave para casar com o nome bruto.
 */
const ACRONYM_MAP: Readonly<Record<string, string>> = {
  tcn: "TCN",
  mlp: "MLP",
  bilstm: "BiLSTM",
  patchtst: "PatchTST",
  xgboost: "XGBoost",
};

/** Modelo após o pipeline de filtragem — value cru + label limpa. */
export interface DisplayModel {
  /** Nome bruto enviado pelo backend (ex.: `random_forest_v2`). */
  readonly value: string;
  /** Texto pronto para o utilizador (ex.: `Random Forest`). */
  readonly label: string;
}

/**
 * Retorna o "stem" do modelo (sem `_v1`/`_v2`).
 * Ex.: `random_forest_v2` → `random_forest`.
 */
export function stripVersionSuffix(rawName: string): string {
  return rawName.replace(VERSION_RE, "");
}

/**
 * Higieniza o nome para exibição: remove sufixo de versão, converte
 * underscores em espaços e aplica capitalização ou acrônimo conhecido.
 */
export function formatModelName(rawName: string): string {
  const stem = stripVersionSuffix(rawName);

  const words = stem.split("_").filter(Boolean);
  if (words.length === 0) return rawName;

  return words
    .map((word) => {
      const lower = word.toLowerCase();
      if (lower in ACRONYM_MAP) return ACRONYM_MAP[lower];
      return lower.charAt(0).toUpperCase() + lower.slice(1);
    })
    .join(" ");
}

/**
 * Aplica o filtro de versão e devolve a lista pronta para o `<Select>`.
 *
 * Regra: para cada stem, só a MAIOR versão sobrevive (ex.: `random_forest`,
 * `random_forest_v2` e `random_forest_v3` → só `random_forest_v3`, o modelo
 * retreinado sem vazamento). A ordem do backend é preservada.
 */
export function filterAndFormatModels(
  rawNames: readonly string[],
): DisplayModel[] {
  const best = new Map<string, number>();
  for (const name of rawNames) {
    const stem = stripVersionSuffix(name);
    best.set(stem, Math.max(best.get(stem) ?? -1, versionOf(name)));
  }

  const seen = new Set<string>();
  const result: DisplayModel[] = [];
  for (const rawName of rawNames) {
    if (versionOf(rawName) !== best.get(stripVersionSuffix(rawName))) continue;
    if (seen.has(rawName)) continue;
    seen.add(rawName);
    result.push({ value: rawName, label: formatModelName(rawName) });
  }
  return result;
}
