/**
 * Regra de negócio da exibição dos modelos no painel de Simulação.
 *
 * 1. O backend devolve uma lista de modelos brutos (ex.: `random_forest`,
 *    `random_forest_v2`, `xgboost`, `xgboost_v2`, `bilstm`).
 * 2. Quando coexiste uma `_v2`, a versão base é descartada — só a v2 sobrevive.
 * 3. O `value` enviado para a API mantém-se EXATAMENTE igual ao nome bruto.
 *    Apenas o texto exibido é higienizado por `formatModelName`.
 *
 * Tudo aqui é puro/sincrono para facilitar testes unitários.
 */

/** Sufixo de versão reconhecido na nomenclatura do backend. */
const VERSION_SUFFIXES = ["_v2", "_v1"] as const;

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
  for (const suffix of VERSION_SUFFIXES) {
    if (rawName.endsWith(suffix)) {
      return rawName.slice(0, -suffix.length);
    }
  }
  return rawName;
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
 * Regra: se existir `<stem>_v2` na lista bruta, qualquer entrada cujo stem
 * coincide e que NÃO seja a `_v2` é descartada. Em caso de duplicatas
 * residuais, mantemos a primeira ocorrência (ordem do backend é preservada).
 */
export function filterAndFormatModels(
  rawNames: readonly string[],
): DisplayModel[] {
  // 1. Indexa quais "stems" possuem versão v2 no payload.
  const stemsWithV2 = new Set<string>();
  for (const name of rawNames) {
    if (name.endsWith("_v2")) {
      stemsWithV2.add(stripVersionSuffix(name));
    }
  }

  // 2. Filtra fora qualquer entrada não-v2 cujo stem tenha um v2 concorrente.
  // 3. Deduplica preservando a primeira ocorrência.
  const seen = new Set<string>();
  const result: DisplayModel[] = [];

  for (const rawName of rawNames) {
    const stem = stripVersionSuffix(rawName);
    const isV2 = rawName.endsWith("_v2");

    if (stemsWithV2.has(stem) && !isV2) continue;
    if (seen.has(rawName)) continue;

    seen.add(rawName);
    result.push({ value: rawName, label: formatModelName(rawName) });
  }

  return result;
}
