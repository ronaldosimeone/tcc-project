/**
 * Regra de versão dos modelos no seletor (lib/model-name.ts).
 *
 * Só a maior versão de cada modelo aparece: os retreinos sem vazamento
 * (`_v3` das árvores, `_v2` dos modelos profundos) escondem os legados.
 */
import { describe, expect, it } from "vitest";

import {
  filterAndFormatModels,
  formatModelName,
  stripVersionSuffix,
  versionOf,
} from "@/lib/model-name";

describe("model-name", () => {
  it("lê a versão numérica do sufixo", () => {
    expect(versionOf("random_forest")).toBe(0);
    expect(versionOf("random_forest_v2")).toBe(2);
    expect(versionOf("xgboost_v3")).toBe(3);
    expect(stripVersionSuffix("random_forest_v3")).toBe("random_forest");
  });

  it("mantém só a maior versão de cada modelo, na ordem do backend", () => {
    const out = filterAndFormatModels([
      "random_forest",
      "random_forest_v2",
      "random_forest_v3",
      "xgboost_v2",
      "xgboost_v3",
      "tcn",
      "tcn_v2",
      "autoencoder",
    ]);
    expect(out.map((m) => m.value)).toEqual([
      "random_forest_v3",
      "xgboost_v3",
      "tcn_v2",
      "autoencoder",
    ]);
  });

  it("o rótulo não mostra a versão", () => {
    expect(formatModelName("xgboost_v3")).toBe("XGBoost");
    expect(formatModelName("patchtst_v2")).toBe("PatchTST");
  });
});
