# Benchmark de Modelos de Produção — PredictIQ

Gerado automaticamente por `python benchmark_models.py`. Reproduzível: execute o mesmo comando para regenerar este relatório com os mesmos dados.

## Resumo executivo

- **Modelo vencedor:** `random_forest_v2`
- **F1 (classe 1):** 0.9996
- **Latência p95:** 21.028 ms
- **Memória (tracemalloc, carregamento):** 0.03 MB
- **Justificativa:** melhor equilíbrio (score=0.9978) entre F1, latência e memória dentre os modelos elegíveis (p95 < 100ms).

## Tabela comparativa

| Modelo | F1 | Latência média (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Memória (MB) | Elegível | Resultado |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `random_forest_v2` | 0.9996 | 20.008 | 19.839 | 21.028 | 25.998 | 0.03 | ✅ | 🏆 Vencedor |
| `xgboost_v2` | 0.9996 | 20.176 | 20.035 | 21.153 | 26.477 | 0.02 | ✅ | Elegível |
| `mlp` | 0.9901 | 19.786 | 19.642 | 20.790 | 25.709 | 0.04 | ✅ | Elegível |
| `bilstm` | 0.9743 | 20.425 | 20.283 | 21.442 | 25.554 | 0.04 | ✅ | Elegível |
| `tcn` | 0.9675 | 20.890 | 20.776 | 21.762 | 27.418 | 0.02 | ✅ | Elegível |
| `patchtst` | 0.8070 | 20.375 | 20.218 | 21.331 | 26.839 | 0.03 | ✅ | Elegível |
| `random_forest` | 0.9996 | 47.056 | 46.687 | 48.847 | 88.356 | 19.73 | ✅ | Elegível |
| `autoencoder` | 0.4098 | 20.145 | 19.995 | 21.424 | 27.167 | 0.60 | ✅ | Elegível |
| `xgboost` | — | — | — | — | — | — | ❌ | Erro: `feature_names_in_` is defined only when `X` has feature names that are all strings. |

## Análise

- Melhor F1: `random_forest` (0.9996) (empate com `random_forest_v2`, `xgboost_v2`).
- Menor latência p95: `mlp` (20.790 ms).
- Menor memória: `xgboost_v2` (0.02 MB).
- Nenhum modelo violou o limite de 100ms.

`random_forest_v2` representa o melhor equilíbrio: entre os modelos elegíveis, combina F1, p95 e memória segundo o score documentado em `composite_score()` (pesos 0.6 F1 / 0.25 latência / 0.15 memória).

**Nota metodológica:** modelos sequenciais (TCN, BiLSTM, PatchTST) e o autoencoder são avaliados com o mesmo caminho de inferência real do PredictIQ (`InferencePipelineService._infer_with_history`), que hoje passa apenas a última leitura para `predict_from_features` — a janela temporal que esses modelos recebem em produção é, portanto, uma repetição cold-start da última leitura, não uma janela histórica real. Isso é uma característica já existente do sistema (não introduzida por este benchmark) e penaliza o F1 desses modelos aqui de forma consistente com o que acontece em produção hoje.

## Critério

**RF-18** — o modelo vencedor deve ter latência p95 < 100ms. Modelos que violam essa condição são marcados não elegíveis e nunca são escolhidos, independentemente do F1. Entre os elegíveis, vence o melhor equilíbrio F1/latência/memória (`select_winner()`), com desempate determinístico (maior F1 → menor p95 → menor memória) quando os scores diferem por menos de 0.01.

**RNF-36** — todo este processo (carregar modelos, medir, selecionar, gerar relatório, atualizar `ACTIVE_MODEL`) executa com um único comando: `python benchmark_models.py`.

Slice de avaliação: 2000 leituras (dataset original `apps/ml/data/processed/metropt3.parquet`, ordem temporal preservada, sem embaralhar, sem dados sintéticos) — contexto normal + a maior janela de falha real contígua do dataset.
