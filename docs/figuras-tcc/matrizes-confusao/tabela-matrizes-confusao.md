# Matrizes de confusão — protocolo oficial (walk-forward aninhado)

> Gerado por `docs/figuras-tcc/scripts/plotar_matrizes_confusao.py` a partir de
> `apps/ml/models/nested/*.json` (seed 0). Não edite à mão.

Acerto por leitura nos 6 folds de teste. A métrica principal do TCC é por
evento (falhas detectadas de 6); ver `docs/model_evaluation_leakfree.md`.

| Detector | Falhas detectadas | Leitura | VN | FP | FN | VP | Precisão | Recall | F1 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 2/6 | alarmes k = 6 | 1.833.717 | 13.884 | 39.203 | 3.310 | 19,3% | 7,8% | 11,1% |
| Random Forest | 2/6 | por leitura | 1.833.534 | 14.067 | 39.178 | 3.335 | 19,2% | 7,8% | 11,1% |
| XGBoost | 3/6 | alarmes k = 6 | 1.835.711 | 11.890 | 39.676 | 2.837 | 19,3% | 6,7% | 9,9% |
| XGBoost | 3/6 | por leitura | 1.835.689 | 11.912 | 38.508 | 4.005 | 25,2% | 9,4% | 13,7% |
| Regressão Logística | 2/6 | alarmes k = 6 | 1.832.947 | 14.654 | 38.930 | 3.583 | 19,6% | 8,4% | 11,8% |
| Regressão Logística | 2/6 | por leitura | 1.832.422 | 15.179 | 38.866 | 3.647 | 19,4% | 8,6% | 11,9% |
| MLP | 3/6 | alarmes k = 6 | 1.833.022 | 14.579 | 39.400 | 3.113 | 17,6% | 7,3% | 10,3% |
| MLP | 3/6 | por leitura | 1.831.513 | 16.088 | 39.390 | 3.123 | 16,3% | 7,3% | 10,1% |
| TCN | 2/6 | alarmes k = 6 | 1.822.304 | 25.297 | 39.424 | 3.089 | 10,9% | 7,3% | 8,7% |
| TCN | 2/6 | por leitura | 1.822.186 | 25.415 | 39.414 | 3.099 | 10,9% | 7,3% | 8,7% |
| BiLSTM | 3/6 | alarmes k = 6 | 1.835.477 | 12.124 | 29.984 | 12.529 | 50,8% | 29,5% | 37,3% |
| BiLSTM | 3/6 | por leitura | 1.835.430 | 12.171 | 29.872 | 12.641 | 50,9% | 29,7% | 37,6% |
| PatchTST | 3/6 | alarmes k = 6 | 1.834.016 | 13.585 | 39.639 | 2.874 | 17,5% | 6,8% | 9,7% |
| PatchTST | 3/6 | por leitura | 1.830.550 | 17.051 | 39.599 | 2.914 | 14,6% | 6,9% | 9,3% |
| Autoencoder | 1/6 | alarmes k = 6 | 1.827.423 | 20.178 | 41.316 | 1.197 | 5,6% | 2,8% | 3,7% |
| Autoencoder | 1/6 | por leitura | 1.825.954 | 21.647 | 41.294 | 1.219 | 5,3% | 2,9% | 3,7% |
| CUSUM | 2/6 | alarmes k = 6 | 1.825.900 | 21.701 | 35.801 | 6.712 | 23,6% | 15,8% | 18,9% |
| CUSUM | 2/6 | por leitura | 1.825.865 | 21.736 | 35.791 | 6.722 | 23,6% | 15,8% | 18,9% |
| EWMA | 4/6 | alarmes k = 6 | 1.825.593 | 22.008 | 24.583 | 17.930 | 44,9% | 42,2% | 43,5% |
| EWMA | 4/6 | por leitura | 1.825.413 | 22.188 | 24.548 | 17.965 | 44,7% | 42,3% | 43,5% |
