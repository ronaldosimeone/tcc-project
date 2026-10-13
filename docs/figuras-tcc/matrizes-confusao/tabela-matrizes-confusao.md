| Modelo | Unidade | n | Limiar | VN | FP | FN | VP | Precisão | Recall | F1 | Verificação |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Random Forest | tabular | 303390 | 0.6598 | 297347 | 52 | 23 | 5968 | 0.9914 | 0.9962 | 0.9938 | derivado do model card |
| XGBoost | tabular | 303390 | 0.5909 | 297350 | 49 | 25 | 5966 | 0.9919 | 0.9958 | 0.9938 | reexecutado, idêntico |
| MLP | tabular | 303390 | 0.8364 | 296876 | 523 | 88 | 5903 | 0.9186 | 0.9853 | 0.9508 | derivado do model card |
| TCN | janelas | 30338 | 0.9069 | 29677 | 57 | 10 | 594 | 0.9124 | 0.9834 | 0.9466 | reexecutado, ±1 amostra |
| BiLSTM | janelas | 30338 | 0.7420 | 29641 | 93 | 10 | 594 | 0.8646 | 0.9834 | 0.9202 | reexecutado, ±1 amostra |
| PatchTST | janelas | 30338 | 0.9171 | 29663 | 71 | 16 | 588 | 0.8923 | 0.9735 | 0.9311 | reexecutado, idêntico |
| Autoencoder | janelas | 30338 | 0.0629 | 29436 | 298 | 588 | 16 | 0.0510 | 0.0265 | 0.0349 | origem: recalculado |
