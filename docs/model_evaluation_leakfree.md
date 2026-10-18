# Avaliação de doutorado de todos os modelos — protocolo aninhado

> Gerado por `python -m src.render_nested_report` a partir de `apps/ml/models/nested/*.json`. Não edite à mão.

## Protocolo

* **Dados**: MetroPT-3 (2020) + MetroPT 2022, sensores harmonizados (sem `Reservoirs`/`Oil_level`, que mudam de significado entre os datasets).
* **Loop externo**: walk-forward purgado por evento, 7 falhas → **6 folds fora do tempo**; cada fold treina só com o passado e é testado numa falha que o modelo nunca viu.
* **Loop interno** (só nos dados de treino): leave-one-failure-out para (i) escolher os hiperparâmetros pelo custo esperado RN-02, (ii) calibrar por Platt e (iii) escolher o limiar sobre a probabilidade calibrada. Os parâmetros antigos do Optuna (escolhidos num split aleatório) foram descartados.
* **Mesmo protocolo para os 8 modelos**, árvores e redes; **3 seeds** (a seed 0 escolhe os hiperparâmetros, as seeds 1–2 medem a variância do treino). Balanceamento de classes só no treino; teste na prevalência real.
* **Fold 1** tem uma única falha de treino: não há o que validar internamente, então usa os hiperparâmetros padrão e limiar a priori (registrado como *fallback*).

## Por evento — 6 falhas fora do tempo, média de 3 seeds

| Modelo | Artefato | Falhas detectadas (média [mín; máx]) | Episódios de falso alarme | Custo esperado médio | ECE bruto → calibrado |
|---|---|---|---|---|---|
| Random Forest | `random_forest_v3` | **2.0** [2; 2] | 30 | 71.6 | 0.028 → 0.033 |
| XGBoost | `xgboost_v3` | **3.3** [3; 4] | 12 | 46.5 | 0.029 → 0.036 |
| LogReg | `—` | **2.0** [2; 2] | 16 | 69.3 | 0.021 → 0.037 |
| MLP | `mlp_v2` | **3.0** [3; 3] | 10 | 51.7 | 0.023 → 0.036 |
| TCN | `tcn_v2` | **2.0** [2; 2] | 18 | 69.7 | 0.023 → 0.036 |
| BiLSTM | `bilstm_v2` | **3.0** [3; 3] | 19 | 53.2 | 0.031 → 0.035 |
| PatchTST | `patchtst_v2` | **3.0** [3; 3] | 127 | 71.2 | 0.031 → 0.039 |
| Autoencoder | `autoencoder_v2` | **2.0** [2; 2] | 216 | 102.7 | — |

## Matrizes de confusão — por amostra após k = 6 (seed 0)

![Matrizes de confusão — alarmes](figures-ml/confusion_alarms.png)

| Modelo | TP | FP | FN | TN | Precisão | Recall | Especificidade | F1 |
|---|---|---|---|---|---|---|---|---|
| Random Forest | 3.359 | 14.098 | 39.154 | 1.833.503 | 19.2% | 7.9% | 99.2% | 11.2% |
| XGBoost | 2.389 | 12.007 | 40.124 | 1.835.594 | 16.6% | 5.6% | 99.4% | 8.4% |
| LogReg | 3.594 | 14.345 | 38.919 | 1.833.256 | 20.0% | 8.5% | 99.2% | 11.9% |
| MLP | 3.058 | 12.280 | 39.455 | 1.835.321 | 19.9% | 7.2% | 99.3% | 10.6% |
| TCN | 3.093 | 25.687 | 39.420 | 1.821.914 | 10.7% | 7.3% | 98.6% | 8.7% |
| BiLSTM | 5.380 | 13.393 | 37.133 | 1.834.208 | 28.7% | 12.7% | 99.3% | 17.6% |
| PatchTST | 3.006 | 21.782 | 39.507 | 1.825.819 | 12.1% | 7.1% | 98.8% | 8.9% |
| Autoencoder | 2.686 | 481.647 | 39.827 | 1.365.954 | 0.6% | 6.3% | 73.9% | 1.0% |

Por amostra no limiar, sem persistência: ![Matrizes de confusão — amostras](figures-ml/confusion_samples.png)

## Detecção por falha (seed 0) e hiperparâmetros escolhidos

| Modelo | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | fold 6 |
|---|---|---|---|---|---|---|
| Random Forest | ✅ `max_depth=20, min_samples_leaf=1, max_features=sqrt` | ❌ `max_depth=None, min_samples_leaf=20, max_features=sqrt` | ❌ `max_depth=None, min_samples_leaf=20, max_features=sqrt` | ✅ `max_depth=10, min_samples_leaf=1, max_features=sqrt` | ❌ `max_depth=20, min_samples_leaf=1, max_features=sqrt` | ❌ `max_depth=20, min_samples_leaf=1, max_features=sqrt` |
| XGBoost | ✅ `max_depth=6, learning_rate=0.1, n_estimators=300, subsample=0.9, colsample_bytree=0.8` | ✅ `max_depth=6, learning_rate=0.2, n_estimators=150, subsample=0.7, colsample_bytree=0.5` | ✅ `max_depth=4, learning_rate=0.05, n_estimators=600, subsample=0.8, colsample_bytree=0.6` | ✅ `max_depth=6, learning_rate=0.1, n_estimators=300, subsample=0.9, colsample_bytree=0.8` | ❌ `max_depth=4, learning_rate=0.05, n_estimators=600, subsample=0.8, colsample_bytree=0.6` | ❌ `max_depth=6, learning_rate=0.2, n_estimators=150, subsample=0.7, colsample_bytree=0.5` |
| LogReg | ✅ `C=1.0` | ❌ `C=0.1` | ❌ `C=10.0` | ✅ `C=0.01` | ❌ `C=0.01` | ❌ `C=0.01` |
| MLP | ✅ `epochs=8, lr=0.001` | ✅ `epochs=16, lr=0.0003` | ❌ `epochs=8, lr=0.001` | ✅ `epochs=16, lr=0.0003` | ❌ `epochs=8, lr=0.001` | ❌ `epochs=8, lr=0.001` |
| TCN | ✅ `epochs=8, lr=0.001` | ❌ `epochs=16, lr=0.0003` | ❌ `epochs=8, lr=0.001` | ✅ `epochs=16, lr=0.0003` | ❌ `epochs=16, lr=0.0003` | ❌ `epochs=8, lr=0.001` |
| BiLSTM | ✅ `epochs=8, lr=0.001` | ✅ `epochs=8, lr=0.001` | ❌ `epochs=16, lr=0.0003` | ✅ `epochs=16, lr=0.0003` | ❌ `epochs=8, lr=0.001` | ❌ `epochs=8, lr=0.001` |
| PatchTST | ✅ `epochs=8, lr=0.001` | ✅ `epochs=8, lr=0.001` | ❌ `epochs=16, lr=0.0003` | ✅ `epochs=16, lr=0.0003` | ❌ `epochs=16, lr=0.0003` | ❌ `epochs=8, lr=0.001` |
| Autoencoder | ❌ `epochs=8, lr=0.001` | ❌ `epochs=8, lr=0.001` | ✅ `epochs=16, lr=0.0003` | ✅ `epochs=8, lr=0.001` | ❌ `epochs=8, lr=0.001` | ❌ `epochs=8, lr=0.001` |

Folds 1–3: falhas 2–4 de 2020. Folds 4–6: as 3 falhas de 2022 (vazamento de ar nos consumidores, válvula do secador de ar, vazamento de óleo).

## Gate RN-05 contra o champion (`random_forest_v3`)

Wilcoxon unilateral pareado por fold (custo médio das 3 seeds); p mínimo atingível com 6 folds = 0,016.

| Candidato | Melhora mediana | p | Promove? |
|---|---|---|---|
| XGBoost | +12.8 | 0.125 | não |
| LogReg | -1.0 | 0.781 | não |
| MLP | +0.0 | 0.250 | não |
| TCN | +0.0 | 0.500 | não |
| BiLSTM | +0.0 | 0.312 | não |
| PatchTST | +0.0 | 0.562 | não |
| Autoencoder | -3.7 | 0.922 | não |

## Sensibilidade às escolhas de avaliação (k, horizonte, razão de custo)

Grade: k ∈ [3, 6, 12], horizonte ∈ [2.0, 6.0, 12.0] h, C_FN/C_FP ∈ [10.0, 100.0, 1000.0] — 27 configurações; o limiar é re-escolhido nas predições out-of-fold calibradas de cada configuração (sem retreino).

* Estabilidade do ranking por custo (τ de Kendall contra a referência k = 6, 6 h, 100): mediana **0.50**, mínimo 0.21.
* Melhor modelo por configuração: XGBoost em 13, BiLSTM em 6, MLP em 6, Autoencoder em 2.

| Modelo | Falhas detectadas (mín–máx entre configurações) |
|---|---|
| Random Forest | 1.0 – 2.7 |
| XGBoost | 1.7 – 4.3 |
| LogReg | 2.0 – 3.0 |
| MLP | 3.0 – 4.0 |
| TCN | 2.0 – 2.0 |
| BiLSTM | 2.3 – 3.7 |
| PatchTST | 3.0 – 3.7 |
| Autoencoder | 0.0 – 5.0 |

## Incerteza — conformal estático × adaptativo (ACI), alvo 90%

Conjunto de calibração = predições out-of-fold calibradas (só treino); fluxo de teste fora do tempo. Hipótese: o rótulo de cada linha fica conhecido logo depois (na operação, chega com o veredito da OS).

| Modelo | Cobertura da classe falha: estático → ACI | Cobertura normal: estático → ACI | Tamanho médio do conjunto: estático → ACI |
|---|---|---|---|
| Random Forest | 0.51 → **0.90** | 0.91 → 0.90 | 0.93 → 1.20 |
| XGBoost | 0.46 → **0.90** | 0.83 → 0.90 | 0.84 → 1.03 |
| LogReg | 0.38 → **0.89** | 0.68 → 0.90 | 0.70 → 1.15 |
| MLP | 0.21 → **0.89** | 0.94 → 0.90 | 0.93 → 1.06 |
| TCN | 0.24 → **0.89** | 0.91 → 0.90 | 1.04 → 1.26 |
| BiLSTM | 0.19 → **0.88** | 0.88 → 0.90 | 0.92 → 1.18 |
| PatchTST | 0.12 → **0.88** | 0.94 → 0.90 | 0.96 → 1.16 |
| Autoencoder | 0.54 → **0.91** | 0.95 → 0.90 | 1.06 → 1.15 |

## RUL com as 7 falhas (6 folds)

* XGBoost AFT: C-index 0.48 [0.45; 0.51].
* Baseline de renovação: C-index 0.65 [0.52; 0.81].

Com 7 falhas o tempo até a falha continua não aprendível pelas features (C-index ≈ 0,5): o resultado negativo do RF-30 se mantém com o dobro de folds.
