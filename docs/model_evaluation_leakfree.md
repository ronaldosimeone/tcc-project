# Avaliação sem vazamento de todos os modelos — matrizes de confusão

> Gerado por `python -m src.render_all_models_report` a partir de `apps/ml/models/eval_all_models.json`. Não edite à mão.

**Protocolo.** MetroPT-3 (2020) + MetroPT 2022, sensores harmonizados (sem `Reservoirs`/`Oil_level`), walk-forward purgado por evento: 7 falhas → **6 folds fora do tempo**. Cada fold treina só com o passado, escolhe o limiar só com dados de treino e é testado numa falha que o modelo nunca viu. Balanceamento de classes **apenas no treino**; o teste mantém a proporção real (≈ 2 % de amostras de falha). Árvores: limiar por custo em predições out-of-fold; redes: limiar por custo na falha de treino mais recente (validação temporal). Seed 0.

As matrizes somam os 6 folds de teste. A matriz do modelo final medida nos próprios dados de treino seria vazamento e não é reportada.

## Por evento — o que importa para a manutenção

| Modelo | Artefato | Falhas detectadas | Falhas perdidas | Episódios de falso alarme | Custo esperado médio |
|---|---|---|---|---|---|
| Random Forest | `random_forest_v3` | **3/6** | 3 | 56 | 59.3 |
| XGBoost | `xgboost_v3` | **2/6** | 4 | 4 | 67.3 |
| LogReg (baseline) | `—` | **3/6** | 3 | 29 | 54.8 |
| MLP | `mlp_v2` | **3/6** | 3 | 99 | 66.5 |
| TCN | `tcn_v2` | **3/6** | 3 | 14 | 52.3 |
| BiLSTM | `bilstm_v2` | **3/6** | 3 | 12 | 52.0 |
| PatchTST | `patchtst_v2` | **3/6** | 3 | 272 | 95.3 |
| Autoencoder | `autoencoder_v2` | **1/6** | 5 | 179 | 113.2 |

## Por amostra, após a persistência k = 6 (alarme visto pelo operador)

![Matrizes de confusão — alarmes](figures-ml/confusion_alarms.png)

| Modelo | TP | FP | FN | TN | Precisão | Recall | Especificidade | F1 |
|---|---|---|---|---|---|---|---|---|
| Random Forest | 3.304 | 16.074 | 39.209 | 1.831.527 | 17.1% | 7.8% | 99.1% | 10.7% |
| XGBoost | 5.409 | 11.929 | 37.104 | 1.835.672 | 31.2% | 12.7% | 99.4% | 18.1% |
| LogReg (baseline) | 20.636 | 15.413 | 21.877 | 1.832.188 | 57.2% | 48.5% | 99.2% | 52.5% |
| MLP | 3.143 | 14.508 | 39.370 | 1.833.093 | 17.8% | 7.4% | 99.2% | 10.4% |
| TCN | 4.163 | 26.165 | 38.350 | 1.821.436 | 13.7% | 9.8% | 98.6% | 11.4% |
| BiLSTM | 3.726 | 13.526 | 38.787 | 1.834.075 | 21.6% | 8.8% | 99.3% | 12.5% |
| PatchTST | 3.080 | 15.097 | 39.433 | 1.832.504 | 16.9% | 7.2% | 99.2% | 10.1% |
| Autoencoder | 1.064 | 15.244 | 41.449 | 1.832.357 | 6.5% | 2.5% | 99.2% | 3.6% |

## Por amostra, no limiar (sem persistência)

![Matrizes de confusão — amostras](figures-ml/confusion_samples.png)

| Modelo | TP | FP | FN | TN | Precisão | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Random Forest | 3.348 | 16.552 | 39.165 | 1.831.049 | 16.8% | 7.9% | 10.7% |
| XGBoost | 5.548 | 11.958 | 36.965 | 1.835.643 | 31.7% | 13.1% | 18.5% |
| LogReg (baseline) | 20.721 | 16.911 | 21.792 | 1.830.690 | 55.1% | 48.7% | 51.7% |
| MLP | 3.173 | 18.394 | 39.340 | 1.829.207 | 14.7% | 7.5% | 9.9% |
| TCN | 4.566 | 26.213 | 37.947 | 1.821.388 | 14.8% | 10.7% | 12.5% |
| BiLSTM | 5.348 | 13.596 | 37.165 | 1.834.005 | 28.2% | 12.6% | 17.4% |
| PatchTST | 3.091 | 21.066 | 39.422 | 1.826.535 | 12.8% | 7.3% | 9.3% |
| Autoencoder | 1.100 | 18.864 | 41.413 | 1.828.737 | 5.5% | 2.6% | 3.5% |

## Detecção por falha (cada fold testa uma falha)

| Modelo | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | fold 6 |
|---|---|---|---|---|---|---|
| Random Forest | ✅ (2020, limiar 0.5) | ❌ (2020, limiar 0.8) | ✅ (2020, limiar 0.1) | ✅ (2022, limiar 0.2) | ❌ (2022, limiar 0.25) | ❌ (2022, limiar 0.25) |
| XGBoost | ✅ (2020, limiar 0.5) | ✅ (2020, limiar 0.55) | ❌ (2020, limiar 0.8) | ❌ (2022, limiar 0.9) | ❌ (2022, limiar 0.6) | ❌ (2022, limiar 0.7) |
| LogReg (baseline) | ✅ (2020, limiar 0.5) | ✅ (2020, limiar 0.15) | ❌ (2020, limiar 0.8) | ✅ (2022, limiar 0.95) | ❌ (2022, limiar 0.35) | ❌ (2022, limiar 0.55) |
| MLP | ✅ (2020, limiar 0.5) | ✅ (2020, limiar 0.995) | ❌ (2020, limiar 0.99) | ✅ (2022, limiar 0.5) | ❌ (2022, limiar 0.995) | ❌ (2022, limiar 0.995) |
| TCN | ✅ (2020, limiar 0.5) | ✅ (2020, limiar 0.995) | ❌ (2020, limiar 0.99) | ✅ (2022, limiar 0.995) | ❌ (2022, limiar 0.995) | ❌ (2022, limiar 0.995) |
| BiLSTM | ✅ (2020, limiar 0.5) | ✅ (2020, limiar 0.995) | ❌ (2020, limiar 0.995) | ✅ (2022, limiar 0.995) | ❌ (2022, limiar 0.995) | ❌ (2022, limiar 0.995) |
| PatchTST | ✅ (2020, limiar 0.5) | ✅ (2020, limiar 0.995) | ❌ (2020, limiar 0.995) | ✅ (2022, limiar 0.995) | ❌ (2022, limiar 0.97) | ❌ (2022, limiar 0.995) |
| Autoencoder | ❌ (2020, limiar 0.203) | ❌ (2020, limiar 0.497) | ❌ (2020, limiar 0.302) | ✅ (2022, limiar 0.112) | ❌ (2022, limiar 0.231) | ❌ (2022, limiar 0.179) |

Folds 1–3 testam as falhas 2–4 de 2020; folds 4–6 testam as 3 falhas de 2022 (vazamento de ar nos consumidores, válvula do secador de ar, vazamento de óleo).

## Gate RN-05 contra o champion (`random_forest_v3`)

Wilcoxon unilateral pareado por fold, custo esperado; p mínimo atingível com 6 folds = 0,016.

| Candidato | Melhora mediana | p | Promove? |
|---|---|---|---|
| XGBoost | -0.5 | 0.688 | não |
| LogReg (baseline) | -4.0 | 0.844 | não |
| MLP | -0.5 | 0.688 | não |
| TCN | +0.0 | 0.562 | não |
| BiLSTM | +0.0 | 0.562 | não |
| PatchTST | -1.5 | 0.844 | não |
| Autoencoder | -21.5 | 1.000 | não |

## Leitura

* **Nenhum modelo detecta as falhas 2 e 3 de 2022** (folds 5 e 6: válvula do secador de ar, 30 min; vazamento de óleo). São modos de falha ausentes do treino. O teto atual por evento é 3 de 6 falhas.
* **Nenhum candidato supera o Random Forest com significância**; por isso o champion continua `random_forest_v3`. BiLSTM e TCN têm o menor custo médio e menos episódios de falso alarme, mas a diferença por fold é nula na mediana.
* **A regressão logística tem o maior recall por amostra**: acompanha a longa falha de óleo amostra a amostra, mas sem disparar o evento dentro do horizonte de antecedência.
* **Redes superconfiantes**: TCN, BiLSTM, MLP e PatchTST escolheram o limiar no topo da grade (0,995) na maioria dos folds; os logits chegam a centenas, e a probabilidade fica quase binária. A calibração (por exemplo, *temperature scaling* na validação temporal) é trabalho futuro.
* **Cards legados** (F1 ≈ 0,96–0,98) vinham de split aleatório, com o limiar ajustado no próprio teste. Os números acima são a medida honesta do que cada modelo faz numa falha futura.

