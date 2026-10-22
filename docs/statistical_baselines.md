# Baselines estatísticos (CUSUM / EWMA) no protocolo aninhado

> Gerado por `python -m src.render_baselines_report` a partir de `apps/ml/models/nested/*.json`. Não edite à mão.

## Pergunta

A complexidade dos modelos de ML/DL se paga com 7 falhas? O benchmark público do MetroPT-3 recomenda um CUSUM na corrente do motor + z-score da temperatura do óleo (4/4 falhas, 0,71 FA/semana), mas escolheu os detectores olhando as mesmas 4 falhas que testou.

## Protocolo (o mesmo dos 8 modelos)

* Walk-forward purgado externo (6 folds fora do tempo, MetroPT-3 + 2022).
* Tudo o que é livre é escolhido **só no treino**, por leave-one-failure-out interno e custo RN-02: sensor (corrente do motor, temperatura do óleo, TP2, ou corrente + óleo), `k` do CUSUM ∈ {0,5; 1,0} ou `λ` do EWMA ∈ {0,01; 0,1}, referência (estática = média/desvio do treino saudável; móvel = 7 dias anteriores, causal, com 1 h de defasagem), e o limiar `h` (Platt + ótimo de custo nos OOF internos).
* Sinal = média móvel causal de 10 min dentro de cada segmento contínuo (lacuna > 60 s reinicia), estatística reiniciada em cada segmento.
* Fold 1 tem só 1 falha no treino: limiar a priori (percentil 99 da estatística no treino saudável), parâmetros a priori = escolha do benchmark (corrente + óleo, k = 0,5 / λ = 0,1, referência móvel). Ressalva: essa escolha a priori veio de um trabalho que viu as falhas do MetroPT-3, inclusive a do fold 1; os folds 2–6 não dependem dela.
* Determinísticos: 1 seed. Espaço de busca fixado antes de rodar.

## Resultado (rótulos originais)

| Detector / modelo | Falhas detectadas | Alarmes falsos (total) | FA/dia (média dos folds) | Custo esperado médio |
|---|---|---|---|---|
| `cusum` | 2.0/6 | 74.0 | 0.47 | 79.0 |
| `ewma` | 4.0/6 | 101.0 | 0.64 | 50.2 |
| `random_forest` | 2.0/6 | 48.0 | 0.28 | 74.7 |
| `xgboost` | 2.7/6 | 37.0 | 0.26 | 61.7 |
| `logistic_regression` | 2.0/6 | 63.0 | 0.36 | 77.2 |
| `mlp` | 3.0/6 | 178.7 | 0.60 | 79.8 |
| `tcn` | 2.0/6 | 154.0 | 0.79 | 92.3 |
| `bilstm` | 3.0/6 | 45.3 | 0.29 | 57.6 |
| `patchtst` | 3.0/6 | 66.0 | 0.45 | 61.0 |
| `autoencoder` | 1.0/6 | 202.7 | 1.04 | 117.1 |

## Por fold

| Fold | Teste | `cusum` det / FA / custo | `ewma` det / FA / custo | `random_forest` custo |
|---|---|---|---|---|
| 1 | 2020 | 0/1 / 0 / 100 | 1/1 / 19 / 19 | 36 |
| 2 | 2020 | 1/1 / 56 / 56 | 1/1 / 64 / 64 | 100 |
| 3 | 2020 | 1/1 / 0 / 0 | 1/1 / 3 / 3 | 100 |
| 4 | 2022 | 0/1 / 0 / 100 | 1/1 / 2 / 2 | 12 |
| 5 | 2022 | 0/1 / 0 / 100 | 0/1 / 0 / 100 | 100 |
| 6 | 2022 | 0/1 / 18 / 118 | 0/1 / 13 / 113 | 100 |

## Parâmetros escolhidos no loop interno

**`cusum`**

| Fold | Parâmetros | Origem do limiar |
|---|---|---|
| 1 | sensors=Motor_current+Oil_temperature, k=0.5, reference=rolling | fallback (<2 failures in training: a-priori threshold, uncalibrated) |
| 2 | sensors=TP2, k=1.0, reference=rolling | nested LOFO OOF (2 inner folds) |
| 3 | sensors=Oil_temperature, k=1.0, reference=rolling | nested LOFO OOF (3 inner folds) |
| 4 | sensors=Oil_temperature, k=1.0, reference=rolling | nested LOFO OOF (4 inner folds) |
| 5 | sensors=Oil_temperature, k=1.0, reference=rolling | nested LOFO OOF (5 inner folds) |
| 6 | sensors=Oil_temperature, k=1.0, reference=rolling | nested LOFO OOF (6 inner folds) |

**`ewma`**

| Fold | Parâmetros | Origem do limiar |
|---|---|---|
| 1 | sensors=Motor_current+Oil_temperature, lam=0.1, reference=rolling | fallback (<2 failures in training: a-priori threshold, uncalibrated) |
| 2 | sensors=TP2, lam=0.01, reference=static | nested LOFO OOF (2 inner folds) |
| 3 | sensors=Motor_current, lam=0.01, reference=rolling | nested LOFO OOF (3 inner folds) |
| 4 | sensors=Motor_current, lam=0.01, reference=rolling | nested LOFO OOF (4 inner folds) |
| 5 | sensors=Motor_current, lam=0.01, reference=rolling | nested LOFO OOF (5 inner folds) |
| 6 | sensors=Motor_current, lam=0.01, reference=rolling | nested LOFO OOF (6 inner folds) |

## Gate RN-05

Wilcoxon unilateral exato, pareado por fold (6 blocos → menor p possível 1/64 ≈ 0,016), custo esperado RN-02, α = 0,05.

| Candidato | Baseline | Melhora mediana do custo | p | Promove? |
|---|---|---|---|---|
| `cusum` | `random_forest` | -9.0 | 0.594 | não |
| `ewma` | `random_forest` | +13.5 | 0.094 | não |
| `random_forest` | `cusum` | +9.0 | 0.500 | não |
| `xgboost` | `cusum` | +37.0 | 0.312 | não |
| `logistic_regression` | `cusum` | +7.5 | 0.500 | não |
| `mlp` | `cusum` | +9.0 | 0.500 | não |
| `tcn` | `cusum` | +3.8 | 0.688 | não |
| `bilstm` | `cusum` | +35.7 | 0.281 | não |
| `patchtst` | `cusum` | +26.8 | 0.281 | não |
| `autoencoder` | `cusum` | -22.0 | 0.906 | não |
| `random_forest` | `ewma` | -13.5 | 0.938 | não |
| `xgboost` | `ewma` | -9.0 | 0.781 | não |
| `logistic_regression` | `ewma` | -19.5 | 0.969 | não |
| `mlp` | `ewma` | -9.0 | 0.844 | não |
| `tcn` | `ewma` | -47.8 | 0.969 | não |
| `bilstm` | `ewma` | -2.5 | 0.719 | não |
| `patchtst` | `ewma` | +0.2 | 0.578 | não |
| `autoencoder` | `ewma` | -57.8 | 0.969 | não |
