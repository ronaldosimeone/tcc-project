# Avaliação cross-dataset — RF-33

> Gerado por `python -m src.render_cross_dataset_report` a partir de `apps/ml/models/eval_cross_dataset.json`. Não edite à mão.

**Dados.** MetroPT-3 (fev–set/2020, 4 falhas) + MetroPT 2022 (Veloso et al., *Scientific Data* 9:764, 2022; Zenodo 10.5281/zenodo.6854240, CC BY 4.0). O 2022 foi harmonizado: as 4 colunas de GPS foram descartadas, o `Flowmeter` ficou só para a ablação, e a amostragem pontual a 10 s levou 10,773,588 linhas a 1 Hz a 1,077,495 linhas. Falhas de 2022 (Tabela 2 do artigo): air_leak_clients (2022-02-28 21:53 → 2022-03-01 02:00, 1483 linhas), air_leak_air_dryer (2022-03-23 14:54 → 2022-03-23 15:24, 181 linhas), oil_leak_compressor (2022-05-30 12:00 → 2022-06-02 06:18, 19552 linhas).

## Linha de base — 12 sensores casados pelo nome

### Q1 — Quão diferentes são os dois períodos?

**Validação adversarial**: um RF treinado para distinguir linhas *normais* de 2020 e de 2022 atinge ROC-AUC **1.000** em folds bloqueados no tempo (0,5 = indistinguíveis). Sensores que mais separam os períodos: `Reservoirs` (0.40), `Oil_level` (0.29), `DV_pressure` (0.14), `TP2` (0.08), `Motor_current` (0.06).

| Sensor | média ± dp 2020 | média ± dp 2022 | PSI | KS |
|---|---|---|---|---|
| `TP2` | 1.23 ± 3.13 | 1.15 ± 3.07 | 1.98 | 0.59 |
| `TP3` | 9.00 ± 0.63 | 8.98 ± 0.67 | 0.01 | 0.03 |
| `H1` | 7.72 ± 3.19 | 7.76 ± 3.05 | 0.01 | 0.04 |
| `DV_pressure` | 0.02 ± 0.28 | -0.02 ± 0.15 | 4.37 | 0.82 |
| `Reservoirs` | 9.00 ± 0.63 | 1.57 ± 0.09 | 8.30 | 1.00 |
| `Motor_current` | 1.98 ± 2.27 | 2.39 ± 2.19 | 5.43 | 0.44 |
| `Oil_temperature` | 62.38 ± 6.30 | 67.32 ± 5.33 | 0.76 | 0.33 |
| `COMP` | 0.85 ± 0.35 | 0.87 ± 0.34 | — | 0.02 |
| `DV_eletric` | 0.14 ± 0.35 | 0.13 ± 0.34 | — | 0.01 |
| `Towers` | 0.93 ± 0.26 | 0.94 ± 0.25 | — | 0.01 |
| `MPG` | 0.85 ± 0.36 | 0.87 ± 0.34 | — | 0.02 |
| `Oil_level` | 0.90 ± 0.30 | 0.00 ± 0.00 | — | 0.90 |

### Q2 — Um modelo de um período funciona no outro?

Treino no período inteiro de origem, teste no período inteiro de destino; limiar escolhido **só** no treino (OOF interno, RN-02). Média [IC 95 %] sobre 3 seeds.

| Direção | Modelo | PR-AUC | Recall de evento | FA/dia | Custo esperado |
|---|---|---|---|---|---|
| 2020 → 2022 | RF | 0.058 [0.057; 0.061] | 0.33 [0.33; 0.33] | 0.03 [0.02; 0.04] | 204.3 [203.0; 206.0] |
| 2020 → 2022 | XGBoost | 0.059 [0.057; 0.063] | 0.33 [0.33; 0.33] | 0.19 [0.00; 0.55] | 229.7 [200.0; 284.0] |
| 2020 → 2022 | LogReg | 0.019 [0.019; 0.019] | 1.00 [1.00; 1.00] | 1.10 [1.10; 1.10] | 167.0 [167.0; 167.0] |
| 2022 → 2020 | RF | 0.328 [0.316; 0.334] | 0.00 [0.00; 0.00] | 0.00 [0.00; 0.00] | 400.0 [400.0; 400.0] |
| 2022 → 2020 | XGBoost | 0.502 [0.462; 0.523] | 0.75 [0.50; 1.00] | 0.63 [0.26; 0.87] | 233.7 [156.0; 359.0] |
| 2022 → 2020 | LogReg | 0.019 [0.019; 0.019] | 0.75 [0.75; 0.75] | 48.54 [48.54; 48.54] | 10447.0 [10447.0; 10447.0] |

### Q3 — Vale a pena somar os dois?

Walk-forward purgado na linha do tempo conjunta: 7 falhas → **6 folds** de teste. Cada fold treina só com o passado.

| Modelo | PR-AUC | Recall de evento | FA/dia | Custo esperado |
|---|---|---|---|---|
| RF | 0.398 [0.251; 0.554] | 0.50 [0.28; 0.72] | 0.15 [0.04; 0.28] | 59.1 [38.4; 78.6] |
| XGBoost | 0.307 [0.194; 0.434] | 0.50 [0.28; 0.72] | 0.11 [0.02; 0.25] | 55.8 [33.7; 77.4] |
| LogReg | 0.180 [0.044; 0.332] | 0.50 [0.28; 0.72] | 0.29 [0.12; 0.50] | 66.5 [45.6; 85.7] |

Custo esperado por período de teste:

| Modelo | folds de 2020 | folds de 2022 |
|---|---|---|
| RF | 50.1 [23.8; 77.3] | 68.0 [36.0; 100.0] |
| XGBoost | 35.0 [11.8; 67.3] | 76.6 [45.6; 100.0] |
| LogReg | 38.0 [14.7; 69.7] | 95.0 [89.1; 100.4] |

**Conjunto × só 2022** — mesmos folds de 2022 que já têm uma falha de 2022 no passado, treinados (a) com 2020 + 2022 e (b) só com 2022:

| Modelo | folds | Custo conjunto | Custo só 2022 | PR-AUC conjunto | PR-AUC só 2022 | Wilcoxon p (custo) |
|---|---|---|---|---|---|---|
| RF | [5, 6] | 100.0 | 100.0 | 0.055 | 0.053 | 1.000 |
| XGBoost | [5, 6] | 100.0 | 100.0 | 0.052 | 0.052 | 1.000 |
| LogReg | [5, 6] | 101.0 | 53.5 | 0.043 | 0.065 | 0.031 |

**Gate RN-05** (Wilcoxon unilateral pareado por fold, custo esperado):

| Candidato × base | folds | melhora mediana | p | p mínimo atingível | promove? |
|---|---|---|---|---|---|
| XGBoost × RF | 6 | -0.8 | 0.688 | 0.016 | não |
| RF × LogReg | 6 | 4.0 | 0.312 | 0.016 | não |
| XGBoost × LogReg | 6 | 2.3 | 0.031 | 0.016 | sim |

Friedman (custo esperado, 6 folds): p = 0.076.

### Q4 — Quanto custa descartar o `Flowmeter`?

Só 2022 (o MetroPT-3 não tem medidor de vazão), blocked (3 folds, 2022 only). Diferença pareada por fold×seed, com − sem (IC 95 %):

Leitura: só com o 2022 (no máximo 2 falhas de treino por fold) nenhum modelo detecta a falha de teste, com ou sem o sensor (recall 0). A ablação fica **inconclusiva por efeito piso** e não prova que o `Flowmeter` seja inútil.

| Modelo | Métrica | com | sem | Δ (com − sem) |
|---|---|---|---|---|
| RF | pr_auc | 0.038 | 0.037 | 0.001 [-0.007; 0.007] |
| RF | event_recall_inner | 0.000 | 0.000 | 0.000 [0.000; 0.000] |
| RF | expected_cost_inner | 100.000 | 100.000 | 0.000 [0.000; 0.000] |
| XGBoost | pr_auc | 0.030 | 0.036 | -0.007 [-0.015; 0.000] |
| XGBoost | event_recall_inner | 0.000 | 0.000 | 0.000 [0.000; 0.000] |
| XGBoost | expected_cost_inner | 315.556 | 335.000 | -19.444 [-40.222; -1.444] |

## Harmonização semântica — mesmo nome, outro sinal

O bloco Q1 mostrou que casar as colunas pelo nome não basta:

* **`Reservoirs`** — no MetroPT-3 é uma cópia do `TP3` (correlação 1,0000, diferença máxima 0,18 bar); no 2022 é outra grandeza (≈ 1,6 bar, correlação −0,001 com o `TP3`).
* **`Oil_level`** — no MetroPT-3 vale 1 em 90 % das linhas normais e em 100 % das de falha; no 2022 é constante 0.

Os dois respondem pela maior parte da separação adversarial. A **versão harmonizada** abaixo repete Q1–Q3 sem esses dois sensores (`--drop-sensors Reservoirs Oil_level`). A linha de base com as 12 colunas fica acima, e a diferença entre as duas versões mede o custo de uma harmonização ingênua.

## Versão harmonizada — sem `Reservoirs`, `Oil_level`

### Q1 — Quão diferentes são os dois períodos?

**Validação adversarial**: um RF treinado para distinguir linhas *normais* de 2020 e de 2022 atinge ROC-AUC **0.998** em folds bloqueados no tempo (0,5 = indistinguíveis). Sensores que mais separam os períodos: `DV_pressure` (0.50), `TP2` (0.19), `Motor_current` (0.17), `Oil_temperature` (0.09), `H1` (0.02).

| Sensor | média ± dp 2020 | média ± dp 2022 | PSI | KS |
|---|---|---|---|---|
| `TP2` | 1.23 ± 3.13 | 1.15 ± 3.07 | 1.98 | 0.59 |
| `TP3` | 9.00 ± 0.63 | 8.98 ± 0.67 | 0.01 | 0.03 |
| `H1` | 7.72 ± 3.19 | 7.76 ± 3.05 | 0.01 | 0.04 |
| `DV_pressure` | 0.02 ± 0.28 | -0.02 ± 0.15 | 4.37 | 0.82 |
| `Motor_current` | 1.98 ± 2.27 | 2.39 ± 2.19 | 5.43 | 0.44 |
| `Oil_temperature` | 62.38 ± 6.30 | 67.32 ± 5.33 | 0.76 | 0.33 |
| `COMP` | 0.85 ± 0.35 | 0.87 ± 0.34 | — | 0.02 |
| `DV_eletric` | 0.14 ± 0.35 | 0.13 ± 0.34 | — | 0.01 |
| `Towers` | 0.93 ± 0.26 | 0.94 ± 0.25 | — | 0.01 |
| `MPG` | 0.85 ± 0.36 | 0.87 ± 0.34 | — | 0.02 |

### Q2 — Um modelo de um período funciona no outro?

Treino no período inteiro de origem, teste no período inteiro de destino; limiar escolhido **só** no treino (OOF interno, RN-02). Média [IC 95 %] sobre 3 seeds.

| Direção | Modelo | PR-AUC | Recall de evento | FA/dia | Custo esperado |
|---|---|---|---|---|---|
| 2020 → 2022 | RF | 0.049 [0.041; 0.055] | 0.33 [0.33; 0.33] | 0.03 [0.02; 0.04] | 204.7 [203.0; 206.0] |
| 2020 → 2022 | XGBoost | 0.058 [0.055; 0.061] | 0.33 [0.33; 0.33] | 0.01 [0.00; 0.03] | 202.0 [200.0; 205.0] |
| 2020 → 2022 | LogReg | 0.063 [0.063; 0.063] | 0.33 [0.33; 0.33] | 0.07 [0.07; 0.07] | 211.0 [211.0; 211.0] |
| 2022 → 2020 | RF | 0.163 [0.140; 0.176] | 0.00 [0.00; 0.00] | 0.00 [0.00; 0.00] | 400.0 [400.0; 400.0] |
| 2022 → 2020 | XGBoost | 0.412 [0.376; 0.453] | 0.25 [0.25; 0.25] | 0.00 [0.00; 0.00] | 300.0 [300.0; 300.0] |
| 2022 → 2020 | LogReg | 0.019 [0.019; 0.019] | 0.00 [0.00; 0.00] | 0.73 [0.73; 0.73] | 556.0 [556.0; 556.0] |

### Q3 — Vale a pena somar os dois?

Walk-forward purgado na linha do tempo conjunta: 7 falhas → **6 folds** de teste. Cada fold treina só com o passado.

| Modelo | PR-AUC | Recall de evento | FA/dia | Custo esperado |
|---|---|---|---|---|
| RF | 0.412 [0.267; 0.568] | 0.50 [0.28; 0.72] | 0.16 [0.05; 0.29] | 59.3 [38.9; 78.8] |
| XGBoost | 0.311 [0.195; 0.439] | 0.50 [0.28; 0.72] | 0.04 [0.01; 0.09] | 51.2 [29.2; 73.1] |
| LogReg | 0.287 [0.135; 0.457] | 0.50 [0.28; 0.72] | 0.13 [0.09; 0.19] | 54.8 [33.0; 76.4] |

Custo esperado por período de teste:

| Modelo | folds de 2020 | folds de 2022 |
|---|---|---|
| RF | 50.6 [24.4; 77.6] | 68.1 [36.2; 100.0] |
| XGBoost | 35.1 [11.6; 67.4] | 67.3 [34.7; 100.0] |
| LogReg | 38.0 [14.7; 69.7] | 71.7 [40.0; 102.7] |

**Conjunto × só 2022** — mesmos folds de 2022 que já têm uma falha de 2022 no passado, treinados (a) com 2020 + 2022 e (b) só com 2022:

| Modelo | folds | Custo conjunto | Custo só 2022 | PR-AUC conjunto | PR-AUC só 2022 | Wilcoxon p (custo) |
|---|---|---|---|---|---|---|
| RF | [5, 6] | 100.0 | 100.0 | 0.102 | 0.096 | 1.000 |
| XGBoost | [5, 6] | 100.0 | 100.0 | 0.048 | 0.046 | 1.000 |
| LogReg | [5, 6] | 103.5 | 61.0 | 0.038 | 0.060 | 0.344 |

**Gate RN-05** (Wilcoxon unilateral pareado por fold, custo esperado):

| Candidato × base | folds | melhora mediana | p | p mínimo atingível | promove? |
|---|---|---|---|---|---|
| XGBoost × RF | 6 | 0.0 | 0.438 | 0.016 | não |
| RF × LogReg | 6 | 4.3 | 0.219 | 0.016 | não |
| XGBoost × LogReg | 6 | 3.3 | 0.016 | 0.016 | sim |

Friedman (custo esperado, 6 folds): p = 0.032.

