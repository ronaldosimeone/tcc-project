# Auditoria dos rótulos — resultado

> Gerado por `python -m src.audit_labels` a partir de `apps/ml/models/label_audit.json`. Não edite à mão. Protocolo pré-registrado: [`label_audit_protocol.md`](label_audit_protocol.md).

**Rótulos originais = métrica principal. Rótulos auditados = análise secundária. Nenhum modelo foi retreinado nem ajustado com a auditoria.** Custos aqui são da seed 0 (média por fold); os relatórios principais fazem a média de 3 seeds.

## 1. Critérios pré-registrados (resultado principal da auditoria)

### Etapa A — fronteiras dos eventos

Deslocamento negativo = a evidência física começa **antes** do rótulo.

| Falha | Rótulo | Intervalo físico (E1–E3) | Δ início (h) | Δ fim (h) | % blocos do rótulo com E1 / E2 / E3 |
|---|---|---|---|---|---|
| F20_1 | 2020-04-18 00:00:00 → 2020-04-18 23:59:00 | sem evidência nos sensores | — | — | 0% / 0% / 0% |
| F20_2 | 2020-05-29 23:30:00 → 2020-05-30 06:00:00 | sem evidência nos sensores | — | — | 0% / 0% / 0% |
| F20_3 | 2020-06-05 10:00:00 → 2020-06-07 14:30:00 | 2020-06-05 12:00:00 → 2020-06-07 03:20:00 | +2.0 | -11.2 | 9% / 0% / 2% |
| F20_4 | 2020-07-15 14:30:00 → 2020-07-15 19:00:00 | 2020-07-15 16:50:00 → 2020-07-15 18:50:00 | +2.3 | -0.2 | 43% / 0% / 32% |
| F22_1 | 2022-02-28 21:53:00 → 2022-03-01 02:00:00 | 2022-02-28 22:20:00 → 2022-03-01 02:10:00 | +0.5 | +0.2 | 88% / 85% / 77% |
| F22_2 | 2022-03-23 14:54:00 → 2022-03-23 15:24:00 | sem evidência nos sensores | — | — | 0% / 0% / 0% |
| F22_3 | 2022-05-30 12:00:00 → 2022-06-02 06:18:00 | 2022-05-30 18:50:00 → 2022-06-02 06:30:00 | +6.8 | +0.2 | 3% / 0% / 0% |

### Etapa B — falsos alarmes

Modelos: `random_forest`, `xgboost`, `logistic_regression`, `mlp`, `tcn`, `bilstm`, `patchtst`, `autoencoder`, `cusum`, `ewma` (seed 0, 6 folds fora do tempo). Episódios fora das janelas da métrica, agrupados entre modelos (≤ 1 h).

| Classe | Agrupamentos | Horas de alarme |
|---|---|---|
| Fronteira (perto de falha rotulada) | 7 | 5.1 |
| Candidato a falha não reportada | 15 | 27.2 |
| Evidência fraca | 42 | 105.0 |
| Sem evidência (alarme falso de fato) | 43 | 56.4 |

#### Candidatos a falha não reportada

| Início | Fim | Horas | Modelos | Blocos E1 / E2 / E3 |
|---|---|---|---|---|
| 2020-05-13 13:45:04 | 2020-05-14 01:18:03 | 11.5 | 9 | 65 / 2 / 2 |
| 2020-05-19 01:57:17 | 2020-05-19 02:40:33 | 0.7 | 3 | 6 / 0 / 5 |
| 2020-07-14 02:28:59 | 2020-07-14 02:54:05 | 0.4 | 1 | 3 / 0 / 4 |
| 2022-01-01 21:58:00 | 2022-01-01 22:51:30 | 0.9 | 1 | 2 / 0 / 3 |
| 2022-01-10 07:24:50 | 2022-01-10 08:15:30 | 0.8 | 1 | 1 / 6 / 3 |
| 2022-01-12 09:29:00 | 2022-01-12 09:49:40 | 0.3 | 1 | 2 / 3 / 2 |
| 2022-01-28 07:11:10 | 2022-01-28 07:44:50 | 0.6 | 1 | 2 / 1 / 3 |
| 2022-02-08 20:39:00 | 2022-02-08 21:47:20 | 1.1 | 1 | 4 / 7 / 5 |
| 2022-02-11 19:46:40 | 2022-02-12 01:56:40 | 6.2 | 6 | 37 / 37 / 37 |
| 2022-02-13 09:58:00 | 2022-02-13 10:46:40 | 0.8 | 1 | 2 / 0 / 3 |
| 2022-02-15 12:17:40 | 2022-02-15 13:01:10 | 0.7 | 1 | 2 / 0 / 3 |
| 2022-02-20 19:16:40 | 2022-02-20 20:05:20 | 0.8 | 1 | 2 / 0 / 3 |
| 2022-02-24 20:06:00 | 2022-02-24 21:54:20 | 1.8 | 1 | 3 / 1 / 5 |
| 2022-02-27 08:49:50 | 2022-02-27 09:03:30 | 0.2 | 1 | 2 / 0 / 3 |
| 2022-03-09 16:18:50 | 2022-03-09 16:32:30 | 0.2 | 1 | 2 / 1 / 3 |

### Análise secundária — falsos alarmes e custo com rótulos auditados

Tempo de alarme dentro de agrupamentos *fronteira* e *candidato* tratado como ambíguo (nem acerto nem alarme falso). Detecções inalteradas.

| Modelo | Detectadas | FA original | FA auditado | Custo original | Custo auditado |
|---|---|---|---|---|---|
| `ewma` | 4/6 | 101 | 91 | 50.2 | 48.5 |
| `xgboost` | 3/6 | 36 | 22 | 56.0 | 53.7 |
| `bilstm` | 3/6 | 37 | 23 | 56.2 | 53.8 |
| `patchtst` | 3/6 | 119 | 106 | 69.8 | 67.7 |
| `mlp` | 3/6 | 134 | 98 | 72.3 | 66.3 |
| `random_forest` | 2/6 | 48 | 27 | 74.7 | 71.2 |
| `logistic_regression` | 2/6 | 63 | 34 | 77.2 | 72.3 |
| `cusum` | 2/6 | 74 | 74 | 79.0 | 79.0 |
| `tcn` | 2/6 | 87 | 70 | 81.2 | 78.3 |
| `autoencoder` | 1/6 | 177 | 104 | 112.8 | 100.7 |

## 2. Desvio declarado — E2 com `≥` em vez de `>`

A etapa A mostrou um defeito do critério E2: o resíduo de ciclo de trabalho tem teto (compressor em carga a hora inteira = 1 − média saudável). Em 2020, **3.4% dos blocos rotulados como saudáveis** (2022: 0.2%) já estão nesse teto, então o percentil 99 da referência coincide com o máximo e `duty > p99` nunca dispara, nem dentro de F20_1/F20_2, onde o compressor fica em carga contínua por horas. Esses blocos saudáveis no teto têm a **mesma assinatura das falhas rotuladas de 2020**, o que já é evidência de vazamentos não reportados no período "saudável".

Pelo protocolo, o critério principal não é corrigido. A variante abaixo (`duty ≥ p99`) foi decidida **depois** de ver a etapa A e **antes** de ver a etapa B com ela; é análise de sensibilidade, não substitui a seção 1.

### Etapa A — fronteiras dos eventos

Deslocamento negativo = a evidência física começa **antes** do rótulo.

| Falha | Rótulo | Intervalo físico (E1–E3) | Δ início (h) | Δ fim (h) | % blocos do rótulo com E1 / E2 / E3 |
|---|---|---|---|---|---|
| F20_1 | 2020-04-18 00:00:00 → 2020-04-18 23:59:00 | 2020-04-18 01:30:00 → 2020-04-19 01:50:00 | +1.5 | +1.9 | 0% / 94% / 0% |
| F20_2 | 2020-05-29 23:30:00 → 2020-05-30 06:00:00 | 2020-05-30 00:20:00 → 2020-05-30 05:50:00 | +0.8 | -0.2 | 0% / 82% / 0% |
| F20_3 | 2020-06-05 10:00:00 → 2020-06-07 14:30:00 | 2020-06-05 10:50:00 → 2020-06-07 14:20:00 | +0.8 | -0.2 | 9% / 98% / 2% |
| F20_4 | 2020-07-15 14:30:00 → 2020-07-15 19:00:00 | 2020-07-15 15:30:00 → 2020-07-15 18:50:00 | +1.0 | -0.2 | 43% / 71% / 32% |
| F22_1 | 2022-02-28 21:53:00 → 2022-03-01 02:00:00 | 2022-02-28 22:20:00 → 2022-03-01 02:10:00 | +0.5 | +0.2 | 88% / 85% / 77% |
| F22_2 | 2022-03-23 14:54:00 → 2022-03-23 15:24:00 | sem evidência nos sensores | — | — | 0% / 0% / 0% |
| F22_3 | 2022-05-30 12:00:00 → 2022-06-02 06:18:00 | 2022-05-30 18:50:00 → 2022-06-02 06:30:00 | +6.8 | +0.2 | 3% / 0% / 0% |

### Etapa B — falsos alarmes

Modelos: `random_forest`, `xgboost`, `logistic_regression`, `mlp`, `tcn`, `bilstm`, `patchtst`, `autoencoder`, `cusum`, `ewma` (seed 0, 6 folds fora do tempo). Episódios fora das janelas da métrica, agrupados entre modelos (≤ 1 h).

| Classe | Agrupamentos | Horas de alarme |
|---|---|---|
| Fronteira (perto de falha rotulada) | 7 | 5.1 |
| Candidato a falha não reportada | 16 | 48.6 |
| Evidência fraca | 41 | 83.6 |
| Sem evidência (alarme falso de fato) | 43 | 56.4 |

#### Candidatos a falha não reportada

| Início | Fim | Horas | Modelos | Blocos E1 / E2 / E3 |
|---|---|---|---|---|
| 2020-05-13 13:45:04 | 2020-05-14 01:18:03 | 11.5 | 9 | 65 / 63 / 2 |
| 2020-05-19 01:57:17 | 2020-05-19 02:40:33 | 0.7 | 3 | 6 / 0 / 5 |
| 2020-05-19 22:24:46 | 2020-05-20 19:48:20 | 21.4 | 8 | 54 / 122 / 0 |
| 2020-07-14 02:28:59 | 2020-07-14 02:54:05 | 0.4 | 1 | 3 / 0 / 4 |
| 2022-01-01 21:58:00 | 2022-01-01 22:51:30 | 0.9 | 1 | 2 / 0 / 3 |
| 2022-01-10 07:24:50 | 2022-01-10 08:15:30 | 0.8 | 1 | 1 / 6 / 3 |
| 2022-01-12 09:29:00 | 2022-01-12 09:49:40 | 0.3 | 1 | 2 / 3 / 2 |
| 2022-01-28 07:11:10 | 2022-01-28 07:44:50 | 0.6 | 1 | 2 / 1 / 3 |
| 2022-02-08 20:39:00 | 2022-02-08 21:47:20 | 1.1 | 1 | 4 / 7 / 5 |
| 2022-02-11 19:46:40 | 2022-02-12 01:56:40 | 6.2 | 6 | 37 / 37 / 37 |
| 2022-02-13 09:58:00 | 2022-02-13 10:46:40 | 0.8 | 1 | 2 / 0 / 3 |
| 2022-02-15 12:17:40 | 2022-02-15 13:01:10 | 0.7 | 1 | 2 / 0 / 3 |
| 2022-02-20 19:16:40 | 2022-02-20 20:05:20 | 0.8 | 1 | 2 / 0 / 3 |
| 2022-02-24 20:06:00 | 2022-02-24 21:54:20 | 1.8 | 1 | 3 / 1 / 5 |
| 2022-02-27 08:49:50 | 2022-02-27 09:03:30 | 0.2 | 1 | 2 / 0 / 3 |
| 2022-03-09 16:18:50 | 2022-03-09 16:32:30 | 0.2 | 1 | 2 / 1 / 3 |

### Análise secundária — falsos alarmes e custo com rótulos auditados

Tempo de alarme dentro de agrupamentos *fronteira* e *candidato* tratado como ambíguo (nem acerto nem alarme falso). Detecções inalteradas.

| Modelo | Detectadas | FA original | FA auditado | Custo original | Custo auditado |
|---|---|---|---|---|---|
| `ewma` | 4/6 | 101 | 91 | 50.2 | 48.5 |
| `xgboost` | 3/6 | 36 | 0 | 56.0 | 50.0 |
| `bilstm` | 3/6 | 37 | 1 | 56.2 | 50.2 |
| `patchtst` | 3/6 | 119 | 84 | 69.8 | 64.0 |
| `mlp` | 3/6 | 134 | 76 | 72.3 | 62.7 |
| `random_forest` | 2/6 | 48 | 3 | 74.7 | 67.2 |
| `logistic_regression` | 2/6 | 63 | 12 | 77.2 | 68.7 |
| `cusum` | 2/6 | 74 | 74 | 79.0 | 79.0 |
| `tcn` | 2/6 | 87 | 48 | 81.2 | 74.7 |
| `autoencoder` | 1/6 | 177 | 101 | 112.8 | 100.2 |
