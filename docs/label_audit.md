# Auditoria dos rótulos — resultado

> Gerado por `python -m src.audit_labels` a partir de `apps/ml/models/label_audit.json`. Não edite à mão. Protocolo pré-registrado: [`label_audit_protocol.md`](label_audit_protocol.md).

**Rótulos originais = métrica principal. Rótulos auditados = análise secundária. Nenhum modelo foi retreinado nem ajustado com a auditoria.** Custos aqui são da seed 0 (média por fold); os relatórios principais fazem a média de 3 seeds.

## 1. Critérios pré-registrados (resultado principal da auditoria)

### Etapa A — fronteiras dos eventos

Evidência observável nos sensores, não o início físico real (que não é conhecido). Δ = primeira evidência − início documentado; negativo = a evidência aparece **antes** do registro.

| Falha | Intervalo documentado | Primeira → última evidência (E1–E3) | Δ início (h) | Δ fim (h) | % blocos do rótulo com E1 / E2 / E3 |
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
| Anomalia não confirmada | 15 | 27.2 |
| Evidência fraca | 42 | 105.0 |
| Sem evidência (alarme falso de fato) | 43 | 56.4 |

#### Anomalias não confirmadas

Comportamento anormal nos sensores, sem registro de manutenção e sem diagnóstico. **Não são falhas presumidas**: os detectores e as evidências E1–E3 usam os mesmos sensores (TP3, COMP, LPS), então a concordância entre eles mostra uma mudança forte no sinal, não sua causa. Mudança operacional ou de instrumentação também explicaria.

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

### Intervalos excluídos na análise secundária

Todo o tempo destes agrupamentos fora das janelas de evento da métrica (início − 6 h, fim + 1 h) sai do numerador **e** do denominador dos alarmes falsos. Nenhum outro critério ou limiar muda.

| Início | Fim | Horas | Classe |
|---|---|---|---|
| 2020-05-13 13:45:04 | 2020-05-14 01:18:03 | 11.5 | Anomalia não confirmada |
| 2020-05-19 01:57:17 | 2020-05-19 02:40:33 | 0.7 | Anomalia não confirmada |
| 2020-05-31 04:31:49 | 2020-05-31 04:59:54 | 0.5 | Fronteira (perto de falha rotulada) |
| 2020-06-08 12:46:52 | 2020-06-08 13:07:12 | 0.3 | Fronteira (perto de falha rotulada) |
| 2020-07-14 02:28:59 | 2020-07-14 02:54:05 | 0.4 | Anomalia não confirmada |
| 2022-01-01 21:58:00 | 2022-01-01 22:51:30 | 0.9 | Anomalia não confirmada |
| 2022-01-10 07:24:50 | 2022-01-10 08:15:30 | 0.8 | Anomalia não confirmada |
| 2022-01-12 09:29:00 | 2022-01-12 09:49:40 | 0.3 | Anomalia não confirmada |
| 2022-01-28 07:11:10 | 2022-01-28 07:44:50 | 0.6 | Anomalia não confirmada |
| 2022-02-08 20:39:00 | 2022-02-08 21:47:20 | 1.1 | Anomalia não confirmada |
| 2022-02-11 19:46:40 | 2022-02-12 01:56:40 | 6.2 | Anomalia não confirmada |
| 2022-02-13 09:58:00 | 2022-02-13 10:46:40 | 0.8 | Anomalia não confirmada |
| 2022-02-15 12:17:40 | 2022-02-15 13:01:10 | 0.7 | Anomalia não confirmada |
| 2022-02-20 19:16:40 | 2022-02-20 20:05:20 | 0.8 | Anomalia não confirmada |
| 2022-02-24 20:06:00 | 2022-02-24 21:54:20 | 1.8 | Anomalia não confirmada |
| 2022-02-27 08:49:50 | 2022-02-27 09:03:30 | 0.2 | Anomalia não confirmada |
| 2022-02-27 21:07:10 | 2022-02-27 22:00:40 | 0.9 | Fronteira (perto de falha rotulada) |
| 2022-02-28 06:01:20 | 2022-02-28 06:01:20 | 0.0 | Fronteira (perto de falha rotulada) |
| 2022-03-01 06:00:00 | 2022-03-01 08:16:30 | 2.3 | Fronteira (perto de falha rotulada) |
| 2022-03-01 23:43:30 | 2022-03-02 00:22:40 | 0.7 | Fronteira (perto de falha rotulada) |
| 2022-03-09 16:18:50 | 2022-03-09 16:32:30 | 0.2 | Anomalia não confirmada |
| 2022-03-24 13:13:20 | 2022-03-24 13:41:00 | 0.5 | Fronteira (perto de falha rotulada) |

### Análise secundária — falsos alarmes e custo com rótulos auditados

Tempo de alarme dentro dos intervalos excluídos tratado como ambíguo (nem acerto nem alarme falso). Detecções inalteradas. FA/dia = alarmes falsos somados ÷ dias de exposição somados dos 6 folds de teste; no auditado, a exposição desconta o tempo excluído. É a taxa agregada, por isso difere da média das taxas por fold dos outros relatórios.

| Modelo | Detectadas | FA original | FA auditado | Exposição original (dias) | Exposição auditada (dias) | FA/dia original | FA/dia auditado | Custo original | Custo auditado |
|---|---|---|---|---|---|---|---|---|---|
| `ewma` | 4/6 | 101 | 91 | 267.1 | 265.7 | 0.38 | 0.34 | 50.2 | 48.5 |
| `xgboost` | 3/6 | 36 | 22 | 267.1 | 265.7 | 0.13 | 0.08 | 56.0 | 53.7 |
| `bilstm` | 3/6 | 37 | 23 | 267.1 | 265.7 | 0.14 | 0.09 | 56.2 | 53.8 |
| `patchtst` | 3/6 | 119 | 106 | 267.1 | 265.7 | 0.45 | 0.40 | 69.8 | 67.7 |
| `mlp` | 3/6 | 134 | 98 | 267.1 | 265.7 | 0.50 | 0.37 | 72.3 | 66.3 |
| `random_forest` | 2/6 | 48 | 27 | 267.1 | 265.7 | 0.18 | 0.10 | 74.7 | 71.2 |
| `logistic_regression` | 2/6 | 63 | 34 | 267.1 | 265.7 | 0.24 | 0.13 | 77.2 | 72.3 |
| `cusum` | 2/6 | 74 | 74 | 267.1 | 265.7 | 0.28 | 0.28 | 79.0 | 79.0 |
| `tcn` | 2/6 | 87 | 70 | 267.1 | 265.7 | 0.33 | 0.26 | 81.2 | 78.3 |
| `autoencoder` | 1/6 | 177 | 104 | 267.1 | 265.7 | 0.66 | 0.39 | 112.8 | 100.7 |

## 2. Desvio declarado — E2 com `≥` em vez de `>`

A etapa A mostrou um defeito do critério E2: o resíduo de ciclo de trabalho tem teto (compressor em carga a hora inteira = 1 − média saudável). Em 2020, **3.4% dos blocos rotulados como saudáveis** (2022: 0.2%) já estão nesse teto, então o percentil 99 da referência coincide com o máximo e `duty > p99` nunca dispara, nem dentro de F20_1/F20_2, onde o compressor fica em carga contínua por horas. Esses blocos saudáveis no teto têm a **mesma assinatura das falhas rotuladas de 2020**. Isso é compatível com vazamentos não reportados no período "saudável", mas não os confirma.

Pelo protocolo, o critério principal não é corrigido. A variante abaixo (`duty ≥ p99`) foi decidida **depois** de ver a etapa A e **antes** de ver a etapa B com ela; é análise de sensibilidade, não substitui a seção 1.

### Etapa A — fronteiras dos eventos

Evidência observável nos sensores, não o início físico real (que não é conhecido). Δ = primeira evidência − início documentado; negativo = a evidência aparece **antes** do registro.

| Falha | Intervalo documentado | Primeira → última evidência (E1–E3) | Δ início (h) | Δ fim (h) | % blocos do rótulo com E1 / E2 / E3 |
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
| Anomalia não confirmada | 16 | 48.6 |
| Evidência fraca | 41 | 83.6 |
| Sem evidência (alarme falso de fato) | 43 | 56.4 |

#### Anomalias não confirmadas

Comportamento anormal nos sensores, sem registro de manutenção e sem diagnóstico. **Não são falhas presumidas**: os detectores e as evidências E1–E3 usam os mesmos sensores (TP3, COMP, LPS), então a concordância entre eles mostra uma mudança forte no sinal, não sua causa. Mudança operacional ou de instrumentação também explicaria.

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

### Intervalos excluídos na análise secundária

Todo o tempo destes agrupamentos fora das janelas de evento da métrica (início − 6 h, fim + 1 h) sai do numerador **e** do denominador dos alarmes falsos. Nenhum outro critério ou limiar muda.

| Início | Fim | Horas | Classe |
|---|---|---|---|
| 2020-05-13 13:45:04 | 2020-05-14 01:18:03 | 11.5 | Anomalia não confirmada |
| 2020-05-19 01:57:17 | 2020-05-19 02:40:33 | 0.7 | Anomalia não confirmada |
| 2020-05-19 22:24:46 | 2020-05-20 19:48:20 | 21.4 | Anomalia não confirmada |
| 2020-05-31 04:31:49 | 2020-05-31 04:59:54 | 0.5 | Fronteira (perto de falha rotulada) |
| 2020-06-08 12:46:52 | 2020-06-08 13:07:12 | 0.3 | Fronteira (perto de falha rotulada) |
| 2020-07-14 02:28:59 | 2020-07-14 02:54:05 | 0.4 | Anomalia não confirmada |
| 2022-01-01 21:58:00 | 2022-01-01 22:51:30 | 0.9 | Anomalia não confirmada |
| 2022-01-10 07:24:50 | 2022-01-10 08:15:30 | 0.8 | Anomalia não confirmada |
| 2022-01-12 09:29:00 | 2022-01-12 09:49:40 | 0.3 | Anomalia não confirmada |
| 2022-01-28 07:11:10 | 2022-01-28 07:44:50 | 0.6 | Anomalia não confirmada |
| 2022-02-08 20:39:00 | 2022-02-08 21:47:20 | 1.1 | Anomalia não confirmada |
| 2022-02-11 19:46:40 | 2022-02-12 01:56:40 | 6.2 | Anomalia não confirmada |
| 2022-02-13 09:58:00 | 2022-02-13 10:46:40 | 0.8 | Anomalia não confirmada |
| 2022-02-15 12:17:40 | 2022-02-15 13:01:10 | 0.7 | Anomalia não confirmada |
| 2022-02-20 19:16:40 | 2022-02-20 20:05:20 | 0.8 | Anomalia não confirmada |
| 2022-02-24 20:06:00 | 2022-02-24 21:54:20 | 1.8 | Anomalia não confirmada |
| 2022-02-27 08:49:50 | 2022-02-27 09:03:30 | 0.2 | Anomalia não confirmada |
| 2022-02-27 21:07:10 | 2022-02-27 22:00:40 | 0.9 | Fronteira (perto de falha rotulada) |
| 2022-02-28 06:01:20 | 2022-02-28 06:01:20 | 0.0 | Fronteira (perto de falha rotulada) |
| 2022-03-01 06:00:00 | 2022-03-01 08:16:30 | 2.3 | Fronteira (perto de falha rotulada) |
| 2022-03-01 23:43:30 | 2022-03-02 00:22:40 | 0.7 | Fronteira (perto de falha rotulada) |
| 2022-03-09 16:18:50 | 2022-03-09 16:32:30 | 0.2 | Anomalia não confirmada |
| 2022-03-24 13:13:20 | 2022-03-24 13:41:00 | 0.5 | Fronteira (perto de falha rotulada) |

### Análise secundária — falsos alarmes e custo com rótulos auditados

Tempo de alarme dentro dos intervalos excluídos tratado como ambíguo (nem acerto nem alarme falso). Detecções inalteradas. FA/dia = alarmes falsos somados ÷ dias de exposição somados dos 6 folds de teste; no auditado, a exposição desconta o tempo excluído. É a taxa agregada, por isso difere da média das taxas por fold dos outros relatórios.

| Modelo | Detectadas | FA original | FA auditado | Exposição original (dias) | Exposição auditada (dias) | FA/dia original | FA/dia auditado | Custo original | Custo auditado |
|---|---|---|---|---|---|---|---|---|---|
| `ewma` | 4/6 | 101 | 91 | 267.1 | 264.8 | 0.38 | 0.34 | 50.2 | 48.5 |
| `xgboost` | 3/6 | 36 | 0 | 267.1 | 264.8 | 0.13 | 0.00 | 56.0 | 50.0 |
| `bilstm` | 3/6 | 37 | 1 | 267.1 | 264.8 | 0.14 | 0.00 | 56.2 | 50.2 |
| `patchtst` | 3/6 | 119 | 84 | 267.1 | 264.8 | 0.45 | 0.32 | 69.8 | 64.0 |
| `mlp` | 3/6 | 134 | 76 | 267.1 | 264.8 | 0.50 | 0.29 | 72.3 | 62.7 |
| `random_forest` | 2/6 | 48 | 3 | 267.1 | 264.8 | 0.18 | 0.01 | 74.7 | 67.2 |
| `logistic_regression` | 2/6 | 63 | 12 | 267.1 | 264.8 | 0.24 | 0.05 | 77.2 | 68.7 |
| `cusum` | 2/6 | 74 | 74 | 267.1 | 264.8 | 0.28 | 0.28 | 79.0 | 79.0 |
| `tcn` | 2/6 | 87 | 48 | 267.1 | 264.8 | 0.33 | 0.18 | 81.2 | 74.7 |
| `autoencoder` | 1/6 | 177 | 101 | 267.1 | 264.8 | 0.66 | 0.38 | 112.8 | 100.2 |

## 3. Cronologia por falha

Três instantes separados, sem presumir que algum seja o início físico real: o **início documentado** (relatório de manutenção), a **primeira evidência** observável nos sensores (E1–E3) e o **primeiro alarme** de cada detector (seed 0, no fold em que a falha é teste; o alarme só conta a partir de 6 h antes do início documentado). Horas relativas ao início documentado; negativo = antes.

| Falha | Início documentado | 1ª evidência (critério principal) | 1ª evidência (variante `≥`) | Detectores que alarmaram | 1º alarme mais cedo | Mediana do 1º alarme |
|---|---|---|---|---|---|---|
| F20_1 | 2020-04-18 00:00:00 | — | +1.5 | nunca é teste (só treino) | — | — |
| F20_2 | 2020-05-29 23:30:00 | — | +0.8 | 8/10 | -0.2 (`mlp`) | -0.2 |
| F20_3 | 2020-06-05 10:00:00 | +2.0 | +0.8 | 6/10 | -0.1 (`mlp`) | +0.0 |
| F20_4 | 2020-07-15 14:30:00 | +2.3 | +1.0 | 2/10 | +0.1 (`ewma`) | +0.9 |
| F22_1 | 2022-02-28 21:53:00 | +0.5 | +0.5 | 9/10 | +0.4 (`random_forest`) | +0.5 |
| F22_2 | 2022-03-23 14:54:00 | — | — | 0/10 | — | — |
| F22_3 | 2022-05-30 12:00:00 | +6.8 | +6.8 | 0/10 | — | — |

Primeiro alarme por detector (h em relação ao início documentado; — = não detectou):

| Falha | `random_forest` | `xgboost` | `logistic_regression` | `mlp` | `tcn` | `bilstm` | `patchtst` | `autoencoder` | `cusum` | `ewma` |
|---|---|---|---|---|---|---|---|---|---|---|
| F20_2 | -0.2 | -0.2 | -0.2 | -0.2 | -0.1 | -0.1 | -0.2 | — | — | +1.2 |
| F20_3 | — | +0.0 | — | -0.1 | — | -0.0 | -0.1 | — | +12.8 | +0.5 |
| F20_4 | — | — | — | — | — | — | — | — | +1.6 | +0.1 |
| F22_1 | +0.4 | +0.5 | +0.4 | +0.4 | +0.6 | +0.6 | +0.4 | +0.7 | — | +0.9 |
| F22_2 | — | — | — | — | — | — | — | — | — | — |
| F22_3 | — | — | — | — | — | — | — | — | — | — |
