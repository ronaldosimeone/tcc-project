# Datasets usados no TCC

> Números medidos em 2026-10-22 sobre os parquets processados
> (`apps/ml/data/processed/*.parquet`) e os manifestos de ingestão
> (`apps/ml/models/metropt2022_ingest.json`, `metropt2_ingest.json`).

Os três vêm da APU (unidade de produção de ar) de trens do Metro do Porto. Todos
foram reduzidos para uma leitura a cada 10 s (MetroPT-3 já é publicado assim; os
outros dois são 1 Hz e foram amostrados pontualmente, primeira leitura de cada
intervalo de 10 s).

| | MetroPT-3 (2020) | MetroPT 2022 | MetroPT2 |
|---|---|---|---|
| Fonte | UCI (Veloso et al., Sci Data 2022) | Zenodo 6854240 | Zenodo 7766691 |
| Papel no TCC | treino + teste (walk-forward) | treino + teste (walk-forward agrupado) | **só teste**: outra APU, falhas depois de todo o treino |
| Período | 2020-02-01 → 2020-09-01 | 2022-01-01 → 2022-06-02 | 2022-04-28 → 2022-07-28 |
| Duração | 213 dias (~7 meses) | 152 dias (~5 meses) | 91 dias (~3 meses) |
| Linhas originais | 1.516.948 (10 s) | 10.773.588 (1 Hz) | 7.116.940 (1 Hz) |
| Linhas a 10 s | 1.516.948 | 1.077.495 | 705.563 |
| Lacunas > 60 s | 331 (37,9 dias sem dado) | 152 (27,7 dias) | 103 (9,3 dias) |
| Cobertura do período | 82,2 % | 81,8 % | 89,7 % |
| Falhas rotuladas | 4 (todas vazamento de ar) | 3 (vazamento de ar, secador, vazamento de óleo) | 2 (vazamento de ar, vazamento de óleo) |
| Linhas em falha | 29.954 (1,97 %) | 21.216 (1,97 %) | 23.156 (3,28 %) |
| Sensores | 15 (7 analógicos, 8 digitais) | 16 (+ `Flowmeter`; GPS descartado) | 16 (+ `Flowmeter`) |

## Falhas

| Id | Dataset | Tipo | Início | Fim | Duração |
|---|---|---|---|---|---|
| F20_1 | MetroPT-3 | vazamento de ar | 2020-04-18 00:00 | 2020-04-18 23:59 | 24 h |
| F20_2 | MetroPT-3 | vazamento de ar | 2020-05-29 23:30 | 2020-05-30 06:00 | 6,5 h |
| F20_3 | MetroPT-3 | vazamento de ar | 2020-06-05 10:00 | 2020-06-07 14:30 | 52,5 h |
| F20_4 | MetroPT-3 | vazamento de ar | 2020-07-15 14:30 | 2020-07-15 19:00 | 4,5 h |
| F22_1 | 2022 | vazamento de ar (clientes) | 2022-02-28 21:53 | 2022-03-01 02:00 | 4,1 h |
| F22_2 | 2022 | vazamento de ar (secador) | 2022-03-23 14:54 | 2022-03-23 15:24 | 0,5 h |
| F22_3 | 2022 | vazamento de óleo | 2022-05-30 12:00 | 2022-06-02 06:18 | 66,3 h |
| MT2_1 | MetroPT2 | vazamento de ar | 2022-06-04 10:19 | 2022-06-04 14:22 | 4,1 h (LPS 11:26) |
| MT2_2 | MetroPT2 | vazamento de óleo | 2022-07-11 10:10 | 2022-07-14 10:22 | 72,2 h (LPS 07-13 19:43) |

Total: **9 falhas reais**, 7 usadas no walk-forward (6 folds de teste) e 2 de
teste externo.

## Ressalvas

* `Reservoirs` e `Oil_level` mudam de significado entre 2020 e 2022 (mesmo nome,
  sinal diferente) e ficam fora dos modelos.
* `Flowmeter` só existe em 2022/MetroPT2 e separa as falhas sozinho; os
  próprios autores recomendam removê-lo. Usado só em ablação.
* MetroPT2 é outra APU (correlação de `TP3` ≈ 0 no período sobreposto com o
  MetroPT 2022).
* A auditoria de rótulos ([`label_audit.md`](label_audit.md)) achou que 3,4 %
  dos blocos "saudáveis" de 2020 têm a mesma assinatura de carga contínua das
  falhas F20_1/F20_2.
* Matriz de correlação de todos os atributos dos três datasets:
  [`correlation_matrix.md`](correlation_matrix.md). No MetroPT2, `Oil_level`
  vale 0 em todas as linhas em falha (r = −0,94 com o rótulo).
