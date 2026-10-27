# Congelamento dos resultados do TCC

> Decidido em 2026-10-22, ao encerrar a fase de experimentação. Este arquivo
> diz o que é resultado oficial e como registrar o que vier depois.

## O que fica congelado

Os resultados abaixo são os que o TCC e o artigo apresentam. Os hashes SHA-256
de cada artefato (dados de entrada, modelos servidos e JSONs de resultado) e o
commit em que foram gerados estão em
[`reproducibility_manifest.json`](reproducibility_manifest.json), gerado por
`scripts/write_manifest.py`.

| Resultado | Papel | Relatório | Artefato |
|---|---|---|---|
| 8 modelos no protocolo aninhado (6 folds, 3 seeds) | **principal** | [`model_evaluation_leakfree.md`](model_evaluation_leakfree.md) | `apps/ml/models/nested/<modelo>.json` |
| CUSUM / EWMA no mesmo protocolo e gate RN-05 | **principal** | [`statistical_baselines.md`](statistical_baselines.md) | `apps/ml/models/nested/{cusum,ewma}.json` |
| Métricas clássicas de classificação | principal | [`classification_metrics.md`](classification_metrics.md) | `eval_classification_metrics.json` |
| Falhas inéditas em outra APU (MetroPT2) | principal | [`metropt2_unseen_failures.md`](metropt2_unseen_failures.md) | `eval_metropt2.json` |
| Estudo de antecipação (controle de acaso) | principal | [`anticipation_study.md`](anticipation_study.md) | `eval_anticipation.json` |
| Fold 1 separado | **secundário, descritivo** | seção final de [`statistical_baselines.md`](statistical_baselines.md) | — |
| Auditoria dos rótulos (fronteiras, anomalias não confirmadas, FA/dia auditado, cronologia por falha) | **secundário** | [`label_audit.md`](label_audit.md) | `label_audit.json` |

Protocolos que definem esses resultados: walk-forward purgado × LOFO interno
(`apps/ml/src/nested_protocol.py`), métrica por evento
(`apps/ml/src/event_metrics.py`), gate RN-05 (`apps/ml/src/promotion_gate.py`)
e auditoria pré-registrada ([`label_audit_protocol.md`](label_audit_protocol.md)).

## Regras daqui em diante

1. **Nada acima é reescrito.** Os JSONs e relatórios congelados não são
   regenerados com código, dados ou critérios novos.
2. **Descoberta nova vira adendo**, com data, em uma seção "Adendos" no fim
   deste arquivo ou em um relatório próprio, dizendo se foi motivada pelos
   resultados (e portanto não é teste independente).
3. **Os rótulos originais continuam sendo a métrica principal.** Os rótulos
   auditados só aparecem como análise secundária, e nenhum modelo é
   retreinado ou reajustado com eles.
4. **Sem novas arquiteturas nem engenharia de atributos** antes da defesa.
   Com 7 falhas, cada variante a mais é mais uma chance de achar melhora por
   acaso.
5. Correção de **bug** em código que gerou um resultado congelado é permitida,
   mas o resultado antigo fica registrado ao lado do novo, com o motivo.

## Conclusões que esses resultados sustentam

* **Avaliação temporal:** split aleatório por leitura superestima o desempenho
  (PR-AUC 0,999 → 0,59 no walk-forward).
* **Estatística vs ML:** com 6 folds, nenhum método é estatisticamente
  superior (RN-05). O EWMA tem o menor custo observado, também sem o fold 1,
  mas com mais alarmes falsos; a escolha depende da razão de custo.
* **Rótulos e transferência:** a documentação incompleta das falhas limita a
  interpretação dos alarmes falsos (anomalias não confirmadas concentradas no
  fold 1), e a transferência para outra APU funciona só em parte.

O que esses resultados **não** sustentam: que algum método seja o melhor; que
as anomalias não confirmadas sejam falhas; que o sistema antecipe falhas de
forma aprendível.

## Adendos

* **2026-10-27 — matriz de correlação dos atributos**
  ([`correlation_matrix.md`](correlation_matrix.md)). Análise descritiva nova,
  não altera nenhum resultado congelado. Achado: no MetroPT2, `Oil_level` vale 0
  em todas as 23.156 linhas em falha e em só 2.880 linhas normais
  (r = −0,94 com o rótulo). `Oil_level` já estava fora dos modelos (constante no
  MetroPT 2022), então a avaliação no MetroPT2 não usa esse sinal; fica
  registrado como limitação do rótulo/dataset, não como resultado.
