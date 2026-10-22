# Trabalhos relacionados no MetroPT: o que outros fizeram e quem está certo

> Levantamento feito em 2026-10-21 (artigos e repositórios públicos). Os números
> dos outros trabalhos são os que eles próprios reportam. Os do PredictIQ vêm de
> [`classification_metrics.md`](classification_metrics.md),
> [`model_evaluation_leakfree.md`](model_evaluation_leakfree.md) e
> [`metropt2_unseen_failures.md`](metropt2_unseen_failures.md).

## 1. Resumo do veredito

Os trabalhos se dividem em dois grupos, e a diferença entre eles é o protocolo de
avaliação, não o algoritmo:

| Grupo | Como avalia | Resultado típico | Leitura |
|---|---|---|---|
| **A — split aleatório por leitura** | 80/20 (ou não declarado) sobre leituras de 10 s, métricas por leitura | acurácia / F1 de 0,95 a **1,00** | **Inflado por vazamento**: leituras vizinhas (quase idênticas) caem no treino e no teste, e o "teste" pergunta se a leitura está *dentro* de uma falha já vista. O PredictIQ reproduziu esse efeito: PR-AUC 0,999 com split aleatório contra 0,59 no walk-forward. |
| **B — temporal / por evento** | corte cronológico ou falha inédita no teste, métricas por evento (falhas detectadas, falsos alarmes por semana ou dia, antecedência) | detecção parcial, poucas horas de antecedência ou nenhuma, intervalos de confiança largos | **Metodologicamente correto**, e as conclusões batem com as do PredictIQ. |

**Quem está certo:** o grupo B, e o PredictIQ está nele. Ninguém do grupo B
encontra um modelo que antecipe as falhas de forma confiável com tão poucos
eventos, e todos alertam que 4 a 7 falhas não sustentam conclusões fortes. O
PredictIQ chegou à mesma conclusão por conta própria (estudo de antecipação com
controle de acaso e gate RN-05 sem promoção).

Mas o grupo B também mostra **duas coisas que o PredictIQ ainda não fez**
(seção 4): comparar com um *baseline* estatístico simples (CUSUM/EWMA), que lá
detecta 4/4 com cerca de 0,1 falso alarme por dia; e limpar do treino os episódios
de vazamento não reportados.

## 2. Trabalhos analisados

### Grupo A — split aleatório (resultados inflados)

| Trabalho | Protocolo | Balanceamento | Resultado reportado | Problema |
|---|---|---|---|---|
| IJACSA 2025, *Predictive Modeling for Metro Performance Using MetroPT3* (LSTM, autoencoder, híbrido) | StandardScaler + **split 80/20**; métricas por leitura | — | acurácia / precisão / recall / F1 por leitura; a revisão do próprio artigo cita trabalhos com "99,83 % de acurácia" | Não há corte temporal: janelas sobrepostas vazam entre treino e teste. |
| Pôster Plaksha 2025 (Random Forest) | colunas de rótulo `fault`/`fault type` criadas a partir dos relatórios; **split 80/20** | — | precisão 98,2 %, recall 98,7 %, **acurácia 99,4 %**; multiclasse F1 97 % | Mesmo vazamento; acurácia não informa nada com cerca de 98 % de leituras normais. |
| harveyphm/MetroPT-3-Anomaly-Detection (9 classificadores) | split não declarado | — | precisão = recall = F1 = **1,00** (RF, XGBoost, rede neural, ensemble) | Um F1 perfeito em 9 modelos indica teste contaminado ou métrica por leitura dentro de falhas já vistas. |
| Najjar, Ashqar e Hasasneh 2023, classificação de falhas da APU (RF) | classificação por leitura | — | F1 ≈ 85 % (binário) / 97 % (multiclasse) | Classifica o *estado atual* (já em falha), não antecipa; sem split por evento. |
| Masa-Tantawy/MetroPT-3-Predictive-Maintenance | clustering, distâncias, DBSCAN; a abordagem 4 usa um **dataset balanceado artificialmente** e classificadores | amostra balanceada | nenhuma métrica no README | Balancear por amostragem e depois testar na mesma distribuição balanceada não mede o desempenho real. |

O PredictIQ começou aqui: a geração *legacy* (split aleatório + SMOTE) dava PR-AUC
0,999 (RF). Ela foi arquivada em `apps/ml/models/legacy/` justamente por isso.

### Grupo B — temporal / por evento (metodologicamente sólidos)

| Trabalho | Protocolo | Resultado | Concorda com o PredictIQ? |
|---|---|---|---|
| **mimouniahmed/metro-anomaly-benchmark** (34 configurações, 5 famílias: regra, SPC, ML clássico, séries temporais, autoencoders) | split cronológico; limiares calibrados só no treino; métrica por evento com antecedência, nível de acaso e falsos alarmes por semana; IC de Clopper-Pearson e Poisson | Recomendado: **CUSUM na corrente do motor + z-score da temperatura do óleo contra os 7 dias anteriores → 4/4, 0,71 FA/semana**, falha 4 avisada 12,9 h antes. Autoencoders 1–2/4 (4/4 só com o treino limpo, sem antecedência). ML e DL **igualam, mas não superam** os detectores estatísticos. "4/4 é compatível com taxa real de apenas 40 %." | Sim, nos pontos centrais: protocolo por evento, qualidade do rótulo acima do algoritmo, cautela com n pequeno. Ele **vai além** em dois pontos (baseline estatístico, limpeza do treino). |
| **SahilBh01r1769/metropt3-predictive-maintenance** (XGBoost, TCN, Attention-TCN; horizontes de 1, 3, 6 e 12 h) | segmentos de continuidade (334), janelas que não cruzam lacunas; maio e junho para desenvolvimento, **julho como evento de teste**; purge das janelas que compartilham dados com a avaliação; 3 seeds | AP lift em julho a 1 h: XGBoost 1,63×, TCN 0,75×, Attention-TCN 0,78×. **No limiar 0,5 nenhum modelo detecta julho.** O autor diz que o XGBoost "não resolve a tarefa". | Sim, quase ponto a ponto: o PredictIQ também usa features *gap-aware*, purge, seeds e alvo de antecedência, e também conclui que a antecipação não é aprendível e que cada falha tem precursores diferentes. |
| **Ribeiro, Davari, Veloso, Gama et al., arXiv 2502.07394** (regras interpretáveis, MetroPT2) | treino não supervisionado antes de 2022-06-01 (assumido sem falhas); online, janelas de 30 min com passo de 5 min; avaliação por evento com o LPS do trem como referência | autoencoder convolucional + regras: **2/2 falhas, 0 falso positivo**; vazamento de ar 150 min antes do LPS e vazamento de óleo mais de 2 dias antes. Regra dominante: `Flowmeter_max > 16`. **Os autores recomendam remover o Flowmeter** (separa as falhas sozinho); sem ele o autoencoder "sofre" e o MetroPT-3 é mais difícil. | Sim. O PredictIQ já **exclui o Flowmeter** dos modelos servidos (só aparece na ablação do `evaluate_cross_dataset`). |
| Davari, Veloso, Ribeiro e Gama (DSAA 2021, SAC 2024; equipe autora do dataset) | LSTM-AE e sparse AE na APU, foco em reduzir falsos alarmes; WAE-GAN relatado sem falsos alarmes | o LSTM-AE supera o sparse AE em F1, precisão e recall | Mesma família (autoencoder). No PredictIQ o autoencoder é o pior modelo supervisionado-comparável (1/6, 1,04 FA/dia), o que é coerente com o benchmark acima: um AE treinado em dados com vazamentos não rotulados fica "cego". |
| Veloso et al., *Scientific Data* 2022 (descritor do MetroPT) | descreve sensores, relatórios de falha e o alarme LPS | — | O PredictIQ segue a convenção: falhas dos relatórios de manutenção, LPS como referência operacional. |

## 3. Comparação ponto a ponto

| Aspecto | Grupo A | Grupo B (melhores práticas) | PredictIQ | Situação |
|---|---|---|---|---|
| Split | aleatório 80/20 | cronológico, falha inédita no teste | walk-forward purgado (6 folds, purge 300 s, embargo 3600 s) × LOFO interno | ✅ igual ou mais rigoroso (seleção aninhada) |
| Lacunas nos dados | ignoradas | segmentos de continuidade | features *gap-aware* (lacuna > 60 s abre segmento) | ✅ igual |
| Taxa real | assumida 1 Hz | o benchmark nota "a cada ~10 s, não 1 Hz" | amostragem pontual a 10 s | ✅ |
| Balanceamento | SMOTE / amostra balanceada | pesos ou nenhum; avaliação na distribuição real | `neg_stride` e `pos_weight_scale` ajustados, teste na distribuição real | ✅ |
| Limiar | 0,5 fixo | calibrado no treino | Platt nos OOF + ótimo de custo (RN-02) | ✅ mais completo |
| Métrica | acurácia / F1 por leitura | falhas detectadas, FA/semana, antecedência, nível de acaso | as mesmas + custo esperado, ECE, métricas clássicas por leitura e por alarme | ✅ |
| Incerteza | nenhuma | IC Clopper-Pearson, várias seeds | 3 seeds, Wilcoxon (RN-05), bootstrap | ✅ |
| Flowmeter | usado | removido (recomendação explícita) | excluído dos modelos servidos | ✅ |
| Teste em outra máquina | — | — | MetroPT2 (outra APU, 2 falhas posteriores a todo o treino) | ✅ **além** da literatura |
| Antecipação | não medida | horizontes de 1 a 12 h → fraca | 30 min / 1 h / 2 h com controle de acaso → não aprendível | ✅ mesma conclusão |
| **Baseline estatístico (CUSUM/EWMA)** | — | 18 de 34 configurações detectam 4/4; CUSUM + z-score: 0,71 FA/semana | CUSUM e EWMA no mesmo walk-forward aninhado e no gate RN-05 ([`statistical_baselines.md`](statistical_baselines.md)) | ✅ fechada em 2026-10-22 (seção 6) |
| **Limpeza de vazamentos não reportados no treino** | — | 8 episódios parecidos com vazamento encontrados e removidos; muda o resultado de AE e LOF | auditoria pré-registrada: fronteiras dos eventos, depois falsos alarmes; rótulos auditados só como análise secundária, sem retreino ([`label_audit.md`](label_audit.md)) | ✅ fechada em 2026-10-22 (seção 6) |

## 4. Números lado a lado (com ressalvas)

| Fonte | Dados | Detecção | Falsos alarmes | Antecedência |
|---|---|---|---|---|
| Benchmark mimouniahmed, CUSUM + z-score do óleo | MetroPT-3, 4 falhas | 4/4 | 0,71/semana ≈ **0,10/dia** | até 12,9 h (falha 4) |
| PredictIQ, BiLSTM (melhor custo) | MetroPT-3 + 2022, 7 falhas, 6 folds de teste | 3,0/6 | 0,29/dia | falha em andamento (sem antecipação aprendível) |
| PredictIQ, Random Forest (*champion*) | idem | 2,0/6 | 0,28/dia | idem |
| arXiv 2502.07394, AE + regras **com Flowmeter**, treinado na própria máquina | MetroPT2, 2 falhas | 2/2 | 0 | ar: 150 min antes do LPS; óleo: > 2 dias |
| PredictIQ, BiLSTM **sem Flowmeter**, treinado em **outra** máquina | MetroPT2, 2 falhas | 2/2 | 0,05/dia | ar: +32 min vs LPS; óleo: +56,5 h vs LPS |

As tarefas não são idênticas. O benchmark testa as mesmas 4 falhas que usou para
escolher os detectores (graus de liberdade do pesquisador). O PredictIQ testa 6
folds, incluindo a mudança de regime 2020 → 2022 (fold 4), que nenhum outro
trabalho avalia. O artigo do MetroPT2 treina na mesma máquina e usa o Flowmeter
que os próprios autores consideram fácil demais. Mesmo assim, a vantagem do
CUSUM em falsos alarmes (cerca de 3× menos) é grande o bastante para merecer um
teste no nosso protocolo.

## 5. Conclusões para a defesa

1. **Os 99–100 % da literatura não são comparáveis** e não indicam um modelo
   melhor: vêm de split aleatório por leitura. O próprio PredictIQ mostra o efeito
   (0,999 → 0,59).
2. **Os trabalhos rigorosos chegam às mesmas conclusões do PredictIQ:**
   - a antecipação é fraca ou não aprendível;
   - cada falha tem precursores diferentes;
   - com 4 a 7 eventos as diferenças entre modelos não são significativas;
   - a qualidade do rótulo pesa mais que o algoritmo.
3. **O PredictIQ vai além da literatura** em três pontos:
   - seleção aninhada sem vazamento;
   - teste em outra APU (MetroPT2) sem Flowmeter;
   - paridade treino–serviço medida em replay ao vivo.
4. **Lacunas a fechar para ficar acima do melhor trabalho público:**
   - incluir CUSUM/EWMA (corrente do motor, temperatura do óleo) como baseline
     no mesmo walk-forward e no gate RN-05;
   - detectar e excluir do treino os episódios parecidos com vazamento que não
     estão nos relatórios, e então reavaliar os falsos alarmes.

## 6. Lacunas fechadas (2026-10-22)

### Baseline estatístico no mesmo protocolo

CUSUM e EWMA entraram como "modelos" do protocolo aninhado. Sensor, `k`/`λ`,
referência (treino saudável ou 7 dias anteriores) e limiar são escolhidos só no
treino, por leave-one-failure-out interno e custo RN-02. Nada é escolhido
olhando as falhas de teste, ao contrário do benchmark público.

| | Falhas | FA/dia | Custo médio |
|---|---|---|---|
| EWMA | **4,0/6** | 0,64 | **50,2** |
| BiLSTM (melhor ML) | 3,0/6 | 0,29 | 57,6 |
| XGBoost | 2,7/6 | 0,26 | 61,7 |
| Random Forest (*champion*) | 2,0/6 | 0,28 | 74,7 |
| CUSUM | 2,0/6 | 0,47 | 79,0 |

* O EWMA na corrente do motor (escolhido pelo loop interno nos folds 3–6) tem o
  menor custo e detecta mais falhas que qualquer modelo, inclusive o fold 4
  (mudança de regime 2020 → 2022), mas com cerca de 2× mais alarmes falsos.
* O CUSUM fica no nível do Random Forest.
* **Nenhuma comparação passa no gate RN-05**, em nenhuma direção: EWMA vs RF
  p = 0,094; os melhores modelos vs EWMA p ≥ 0,58. Com 6 folds, ML, DL e
  controle estatístico de processo empatam.
* Ressalva: no fold 1 (só 1 falha no treino) os parâmetros a priori são os do
  benchmark, que viu as falhas do MetroPT-3.

**Leitura para a banca:** a complexidade não se pagou com 7 falhas. É o mesmo
achado do benchmark público, agora sob um protocolo sem graus de liberdade do
pesquisador.

### Auditoria dos rótulos

Protocolo pré-registrado em [`label_audit_protocol.md`](label_audit_protocol.md)
antes de olhar os alarmes; resultado em [`label_audit.md`](label_audit.md).

* **Etapa A (fronteiras):** com os critérios pré-registrados, F20_1 e F20_2 não
  mostram evidência porque o critério de ciclo de trabalho satura. **3,4 % dos
  blocos "saudáveis" de 2020** têm o compressor em carga contínua por 1 h, a
  mesma assinatura das falhas rotuladas. Na variante declarada (`≥ p99`), as
  6 falhas de ar têm evidência física. Ela começa 0,5–1,5 h **depois** do início
  rotulado, ou seja, os rótulos começam cedo. O secador (F22_2) não tem
  evidência; o vazamento de óleo (F22_3) só tem evidência fraca (3 % dos blocos).
* **Etapa B (falsos alarmes):** dois episódios longos em maio de 2020
  (13/05, 11,5 h; 19–20/05, 21,4 h) disparam 8 a 9 dos 10 detectores e têm
  queda de pressão e carga contínua. São os candidatos mais fortes a vazamento
  não reportado.
* **Análise secundária:** tratando esses períodos como ambíguos, os alarmes
  falsos do XGBoost caem de 36 para 0 e os do BiLSTM de 37 para 1 (variante
  `≥`). Pelos critérios principais, caem para 22 e 23. Os do EWMA quase não
  mudam (101 → 91). Ou seja, **a maior parte dos alarmes "falsos" dos melhores
  modelos coincide com prováveis vazamentos reais**, enquanto os do EWMA são,
  em boa parte, alarmes falsos de fato. Isso não muda a métrica principal nem
  nenhum modelo. Os dois períodos estão no teste do fold 1, então não podem
  virar ajuste.

## Fontes

- Veloso et al., *The MetroPT dataset for predictive maintenance*, Scientific Data 2022: https://www.nature.com/articles/s41597-022-01877-3
- Ribeiro, Davari, Veloso, Gama et al., *Interpretable Rules for Online Failure Prediction: A Case Study on the Metro do Porto dataset*: https://arxiv.org/html/2502.07394v1
- Davari et al., *Detecting and Explaining Anomalies in the Air Production Unit of a Train* (SAC 2024): https://dl.acm.org/doi/10.1145/3605098.3635906
- *Predictive Maintenance of a Metro's Air Compressor* (IEEE): https://ieeexplore.ieee.org/document/10689744/
- *Predictive Modeling for Metro Performance Using MetroPT3 Dataset* (IJACSA 16(8), 2025): https://thesai.org/Downloads/Volume16No8/Paper_54-Predictive_Modeling_for_Metro_Performance.pdf
- mimouniahmed/metro-anomaly-benchmark: https://github.com/mimouniahmed/metro-anomaly-benchmark
- SahilBh01r1769/metropt3-predictive-maintenance: https://github.com/SahilBh01r1769/metropt3-predictive-maintenance
- harveyphm/MetroPT-3-Anomaly-Detection: https://github.com/harveyphm/MetroPT-3-Anomaly-Detection
- Masa-Tantawy/MetroPT-3-Predictive-Maintenance: https://github.com/Masa-Tantawy/MetroPT-3-Predictive-Maintenance
- Classificação de falhas da APU (MDSOAR): https://mdsoar.org/items/d5c7490c-f651-4237-b050-ed7765d8d762
- Pôster Plaksha 2025: https://ai3011.plaksha.edu.in/Spring%202025/PDFs/Meher%20Sidhu.pdf
