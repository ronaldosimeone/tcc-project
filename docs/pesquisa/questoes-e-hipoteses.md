# Questões de pesquisa, hipóteses e plano experimental

Este documento amarra cada experimento do repositório a uma pergunta de
pesquisa e a uma hipótese falseável. A regra é: **nenhum experimento sem
hipótese, nenhuma hipótese sem critério de rejeição definido antes de rodar.**

## Problema

A manutenção preditiva de compressores ferroviários enfrenta três lacunas que
a literatura costuma tratar separadamente:

1. **Avaliação otimista.** Trabalhos sobre o MetroPT-3 reportam F1 ≥ 0,95 com
   splits aleatórios sobre amostras de série temporal. Com *features* de
   janela deslizante, isso vaza informação entre treino e teste. O próprio
   histórico deste projeto reproduziu o problema: AUC = 1,0000 no split
   aleatório (ver [`temporal_evaluation.md`](../temporal_evaluation.md)).
2. **Alarme sem ação.** Um alarme diz *que* algo está errado, mas não *o
   quê* nem *o que fazer*. O técnico ainda precisa diagnosticar e consultar
   manuais.
3. **LLM sem fundamentação verificável.** Assistentes baseados em LLM
   produzem procedimentos plausíveis, mas sem garantia de que vêm do manual
   correto. Em manutenção industrial isso é risco de segurança.

## Objetivo geral

Propor e avaliar uma arquitetura que liga **detecção de falha avaliada sem
vazamento**, **explicação por sensor (XAI)** e **recuperação de procedimentos
técnicos via MCP + RAG com LLM local**, medindo em cada elo a qualidade que
chega ao técnico.

## Questões de pesquisa

| ID | Pergunta |
|---|---|
| QP1 | Quanto da performance reportada para o MetroPT-3 sobrevive a um protocolo temporal sem vazamento (walk-forward com *purge*)? |
| QP2 | Os detectores avisam **antes** da falha com antecedência útil (lead time ≥ 2 h) e taxa de falsos alarmes operável? |
| QP3 | As probabilidades e os conjuntos conformais se mantêm calibrados sob a mudança temporal entre falhas? |
| QP4 | As explicações SHAP agregadas por sensor são **fiéis** (deleção) e **estáveis** entre falhas a ponto de servirem de entrada para diagnóstico? |
| QP5 | Injetar a explicação XAI na consulta do RAG via MCP aumenta a precisão do contexto recuperado e a fidelidade da resposta do LLM, em comparação com o RAG só com a descrição do equipamento? |
| QP6 | Um limiar de fundamentação (RN-04) reduz a taxa de alucinação sem bloquear a maioria das respostas corretas? |

## Hipóteses

| ID | Hipótese | Métrica / teste | Critério de rejeição de H0 | Experimento |
|---|---|---|---|---|
| H1 | O split aleatório superestima o desempenho em relação ao walk-forward com *purge* | PR-AUC por fold; diferença legado − temporal | Diferença > 0 em todos os modelos e IC95 % do walk-forward abaixo do valor legado | `src/evaluate_temporal.py` |
| H2 | Pelo menos um detector atinge recall de eventos ≥ 0,75 com ≤ 1 falso alarme/dia e lead time médio ≥ 2 h | Event recall, FA/dia, lead time (RF-29) | Os três critérios satisfeitos com o limiar escolhido **sem** ver o teste | `src/evaluate_temporal.py` (`event_inner`) |
| H3 | Modelos de árvore superam a regressão logística em custo esperado (RN-02) | Friedman + Nemenyi sobre os folds | p < 0,05 (Friedman) e diferença de rank > CD | idem |
| H4 | O conformal de Mondrian mantém cobertura ≥ 0,90 por classe sob o protocolo temporal | Cobertura empírica por classe | Cobertura da classe falha < 0,85 em algum fold **rejeita** a hipótese (não-permutabilidade) | idem (`uncertainty`) |
| H5 | O top-k SHAP é fiel: remover as k features mais importantes derruba P(falha) mais que remover k aleatórias | Curva de deleção | Queda SHAP > queda aleatória para k ∈ {3, 5, 10} em todos os folds | `src/explain_events.py` |
| H6 | Os rankings de sensores são estáveis entre falhas do mesmo tipo (*air leak*) | Kendall τ entre folds | τ > 0,5 em todos os pares | idem |
| H7 | Consulta XAI-aumentada (sensores do SHAP + equipamento) melhora a *context precision* do RAG via MCP | Context precision@5 no benchmark anotado (RF-36) | Wilcoxon pareado por caso, p < 0,05 | RF-36/RF-37 (bloco 2) |
| H8 | O limiar τ calibrado reduz alucinação em ≥ 50 % com cobertura de respostas ≥ 70 % | Taxa de alucinação (RNF-82) × taxa de resposta | Ambos os critérios no benchmark anotado | RF-36/RN-04 (bloco 2) |

## Ameaças à validade

| Tipo | Ameaça | Mitigação adotada | Resíduo declarado |
|---|---|---|---|
| Conclusão | Só 4 falhas reais no MetroPT-3: 3 folds walk-forward, poder estatístico baixo | Bootstrap IC95 %, Friedman/Nemenyi com N explícito; nenhuma afirmação de significância sem p | Testes de rank com N = 3–4 quase nunca rejeitam H0. Replicação em 2º dataset (RF-33) |
| Interna | Limiar ajustado no teste | Três políticas: fixo 0,5, *inner* (OOF do treino) e *oracle* (só limite superior, nunca usado em comparação) | O fold 1 não tem 2 falhas no treino e usa o limiar fixo |
| Interna | Subamostragem de negativos no treino | Teste sempre em resolução total; mesma subamostragem para todos os modelos | Pode reduzir a precisão absoluta, não a comparação |
| Interna | *Features* calculadas sobre a série inteira | Todas são *backward-looking*; *purge* de 300 s > maior janela (150 s) | — |
| Construto | Rótulo = intervalo do relatório de falha, não início físico da degradação | Lead time medido em relação ao início reportado | O lead time real pode ser maior ou menor |
| Construto | Custo C_FN = 100·C_FP é suposição | Parâmetro explícito; análise de sensibilidade na tese | — |
| Externa | Um compressor, um tipo de falha (*air leak*), um trem | RF-33: replicar em MetroPT-2 / outro dataset | Generalização não demonstrada até o RF-33 |
| Externa (RAG) | Corpus pequeno e manual fora do domínio exato (WAP-5) | Curadoria por sidecar e benchmark de corpus ([`CORPUS-CURATION.md`](../../apps/mcp-server/CORPUS-CURATION.md)) | Falta manual de *fault finding* do compressor real |

## Mapa experimento → artefato

| Experimento | Comando | Saída |
|---|---|---|
| Avaliação temporal (H1–H4) | `python -m src.evaluate_temporal` | `apps/ml/models/eval_temporal.json` |
| Tabelas e figuras | `python -m src.render_temporal_report` | `docs/temporal_evaluation.md`, `docs/figures-ml/` |
| Explicabilidade (H5–H6) | `python -m src.explain_events` | `apps/ml/models/explain_events.json` |
| Curadoria do corpus | `python benchmark_corpus_curation.py --wap5 <pdf>` | `apps/mcp-server/corpus_curation_benchmark.json` |
