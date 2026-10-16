# Protocolo de revisão sistemática (PRISMA 2020)

> Protocolo definido **antes** da busca. A execução nas bases exige acesso
> institucional (Scopus, IEEE Xplore, Web of Science, ACM DL) e fica a cargo
> do autor; este arquivo fixa as decisões para que a revisão seja
> reprodutível e auditável pela banca.

## 1. Perguntas da revisão

- **RQ-R1** Como trabalhos de manutenção preditiva em compressores/sistemas pneumáticos ferroviários **validam** seus modelos? Quantos usam um split que respeita a ordem temporal?
- **RQ-R2** Quais trabalhos avaliam a detecção **por evento** (lead time, falsos alarmes/dia) em vez de por amostra?
- **RQ-R3** Quais trabalhos combinam **XAI** com **LLM/RAG** para gerar recomendações de manutenção? Como avaliam a fidelidade da recomendação?
- **RQ-R4** Há uso do **Model Context Protocol** (ou *tool calling* equivalente) em sistemas industriais de manutenção?

## 2. Estratégia de busca

Período: 2015-01-01 a data da busca. Idiomas: inglês, português.

| Bloco | Termos |
|---|---|
| Domínio | ("predictive maintenance" OR "prognostics" OR "condition monitoring" OR "fault detection" OR "anomaly detection") |
| Ativo | ("compressor" OR "air production unit" OR "APU" OR "pneumatic" OR "railway" OR "metro" OR "MetroPT") |
| Método A (avaliação) | ("time series" AND ("cross-validation" OR "walk-forward" OR "temporal split" OR "data leakage")) |
| Método B (LLM) | ("large language model" OR "LLM" OR "retrieval-augmented generation" OR "RAG" OR "Model Context Protocol") AND ("maintenance") |
| Método C (XAI) | ("SHAP" OR "explainable" OR "XAI" OR "interpretab*") |

Strings por base (ajustar a sintaxe de cada uma):

- **S1 (RQ-R1/R2):** Domínio AND Ativo AND Método A
- **S2 (RQ-R3/R4):** Domínio AND Método B
- **S3 (RQ-R3):** Domínio AND Método B AND Método C
- **Snowballing:** backward e forward a partir de Veloso et al. (MetroPT, *Scientific Data*, 2022) e de todo estudo incluído que use o MetroPT-2/3.

## 3. Critérios

**Inclusão**
- I1 Estudo primário com avaliação empírica.
- I2 Dados de sensores de equipamento industrial **ou** sistema LLM/RAG aplicado à manutenção.
- I3 Texto completo disponível.

**Exclusão**
- E1 Só resumo, pôster, editorial, tese não publicada.
- E2 Sem descrição suficiente do protocolo de validação (para S1).
- E3 Domínio sem equipamento físico (ex.: manutenção de software).
- E4 Duplicata ou versão anterior de estudo já incluído.

## 4. Processo

1. Exportar os resultados (RIS/BibTeX) e remover duplicatas (Zotero/Rayyan).
2. Triagem por título e resumo, feita por **dois revisores independentes**, com concordância medida por κ de Cohen. Divergências são resolvidas por discussão.
3. Leitura completa e aplicação de I/E, registrando o motivo de cada exclusão.
4. Extração para a planilha da §5.
5. Fluxograma PRISMA 2020 com as contagens de cada etapa.

## 5. Formulário de extração

| Campo | Valores |
|---|---|
| Referência | autor, ano, venue |
| Dataset | nome, nº de falhas reais, frequência |
| Split | aleatório / estratificado / temporal / walk-forward / por ativo / não informado |
| Vazamento discutido? | sim / não |
| Nível de métrica | amostra / janela / evento |
| Lead time reportado? | sim (valor) / não |
| Falsos alarmes/tempo? | sim / não |
| Incerteza | nenhuma / IC / calibração / conformal |
| XAI | nenhum / SHAP / LIME / atenção / outro; fidelidade avaliada? |
| LLM/RAG | não / sim (modelo, local ou API, avaliação da fidelidade) |
| MCP/tool calling | sim / não |
| Melhor resultado reportado | métrica + valor |

## 6. Qualidade / risco de viés

Checklist adaptado (0–2 pontos cada): protocolo temporal; ajuste de
hiperparâmetros/limiar sem usar o teste; múltiplas seeds/folds; teste
estatístico; código ou dados disponíveis; métrica no nível de evento.

## 7. Síntese esperada

Tabela comparativa estudos × campos da §5, com destaque para a lacuna que
esta tese ocupa: **avaliação temporal por evento + XAI fiel + RAG via MCP
avaliado quantitativamente**, num único sistema implantado.
