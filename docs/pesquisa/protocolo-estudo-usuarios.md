# Protocolo do estudo com usuários

Validação empírica com quem usaria o sistema: o assistente (XAI + RAG via
MCP + histórico com causa raiz) **reduz o tempo e o esforço para chegar ao
procedimento correto**, sem aumentar o risco de seguir uma instrução errada?

> Protocolo definido antes da coleta. Envolve pessoas: **exige aprovação de
> Comitê de Ética em Pesquisa (CEP, via Plataforma Brasil)** antes de
> recrutar qualquer participante. Este documento é a base do projeto a
> submeter ao CEP.

## 1. Questões e hipóteses

| ID | Pergunta | Hipótese | Medida |
|---|---|---|---|
| QU1 | O assistente reduz o tempo até o diagnóstico/procedimento? | H-U1: tempo com assistente < sem assistente | Tempo (s) até a resposta final |
| QU2 | O assistente melhora a correção do procedimento escolhido? | H-U2: acurácia com ≥ sem | Resposta correta segundo o gabarito do especialista |
| QU3 | Reduz a carga de trabalho percebida? | H-U3: NASA-TLX com < sem | NASA-TLX ponderado (0–100) |
| QU4 | O sistema é usável? | H-U4: SUS ≥ 68 (média de referência) | System Usability Scale (0–100) |
| QU5 | Os participantes detectam um plano não fundamentado? | H-U5: ≥ 80 % rejeitam o plano adulterado | Taxa de rejeição no caso-armadilha |

## 2. Participantes

- **Perfil**: técnicos de manutenção, engenheiros de manutenção ou estudantes de último ano de engenharia mecânica/elétrica/mecatrônica com experiência em sistemas pneumáticos.
- **Tamanho**: n = 16 a 24. Justificativa: desenho intra-sujeitos, efeito esperado grande (d ≈ 0,8) no tempo; para Wilcoxon pareado com α = 0,05 e poder 0,8, n ≈ 15. 16 permite contrabalanceamento completo em 2 ordens × 8.
- **Exclusão**: quem participou do desenvolvimento do sistema.
- **Consentimento**: TCLE assinado; participação voluntária; dados anonimizados (ID numérico).

## 3. Desenho

Intra-sujeitos, **2 condições × 6 tarefas**, ordem das condições contrabalanceada (metade começa com assistente):

- **C0 — sem assistente**: histórico real (tabela de eventos, gráficos), manual em PDF e busca textual do leitor de PDF.
- **C1 — com assistente**: o mesmo, mais o drawer de causa raiz (contribuição por sensor), o plano RAG com citações e o selo de fundamentação.

Tarefas: 6 cenários de falha extraídos de `apps/backend/benchmarks/rag_cases_v1.json` (validados pelo especialista, RF-36), 3 por condição, com dificuldade emparelhada entre condições. Um cenário de C1 é **caso-armadilha**: o plano exibido foi adulterado para citar uma página não recuperada e propor um passo inseguro. Mede a confiança excessiva (QU5); o participante é informado no *debriefing*.

## 4. Procedimento (≈ 60 min por participante)

1. TCLE e questionário de perfil (5 min).
2. Treino com 1 cenário fora do conjunto (10 min).
3. Bloco A: 3 tarefas na condição sorteada. Após o bloco, NASA-TLX (20 min).
4. Bloco B: 3 tarefas na outra condição, mais NASA-TLX (20 min).
5. SUS sobre o sistema completo e entrevista semiestruturada com 3 perguntas abertas (10 min).
6. *Debriefing* sobre o caso-armadilha.

O tempo é registrado pela própria interface (início da tarefa até a submissão da resposta) e as ações pela trilha de auditoria (`/v1/audit`), o que dá log objetivo, sem cronômetro manual.

## 5. Instrumentos

- **SUS** (Brooke, 1996), versão em português validada.
- **NASA-TLX** (Hart & Staveland, 1988), com as 6 subescalas e a ponderação por pares.
- **Gabarito**: para cada cenário, o procedimento correto e as páginas do manual, definidos pelo especialista antes do estudo.

## 6. Análise

- H-U1/H-U3: Wilcoxon pareado (C1 × C0), mediana por participante; tamanho de efeito r = Z/√N; IC 95 % por bootstrap.
- H-U2: teste de McNemar sobre acertos pareados por tarefa emparelhada.
- H-U4: média do SUS com IC 95 %, comparada a 68.
- H-U5: proporção de rejeição com IC de Wilson.
- Qualitativo: análise temática das entrevistas.
- Correção para múltiplas hipóteses: Holm–Bonferroni sobre H-U1…H-U3.

## 7. Ameaças à validade

| Ameaça | Mitigação |
|---|---|
| Efeito de aprendizado entre blocos | Contrabalanceamento e cenários diferentes em cada condição |
| Estudantes ≠ técnicos de campo | Estratificar a análise por experiência e reportar separadamente |
| Cenários vindos de um único manual/equipamento | Declarado como limitação externa; replicar com outro manual de *fault finding* |
| Efeito novidade a favor do assistente | Treino antes dos blocos e caso-armadilha para medir confiança excessiva |

## 8. Dados e ética

Os dados ficam anonimizados em `docs/pesquisa/estudo-usuarios/dados/`, fora do repositório público até a aprovação e só com o consentimento. Nada de nome, e-mail ou empresa. A gravação de tela só acontece com consentimento explícito e separado.
