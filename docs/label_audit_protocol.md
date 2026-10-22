# Auditoria dos rótulos — protocolo pré-registrado

> Escrito em 2026-10-22, **antes** de olhar qualquer alarme falso dos modelos.
> Implementação: `apps/ml/src/audit_labels.py`. Resultado:
> [`label_audit.md`](label_audit.md).

## Por que

A comparação com a literatura ([`related_work_comparison.md`](related_work_comparison.md))
apontou que parte dos nossos "falsos alarmes" pode ser vazamento real não
registrado nos relatórios de manutenção. Não dá para classificar um alarme como
"falha não reportada" sem antes saber onde cada falha rotulada começa e termina.
Por isso a auditoria tem duas etapas, nesta ordem: (A) fronteiras dos eventos,
(B) falsos alarmes.

## Regras que não mudam

1. **Rótulos originais continuam sendo a métrica principal.** Os rótulos
   auditados entram só como análise secundária.
2. **Nada da auditoria volta para o modelo.** Nenhum modelo é retreinado,
   nenhum limiar ou hiperparâmetro é escolhido com os rótulos auditados. Se a
   auditoria achar falhas no período de teste, elas mudam apenas a contagem
   secundária de falsos alarmes.
3. Os critérios abaixo são fixos. Se um critério se mostrar inadequado, isso é
   reportado como limitação, não corrigido depois de ver o resultado.

## Evidências (calculadas em blocos de 10 min)

Referência saudável **por dataset**: linhas com `anomaly = 0` a mais de 24 h de
qualquer falha rotulada. É uma auditoria dos rótulos, não um modelo; por isso
usa o dataset inteiro.

| Código | Evidência | Regra no bloco de 10 min |
|---|---|---|
| E1 | Queda de pressão | média de `TP3` abaixo do percentil 1 da referência **ou** resíduo de enchimento do gêmeo físico (`fill_residual`) abaixo do percentil 1 |
| E2 | Ciclo de trabalho anormal | resíduo de ciclo de trabalho do gêmeo físico (`duty_residual`, janela de 1 h) acima do percentil 99 |
| E3 | Sinal LPS | `LPS = 1` em pelo menos uma leitura do bloco |
| E4 | Proximidade de manutenção registrada | bloco a menos de 24 h de uma falha rotulada (início − 24 h, fim + 24 h) |

E1–E3 vêm dos sensores; E4 vem dos relatórios. O gêmeo físico é o de
`digital_twin.py`, ajustado na referência saudável com `pressure_col = "TP3"`.

**Limitação declarada:** E1–E3 descrevem vazamento de ar. O vazamento de óleo
(F22_3) e a falha do secador (F22_2) podem não ter evidência nessas regras.

## Etapa A — fronteiras dos eventos

Para cada falha rotulada, numa janela de busca de ±48 h em torno do rótulo:

* bloco com evidência = pelo menos uma de E1–E3;
* trecho com evidência = blocos com evidência separados por no máximo 1 bloco
  sem evidência;
* **início físico** = início do primeiro trecho com evidência que toca o
  intervalo rotulado; **fim físico** = fim do último trecho com evidência que
  toca o intervalo rotulado;
* se nenhum trecho toca o intervalo, a falha fica marcada como "sem evidência
  nos sensores" e a fronteira rotulada é mantida.

Saída: diferença (h) entre fronteira física e rotulada, por falha.

## Etapa B — falsos alarmes

Fonte: episódios de alarme (persistência k = 6, como RN-03) de todos os modelos
da avaliação aninhada (seed 0, 6 folds fora do tempo), apenas a parte **fora**
das janelas de evento usadas na métrica (início − 6 h, fim + 1 h). Episódios de
modelos diferentes a menos de 1 h um do outro formam um **agrupamento**.

Classe de cada agrupamento, na ordem:

1. **Fronteira** — toca o intervalo físico auditado da etapa A ampliado pela
   mesma janela da métrica (−6 h, +1 h), ou E4 está ativo. É um problema de
   fronteira do rótulo, não um alarme sem causa.
2. **Anomalia não confirmada** (nome original: "candidato a falha não reportada") — pelo menos 2 evidências distintas de
   E1–E3 aparecem no agrupamento, e pelo menos uma delas em 3 ou mais blocos
   (30 min).
3. **Evidência fraca** — exatamente uma de E1–E3.
4. **Sem evidência** — nenhuma; alarme falso de fato.

## Análise secundária

Para cada modelo, recalcula falsos alarmes e custo esperado (RN-02) tratando
como **período ambíguo** (nem acerto nem alarme falso) o tempo dentro de
agrupamentos das classes 1 e 2. Detecções não mudam, pois os eventos rotulados
continuam os mesmos. Reporta-se lado a lado: rótulo original (principal) e
rótulo auditado (secundário).

## Emendas (registradas depois da execução, sem mudar critérios)

* **2026-10-22 — desvio declarado do E2.** A etapa A mostrou que `duty > p99`
  nunca dispara em 2020 (o resíduo satura e o p99 coincide com o teto). O
  critério acima continua sendo o resultado principal; a variante `duty ≥ p99`
  é reportada à parte como análise de sensibilidade.
* **2026-10-22 — nomenclatura e exposição.** A classe 2 passa a se chamar
  **"anomalia não confirmada"** (antes "candidato a falha não reportada"): a
  regra de classificação é a mesma, só o nome deixa de sugerir uma falha. A
  análise secundária passa a listar os intervalos excluídos e a reportar
  alarmes falsos por dia com o tempo excluído descontado da exposição. Também
  foi acrescentada a cronologia por falha (início documentado, primeira
  evidência, primeiro alarme), que é descritiva.
