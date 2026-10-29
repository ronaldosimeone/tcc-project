# Guia de diagnóstico da APU — compressor de ar do trem (PredictIQ)

Guia escrito pelo projeto PredictIQ (TCC). NÃO é manual do fabricante: reúne
a descrição dos sensores do dataset MetroPT (Veloso et al., Scientific Data,
2022) e boas práticas gerais de manutenção de compressores de ar. Em caso de
divergência, vale o manual do fabricante e o procedimento da oficina.

## Sensores da APU e valores de referência

A APU (unidade de produção de ar) alimenta freios, portas e suspensão.
TP3 e Reservoirs: pressão do painel pneumático e dos reservatórios, normalmente
entre 8 e 10 bar. TP2: pressão na saída do compressor; fica perto de 0 bar com
o compressor em alívio. Motor_current: cerca de 0 A desligado, 4 A em alívio,
7 A em carga e 9 A na partida. COMP: ativo quando não há admissão de ar
(compressor desligado ou em alívio). MPG: liga o compressor em carga quando a
pressão cai abaixo de 8,2 bar. LPS: alarme de baixa pressão, ativo abaixo de
7 bar. Towers: indica qual torre do secador está secando o ar. Oil_level:
ativo quando o nível de óleo está abaixo do esperado.

## Vazamento de ar no compressor — sintomas

Vazamento de ar no compressor de ar industrial: o ar comprimido escapa do
circuito pneumático e a pressão cai. Sinais típicos de vazamento de ar:
- TP3 e Reservoirs caem mais rápido que o normal com o compressor em alívio;
- o compressor liga com mais frequência e fica mais tempo em carga
  (Motor_current em 7 A, COMP inativo e MPG acionando mais vezes);
- ciclos de carga curtos e repetidos mesmo sem aumento de consumo de ar;
- nos casos graves, a pressão chega a 7 bar e o alarme LPS dispara.
O vazamento costuma começar pequeno e crescer; perceber a mudança no ciclo do
compressor antes do LPS dá tempo para a manutenção.

## Vazamento de ar no compressor — verificação passo a passo

Como localizar um vazamento de ar no compressor de ar e no circuito:
1. Com o trem parado e o compressor em alívio, anotar TP3 e medir quanto a
   pressão cai em 10 minutos sem consumo (teste de queda de pressão).
2. Comparar com a queda de um trem sem falha; queda maior indica vazamento.
3. Ouvir o circuito: vazamentos maiores produzem chiado audível.
4. Aplicar solução de água e sabão em conexões, mangueiras, juntas, válvulas
   de dreno e válvulas de segurança; bolhas mostram o ponto de fuga.
5. Verificar os consumidores (freio, portas, suspensão) isolando um ramo por
   vez para localizar o trecho com perda.
6. Checar se as válvulas de dreno dos reservatórios fecham completamente.

## Vazamento de ar no compressor — correção e liberação

Como corrigir um vazamento de ar no compressor de ar e liberar o trem:
- Despressurizar o circuito antes de qualquer desmontagem.
- Reapertar conexões frouxas; substituir juntas, anéis de vedação, mangueiras
  ou válvulas danificadas; nunca vedar com fita um componente trincado.
- Após o reparo, repetir o teste de queda de pressão: a queda deve voltar ao
  nível de um trem sem falha.
- Liberar o trem só com o ciclo do compressor normal (carga e alívio em
  intervalos regulares) e sem alarme LPS.
- Registrar na ordem de serviço o ponto do vazamento e a peça trocada.

## Vazamento de óleo no compressor — sintomas

Sinais típicos de vazamento ou perda de óleo no compressor da APU:
- indicador Oil_level ativo (nível de óleo abaixo do esperado);
- temperatura do óleo (Oil_temperature) acima do habitual em carga;
- manchas ou gotejamento de óleo sob o compressor ou nas conexões;
- óleo arrastado para o ar comprimido: névoa ou óleo no dreno dos
  reservatórios e no secador;
- em casos avançados, a eficiência do compressor cai e ele passa mais tempo
  em carga para manter a pressão.

## Vazamento de óleo no compressor — verificação e correção

1. Desligar e despressurizar; aguardar o óleo esfriar (risco de queimadura).
2. Conferir o nível de óleo no visor ou vareta, com o compressor nivelado.
3. Inspecionar retentores, juntas do cárter, bujão de dreno, filtro de óleo,
   conexões e o separador ar/óleo à procura de umidade de óleo.
4. Verificar o dreno dos reservatórios: óleo no condensado indica separador
   saturado ou desgaste interno.
5. Substituir a peça com vazamento, completar o óleo com o tipo especificado
   pelo fabricante e trocar o filtro se estiver contaminado.
6. Funcionar o compressor e confirmar nível estável e temperatura normal.

## Secador de ar (torres) — sintomas e verificação

Falha no secador de ar do compressor: o secador de torre dupla retira a
umidade do ar comprimido; enquanto uma torre seca, a outra se regenera.
Sintomas de falha no secador de ar ou na sua válvula:
- o sinal Towers deixa de alternar ou alterna fora do padrão;
- perda de ar contínua pela purga do secador (queda de TP3 com o compressor
  em alívio), parecida com um vazamento de ar;
- água no dreno dos reservatórios e na linha de ar.
Verificação: observar a troca de torres no ciclo; checar se a válvula de
purga fecha; inspecionar a válvula do secador e o dessecante (saturado ou
contaminado por óleo); substituir a válvula ou o dessecante conforme o caso.

## Alarme LPS — pressão baixa (ação imediata)

O LPS indica pressão do circuito abaixo de 7 bar: falta ar para freios e
portas. Ações:
- comunicar o centro de controle e seguir o procedimento operacional para
  retirar o trem de serviço com segurança;
- não tentar religar repetidamente o compressor sem diagnóstico;
- depois de imobilizado, investigar vazamento de ar, falha do secador ou
  falha do próprio compressor (corrente do motor, válvula de admissão).
O objetivo do monitoramento é detectar a falha antes de o LPS disparar.

## Segurança na manutenção da APU

- Despressurizar todo o circuito e confirmar 0 bar antes de abrir qualquer
  componente pneumático.
- Bloquear e etiquetar a alimentação elétrica do motor do compressor.
- Óleo e cabeçote podem estar quentes: aguardar o resfriamento.
- Usar óculos e luvas; ar comprimido e óleo sob pressão causam lesões.
- Usar somente peças e óleo especificados pelo fabricante.
