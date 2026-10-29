# Guia de diagnóstico e reparo da APU — compressor de ar do trem (PredictIQ)

Guia escrito pelo projeto PredictIQ (TCC). NÃO é manual do fabricante: reúne
a descrição dos sensores do dataset MetroPT (Veloso et al., Scientific Data,
2022) e boas práticas gerais de manutenção de compressores de ar. Valores de
aperto, tipo de óleo, códigos de peça e intervalos vêm SEMPRE do manual do
fabricante; em caso de divergência, vale o manual do fabricante e o
procedimento da oficina.

## Sensores da APU e valores de referência

Sensores da APU (unidade de produção de ar) e valores de referência. A APU
alimenta freios, portas e suspensão.
- TP3 e Reservoirs: pressão do painel pneumático e dos reservatórios,
  normalmente entre 8 e 10 bar.
- TP2: pressão na saída do compressor; perto de 0 bar com o compressor em alívio.
- Motor_current: cerca de 0 A desligado, 4 A em alívio, 7 A em carga e 9 A
  na partida.
- COMP: ativo quando não há admissão de ar (compressor desligado ou em alívio).
- DV_eletric: válvula de saída, ativa com o compressor em carga.
- MPG: liga o compressor em carga quando a pressão cai abaixo de 8,2 bar.
- LPS: alarme de baixa pressão, ativo abaixo de 7 bar.
- Towers: indica qual torre do secador está secando o ar.
- Oil_level: ativo quando o nível de óleo está abaixo do esperado.

## Vazamento de ar no compressor — guia completo de reparo

Vazamento de ar no compressor de ar industrial: o ar comprimido escapa do
circuito pneumático e a pressão cai. Guia de reparo do vazamento de ar.

**Diagnóstico.** TP3 e Reservoirs caem mais rápido que o normal com o
compressor em alívio; o compressor liga com mais frequência e fica mais tempo
em carga (Motor_current em 7 A, MPG acionando mais vezes); ciclos de carga
curtos e repetidos; chiado perto de conexões ou válvulas; em casos graves, a
pressão chega a 7 bar e o alarme LPS dispara.

**Causas prováveis.** Conexões frouxas por vibração; mangueiras ressecadas ou
trincadas; juntas e anéis de vedação gastos; válvula de dreno do reservatório
que não fecha; válvula de retenção da saída com vedação danificada; válvula de
segurança aliviando abaixo da pressão; purga do secador de ar aberta.

**Ferramentas.** Jogo de chaves combinadas e de soquete; chave de torque
calibrada; manômetro de referência; solução de água e sabão ou spray detector
de vazamento; cronômetro; lanterna e espelho de inspeção; cadeado e etiqueta
de bloqueio elétrico.

**Peças e materiais.** Kit de anéis de vedação (o-rings) e juntas do modelo do
compressor; mangueiras e abraçadeiras com a mesma especificação; veda-rosca
próprio para ar comprimido (só em roscas cônicas); reparo ou válvula de dreno,
válvula de retenção e válvula de segurança novas, conforme o componente.

**Localização do vazamento.**
1. Com o trem parado e o compressor em alívio, anotar TP3 e medir quanto a
   pressão cai em 10 minutos sem consumo (teste de queda de pressão).
2. Comparar com a queda de um trem sem falha; queda maior confirma o vazamento.
3. Ouvir o circuito à procura de chiado.
4. Aplicar água e sabão em conexões, mangueiras, juntas e válvulas; bolhas
   mostram o ponto de fuga. Marcar todos os pontos.
5. Isolar um ramo de consumidores por vez (freio, portas, suspensão).

**Reparo passo a passo.**
1. Desligar o compressor, isolar a energia do motor com cadeado e etiqueta,
   despressurizar pelos drenos e confirmar 0 bar no manômetro.
2. Conexão frouxa: reapertar no torque do fabricante; se continuar, trocar o
   anel de vedação. Rosca cônica: limpar e aplicar veda-rosca novo.
3. Mangueira trincada ou ressecada: substituir a mangueira inteira e as
   abraçadeiras; não fazer emendas.
4. Junta de flange: soltar os parafusos em cruz, trocar a junta, limpar as
   faces e apertar em cruz no torque do fabricante.
5. Válvula de dreno que não fecha: limpar a sede e trocar o reparo; trocar a
   válvula se houver desgaste.
6. Válvula de retenção com fuga: trocar a vedação ou a válvula.
7. Válvula de segurança aliviando antes da hora: substituir por uma nova da
   mesma especificação.

**Teste após o reparo.** Pressurizar; aplicar água e sabão nos pontos
reparados (sem bolhas); repetir o teste de queda de pressão de 10 minutos
(queda igual à de um trem sem falha); observar ciclos de carga e alívio
regulares, corrente normal e TP3 entre 8 e 10 bar, sem alarme LPS.

**Ressalvas.** Nunca desmontar com o circuito pressurizado; não usar fita
veda-rosca em vedação por anel ou face plana; não reutilizar anéis e juntas
removidos; válvula de segurança é trocada, nunca ajustada em campo; se a
queda de pressão continuar, procurar outro ponto de fuga ou falha no secador
de ar antes de liberar o trem.

## Vazamento de óleo no compressor — guia completo de reparo

Vazamento de óleo no compressor de ar industrial: perda do óleo lubrificante
do compressor. Guia de reparo do vazamento de óleo.

**Diagnóstico.** Indicador Oil_level ativo (nível abaixo do esperado);
Oil_temperature acima do habitual em carga; manchas ou gotejamento de óleo
sob o compressor; óleo arrastado para o ar comprimido (névoa ou óleo no dreno
dos reservatórios e no secador); compressor mais tempo em carga.

**Causas prováveis.** Retentor do eixo gasto ou ressecado; junta do cárter ou
da tampa danificada; bujão de dreno frouxo ou sem arruela; filtro de óleo mal
apertado; linhas de óleo trincadas ou frouxas; separador ar/óleo saturado;
nível de óleo acima do máximo forçando as vedações.

**Ferramentas.** Chaves combinadas, chave de torque e chave de filtro;
bandeja de contenção; panos absorventes; recipiente para óleo usado;
desengraxante; lanterna; cadeado e etiqueta de bloqueio elétrico.

**Peças e materiais.** Óleo do tipo e da viscosidade especificados pelo
fabricante; filtro de óleo novo; arruela nova do bujão de dreno; retentor do
eixo; juntas do cárter e da tampa do modelo do compressor; elemento separador
ar/óleo novo, se estiver saturado.

**Reparo passo a passo.**
1. Desligar o compressor, isolar a energia com cadeado e etiqueta,
   despressurizar e esperar o óleo esfriar.
2. Limpar o compressor com desengraxante para achar a origem do óleo.
3. Drenar o óleo pelo bujão para a bandeja de contenção.
4. Trocar a peça com vazamento: retentor do eixo, junta do cárter ou da
   tampa, arruela do bujão ou linha de óleo.
5. Trocar o filtro de óleo, lubrificando o anel de vedação do filtro novo.
6. Se havia óleo no ar comprimido, trocar o elemento separador ar/óleo.
7. Completar com óleo novo até a marca do visor ou da vareta.

**Teste após o reparo.** Funcionar o compressor e conferir que não há
gotejamento; desligar, esperar o óleo assentar e conferir o nível no visor;
acompanhar Oil_temperature em carga (deve voltar ao habitual); Oil_level não
deve ficar ativo; drenar os reservatórios sem sair óleo junto com a água.

**Ressalvas.** Óleo quente causa queimaduras: trabalhar com o compressor
frio; não misturar óleos diferentes do especificado; não encher acima do
máximo; descartar óleo usado e panos de forma ambientalmente adequada; se
havia óleo no secador de ar, trocar também o dessecante.

## Falha no secador de ar — guia completo de reparo

Falha no secador de ar (torres) do compressor de ar industrial: o secador de
torre dupla não retira a umidade do ar comprimido. Guia de reparo da falha no
secador de ar.

**Diagnóstico.** O sinal Towers deixa de alternar ou alterna fora do padrão;
perda de ar contínua pela purga do secador (TP3 cai com o compressor em
alívio, parecido com um vazamento de ar); água no dreno dos reservatórios e
na linha de ar; névoa de óleo na saída de purga.

**Causas prováveis.** Válvula de comando das torres presa por sujeira ou
umidade; válvula de purga que não fecha; dessecante saturado ou contaminado
por óleo; filtro de entrada do secador entupido; bobina ou temporizador da
válvula do secador com defeito.

**Ferramentas.** Chaves combinadas e chave de torque; multímetro para testar
a bobina da válvula do secador; solução de água e sabão; cadeado e etiqueta
de bloqueio elétrico.

**Peças e materiais.** Reparo (vedações) ou válvula de comando das torres
nova; válvula de purga nova ou kit de reparo; carga de dessecante nova do tipo
especificado pelo fabricante; elemento do filtro de entrada do secador.

**Reparo passo a passo.**
1. Desligar o compressor, isolar a energia com cadeado e etiqueta e
   despressurizar até 0 bar.
2. Testar a bobina da válvula do secador com o multímetro; trocar se estiver
   aberta ou em curto.
3. Desmontar e limpar a válvula de comando das torres; trocar o reparo ou a
   válvula se houver desgaste.
4. Trocar a válvula de purga que não fecha.
5. Dessecante saturado ou com óleo: esvaziar as torres e colocar carga nova.
6. Trocar o elemento do filtro de entrada do secador.
7. Remontar com vedações novas.

**Teste após o reparo.** Pressurizar e acompanhar o sinal Towers, que deve
alternar em ciclo regular; conferir que a purga só abre na regeneração e
fecha em seguida; repetir o teste de queda de pressão, sem perda contínua
pela purga; drenar os reservatórios depois de alguns ciclos, sem água.

**Ressalvas.** Dessecante contaminado por óleo indica vazamento de óleo no
compressor, que deve ser corrigido antes; não reutilizar dessecante; usar só
o dessecante especificado pelo fabricante.

## Superaquecimento do compressor de ar — guia completo de reparo

Superaquecimento do compressor de ar industrial: temperatura do óleo
(Oil_temperature) acima do habitual. Guia de reparo do superaquecimento.

**Diagnóstico.** Oil_temperature subindo além do habitual em carga; o
compressor pode desarmar por temperatura; cheiro de óleo queimado.

**Causas prováveis.** Nível de óleo baixo; óleo degradado; radiador ou aletas
de resfriamento sujos; ventilador com defeito; filtro de ar de admissão
entupido; compressor em carga contínua por causa de um vazamento de ar.

**Ferramentas.** Chaves combinadas; escova e ar comprimido para limpeza das
aletas; termômetro infravermelho; cadeado e etiqueta de bloqueio elétrico.

**Peças e materiais.** Óleo e filtro de óleo especificados pelo fabricante;
filtro de ar de admissão novo; ventilador de reposição, se defeituoso.

**Reparo passo a passo.**
1. Desligar, isolar a energia com cadeado e etiqueta e esperar esfriar.
2. Conferir nível e estado do óleo; trocar óleo e filtro se estiver degradado.
3. Limpar radiador e aletas de resfriamento.
4. Verificar o ventilador; trocar se não girar livremente.
5. Trocar o filtro de ar de admissão entupido.
6. Se o compressor fica muito tempo em carga, investigar vazamento de ar.

**Teste após o reparo.** Funcionar em carga e confirmar Oil_temperature
normal e estável; conferir ciclos de carga e alívio regulares.

**Ressalvas.** Não abrir o bocal de óleo com o compressor quente ou
pressurizado; não limpar aletas com o ventilador em movimento.

## Compressor de ar não liga ou corrente do motor anormal — guia de reparo

Compressor de ar industrial não liga, desarma ou tem corrente do motor
(Motor_current) fora do padrão. Guia de reparo do motor do compressor.

**Diagnóstico.** Referência de corrente: cerca de 0 A desligado, 4 A em
alívio, 7 A em carga e 9 A na partida. Corrente zero com comando ligado,
corrente alta contínua ou desarme repetido indicam falha.

**Causas prováveis.** Disjuntor desarmado; contator ou relé com defeito; cabo
ou conexão solta; enrolamento do motor danificado; compressor preso
mecanicamente; partida em carga por falha na válvula de admissão.

**Ferramentas.** Multímetro; alicate amperímetro; chaves; cadeado e etiqueta
de bloqueio elétrico.

**Peças e materiais.** Contator, relé, fusível ou cabo de reposição com a
mesma especificação.

**Reparo passo a passo.**
1. Isolar a energia com cadeado e etiqueta.
2. Conferir disjuntor e fusíveis.
3. Medir tensão e continuidade; reapertar conexões soltas.
4. Girar o compressor à mão: se estiver preso, encaminhar para revisão.
5. Trocar contator, relé ou cabo com defeito.

**Teste após o reparo.** Medir a corrente na partida, em carga e em alívio e
comparar com a referência.

**Ressalvas.** Trabalho elétrico só por pessoal habilitado; não rearmar o
disjuntor repetidamente sem achar a causa.

## Compressor de ar não entra em alívio ou não carrega — guia de reparo

Compressor de ar industrial que não entra em alívio (fica sempre em carga)
ou não carrega. Guia de reparo das válvulas de admissão e de saída.

**Diagnóstico.** TP3 acima de 10 bar ou abaixo de 8 bar com o compressor
ligado; sinais COMP (válvula de admissão) ou DV_eletric (válvula de saída)
não mudam de estado; comando MPG não aciona a carga.

**Causas prováveis.** Válvula de admissão ou de saída presa; bobina da
válvula queimada; fiação do comando MPG com defeito.

**Ferramentas.** Multímetro; chaves; cadeado e etiqueta de bloqueio elétrico.

**Peças e materiais.** Reparo das válvulas de admissão e de saída; bobina de
reposição.

**Reparo passo a passo.**
1. Isolar a energia e despressurizar até 0 bar.
2. Testar a bobina das válvulas com o multímetro; trocar se defeituosa.
3. Desmontar, limpar e trocar o reparo da válvula de admissão ou de saída.
4. Conferir o comando MPG e a fiação; trocar o componente com defeito.

**Teste após o reparo.** Ciclos de carga e alívio regulares e TP3 entre 8 e
10 bar.

**Ressalvas.** Compressor que não entra em alívio sobrecarrega o circuito:
retirar o trem de serviço até o reparo.

## Alarme LPS — pressão baixa (ação imediata)

Alarme LPS de pressão baixa: a pressão do circuito ficou abaixo de 7 bar e
falta ar para freios e portas.
- Comunicar o centro de controle e seguir o procedimento operacional para
  retirar o trem de serviço com segurança.
- Não religar o compressor repetidamente sem diagnóstico.
- Depois de imobilizado, investigar nesta ordem: vazamento de ar, falha no
  secador de ar, compressor que não carrega e falha elétrica do motor
  (ver os guias de reparo de cada falha).
O objetivo do monitoramento é detectar a falha antes de o LPS disparar.

## Segurança e encerramento de qualquer reparo no compressor de ar

Segurança antes de qualquer reparo no compressor de ar: retirar o trem de
serviço; desligar o compressor; isolar a energia do motor com cadeado e
etiqueta no disjuntor; despressurizar pelos drenos e confirmar 0 bar; esperar
o resfriamento; usar óculos, luvas e calçado de segurança; usar só peças e
óleo especificados pelo fabricante.
Ao encerrar, registrar na ordem de serviço: sintoma e sinais dos sensores
(TP3, Motor_current, Oil_level, Towers, LPS); causa e componente afetado;
peças, óleo e dessecante usados; resultado dos testes; responsável técnico e
data de liberação do trem.
