# Curadoria do corpus RAG — WAP-5 Scheduled Maintenance Manual

Insumo do RF-37 (ablação do RAG) e do RN-04 (limiar de fundamentação).
Script: [`benchmark_corpus_curation.py`](benchmark_corpus_curation.py) · dados
brutos: [`corpus_curation_benchmark.json`](corpus_curation_benchmark.json).

## Documento avaliado

*Indian Railways — Scheduled Maintenance Manual, WAP-5 Locomotive, Vol. B*
(Bombardier/ABB, 3EHQ600371), 224 páginas, scan com OCR (2010).

| Aspecto | Achado |
|---|---|
| Tipo | Manutenção **programada** (tarefas periódicas), não *fault finding* (sintoma → causa → reparo) |
| Seção relevante | 6.6 *Air Supply and Pneumatic System*, p. 170–177 (tarefas 601–604: compressor principal, secador de ar de torre dupla, reservatórios/drenos, compressor auxiliar) |
| Remissões | 219 de 529 chunks (41 %) contêm "Refer to Suppliers Documentation Volume F…": o procedimento real está em volumes não disponíveis |
| Qualidade do OCR | 5 de 529 chunks com mais de 15 % de tokens corrompidos (pior: p. 215, ficha de óleo) |
| Domínio | Compressor alternativo Metcalfe 2A320D de locomotiva elétrica, não o compressor da APU do MetroPT-3. Componentes análogos: secador de torres (sensor `Towers`), reservatórios (`Reservoirs`), óleo (`Oil_level`/`Oil_temperature`), válvulas de dreno/segurança |
| Licença | Documento RDSO/Bombardier: fica só em `data/manuals/` (no `.gitignore`), citado como fonte; nunca versionado |

## Curadoria aplicada (sidecar `<pdf>.meta.json`)

O indexador passou a aceitar um perfil por manual, sem alterar o PDF:

```json
{
  "pages": [170, 177],
  "strip_cross_references": true,
  "min_chunk_chars": 120,
  "metadata": {"domain": "locomotive", "doc_type": "scheduled_maintenance",
               "equipment": "air_supply_pneumatic", "source_title": "..."}
}
```

- `pages`: indexa só a faixa, mantendo o número de página original nos metadados.
- `strip_cross_references`: remove as frases "Refer to Volume…/Suppliers Documentation…" e mantém a descrição acionável da tarefa.
- `min_chunk_chars`: descarta os chunks que ficam só com ruído depois da limpeza.
- `metadata`: campos escalares extras em cada chunk. Nunca sobrescrevem os campos do núcleo.
- O conteúdo do sidecar entra no hash do arquivo, então mudar o perfil reindexa.

Testes: `tests/test_index_manuals.py` §10 (11 casos novos, 27/27 passando na imagem do mcp-server).

## Resultado: busca semântica real

Modelo `paraphrase-multilingual-MiniLM-L12-v2`, chunk 1000/150, 8 consultas de
falha do compressor (5 em PT, 3 em EN), vector store temporário por corpus.

| Corpus | Chunks | Consultas com ≥ 1 resultado (`score > 0.6`) | Top-1 relevante (sem limiar)¹ | Top-1 médio (sem limiar) |
|---|---:|---:|---:|---:|
| A — manuais atuais | 9 | **0 / 8** | 0 / 8 | 0,485 |
| B — A + WAP-5 inteiro | 524 | 1 / 8 (os 3 resultados vêm de fora da seção 6.6) | 0 / 8² | 0,580 |
| C — A + WAP-5 curado | 25 | 1 / 8 (relevante) | **4 / 8** | 0,541 |

¹ Relevante = chunk do WAP-5 na seção 6.6 ou do manual de compressor. Classificação manual a partir de `top5_no_threshold` no JSON.
² B devolve no top-1 bomba de óleo do transformador, ficha de extintor, ficha de lubrificante (OCR ruim) e válvulas do módulo de freio.

## Conclusões

1. **O corpus atual não atende o RF-22 com consultas reais de falha.** Nas 8 consultas o limiar `score > 0.6` corta tudo, então o assistente responde sem nenhum contexto de manual. Sem limiar, o top-1 do "vazamento de ar no compressor" é o manual da *bomba centrífuga*. O manual `compressor-cx500` nunca aparece em primeiro.
2. **Indexar o WAP-5 inteiro piora a qualidade** (corpus B): o score sobe porque há mais texto parecido, mas o top-1 vira ruído de outros subsistemas. É o caso negativo do RF-37.
3. **O WAP-5 curado acerta o top-1 em 4 de 8 consultas**, todas sobre secador, reservatório e válvulas. As 4 falhas são consultas em português sobre óleo e corrente do motor, o que mostra um gap cross-lingual (consulta PT × manual EN) e a ausência de conteúdo de diagnóstico de óleo no manual.
4. **O limiar fixo 0,6 está descalibrado para este modelo**: chunks corretos pontuam entre 0,53 e 0,58. O τ do RN-04 deve ser calibrado no benchmark anotado do RF-36 (curva precisão × cobertura), não fixado a priori.

## Próximos passos

- Conseguir um manual de **fault finding** (ex.: Vol. C desta mesma série, ou um guia de troubleshooting de compressor/secador de fabricante) e repetir o benchmark.
- Calibrar τ e testar consulta bilíngue ou tradução da consulta (RF-37).
- OCR residual: remissões corrompidas pelo OCR (ex.: `~ef@o Suppliers`) escapam do regex. São poucas na seção 6.6.
