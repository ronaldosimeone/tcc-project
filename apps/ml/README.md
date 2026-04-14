# apps/ml – Machine Learning Service

Módulo de Machine Learning do TCC: ingestão de dados, EDA, treinamento e exportação de modelos ONNX para o compressor de ar do metrô (dataset MetroPT-3).

---

## Estrutura

```
apps/ml/
├── data/
│   ├── raw/            # ZIP e CSV originais (ignorados pelo git)
│   └── processed/      # Arquivos .parquet prontos para uso (ignorados pelo git)
├── notebooks/
│   └── 01_eda_metropt.ipynb
├── src/
│   └── ingest_metropt.py
├── tests/
│   └── test_ingest_metropt.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Pré-requisitos

Instale as dependências (preferencialmente em um ambiente virtual):

```bash
pip install -r requirements.txt
```

---

## Ingestão do Dataset (Sprint 1 – Task 3)

O script baixa o MetroPT-3 da UCI, valida o schema e salva um arquivo `.parquet` em `data/processed/`.

**O script é idempotente**: se o `.parquet` já existir, a execução é ignorada sem efeitos colaterais.

```bash
# A partir da raiz do monorepo
python -m apps.ml.src.ingest_metropt

# Ou a partir de apps/ml/
python src/ingest_metropt.py
```

Saída esperada:
```
[INFO] Data directories are ready: …/data/raw | …/data/processed
[INFO] Downloading MetroPT-3 dataset from https://…
[INFO] Download complete: … (XX.XX MB)
[INFO] Extracting CSV …
[INFO] Loaded NNNNNN rows × 16 columns.
[INFO] Schema validation passed – all 16 expected columns present.
[INFO] Preprocessing complete. Final shape: NNNNNN rows × 16 columns.
[INFO] Saved Parquet to …/data/processed/metropt3.parquet (XX.XX MB).
[INFO] Ingestion pipeline finished successfully.
```

---

## Análise Exploratória (EDA)

Após a ingestão, abra o notebook no JupyterLab:

```bash
jupyter lab notebooks/01_eda_metropt.ipynb
```

O notebook cobre:
- Estatísticas descritivas e verificação de nulos
- Distribuição de classes (labels de falha)
- Séries temporais dos sensores (TP2, TP3, H1, DV_pressure, Motor_current, Oil_temperature)
- Histogramas de distribuição individual
- Matriz de correlação com anotações
- Detecção de outliers via IQR

---

## Testes

```bash
# A partir da raiz do monorepo
pytest apps/ml/tests/ -v

# Com relatório de cobertura
pytest apps/ml/tests/ -v --cov=apps.ml.src --cov-report=term-missing
```

Os testes utilizam a fixture `tmp_path` do pytest – **nenhum arquivo é gravado em `data/`** durante a execução.

---

## Variáveis de Ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| _(nenhuma obrigatória)_ | – | Paths são resolvidos relativamente ao script |

---

## Notas de Arquitetura

- Modelos treinados devem ser exportados para **ONNX** antes de serem servidos pela API (requisito RNF de produção – `CLAUDE.md §4`).
- A pasta `data/` está no `.gitignore`; **nunca** commite os datasets.
