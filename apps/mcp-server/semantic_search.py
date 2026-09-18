"""
PredictIQ — Busca semântica sobre manuais indexados (RF-21 / RNF-45).

Consulta a MESMA collection (`maintenance_manuals`) preparada por
`index_manuals.py` (RF-20) — nenhuma indexação, modelo ou persistência nova é
criada aqui. `SemanticSearchService` só faz a leitura:

    query -> embedding (EMBEDDING_MODEL) -> candidatos via vector store
          -> cosine similarity (calculada aqui) -> filtro score > 0.6 -> top 5

RNF-62: o armazenamento vetorial deixou de ser o pacote `chromadb` (CVEs
High/Critical sem correção no seu servidor HTTP — ver README §17.1) e passou
a ser o `vector_store.py` local (sqlite3 + numpy). A lógica desta função
NÃO mudou: L2 para pré-selecionar candidatos, cosine para o score final.

Métrica de distância — auditada ANTES de implementar, e corrigida depois de
medir com dados reais
--------------------------------------------------------------------------
O `vector_store.py` ordena candidatos por distância L2 ao quadrado (squared
Euclidean) sobre os embeddings brutos do SentenceTransformer, que não são
normalizados — a mesma métrica default que o `chromadb` usava (collection sem
`hnsw:space` explícito). Confirmado empiricamente (distância de um vetor
consigo mesmo = `0.0`).

A primeira versão desta função convertia essa distância com
`similarity = 1 / (1 + distance)` (fórmula padrão para espaços L2, usada por
integrações como LangChain). **Validada contra dados reais desta task,
essa fórmula se mostrou inutilizável**: com o modelo multilíngue configurado,
as distâncias squared-L2 reais entre a query e os chunks indexados ficam na
faixa de ~15 a ~50 (vetores de norma alta, não normalizados) — o que produz
`similarity <= 0.07` mesmo para o chunk mais relevante de verdade. Com o
threshold fixo `score > 0.6` da RF-21, **nenhuma busca jamais retornaria
resultado**, não importa o quão relevante o conteúdo fosse.

Correção adotada, documentada em vez de escondida: calcular a similaridade
de cosseno DIRETAMENTE entre o embedding da query e o embedding de cada
candidato (`collection.query(..., include=["embeddings"])` devolve os
vetores brutos já persistidos por `index_manuals.py` — nenhum dado novo,
nenhuma reindexação). Cosine similarity é invariante à norma dos vetores
(por isso funciona igual não importa a magnitude que este modelo produz) e
fica naturalmente em `[-1, 1]` — `> 0.6` volta a ser um limiar com
significado real. Validado contra o corpus real desta task: consultas
que parafraseiam uma frase do manual chegam a `~0.65`–`~0.75`; uma consulta
deliberadamente irrelevante ("receita de bolo de chocolate") fica em `~0.0`
ou negativa. O limiar `0.6` é estrito de propósito (RF-21) — a maioria das
consultas genéricas devolve `[]`, o que é um resultado válido.

A pré-seleção por distância L2 (feita no `vector_store.py`) continua sendo
usada só para buscar a lista de CANDIDATOS de forma eficiente — o score final
que decide o filtro/ranking é sempre a cosine similarity calculada aqui.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger("predictiq.semantic_search")

# RF-21 — regras de negócio fixas da busca.
MAX_RESULTS = 5
SCORE_THRESHOLD = 0.6  # estrito: score == 0.6 é descartado (`score > 0.6`)

# Quantos candidatos pedir ao ChromaDB (por proximidade L2 aproximada) antes
# de recalcular o score real (cosine) e filtrar — maior que MAX_RESULTS de
# propósito, pra não perder resultado válido que apareça depois de
# candidatos ruins na pré-ordenação do índice ANN.
CANDIDATE_POOL = 20


def cosine_similarity(
    query_embedding: list[float], doc_embedding: list[float]
) -> float:
    """
    Converte um par de embeddings brutos em um score de similaridade em
    `[-1, 1]` — a conversão real usada pela RF-21 (ver docstring do módulo
    para o porquê de NÃO usar a distância L2 que o ChromaDB devolve
    diretamente).

        cosine_similarity(a, b) = (a . b) / (||a|| * ||b||)

    Invariante à norma dos vetores — funciona igual não importa a escala
    que o modelo de embeddings configurado produz.
    """
    a = np.asarray(query_embedding, dtype=float)
    b = np.asarray(doc_embedding, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0  # vetor nulo -> similaridade indefinida, tratada como "sem relação"
    return float(np.dot(a, b) / denom)


@dataclass
class SearchResult:
    """Um trecho recuperado — texto + metadados de origem (RF-20) + score.
    Nunca gera texto novo: `text` é sempre um chunk armazenado, verbatim."""

    text: str
    score: float
    metadata: dict[str, Any]


class SemanticSearchService:
    """Responsabilidade única: busca semântica sobre `maintenance_manuals`.

    Reutiliza a collection e o modelo de embeddings passados no construtor
    (ambos vêm de `index_manuals.get_collection`/`load_embedding_model` —
    mesma infraestrutura da indexação, RF-20). O modelo deve ser carregado
    UMA vez pelo chamador (ver `server._get_service`) e reaproveitado em
    todas as chamadas de `search` — RNF-45 depende disso.
    """

    def __init__(
        self,
        collection: Any,
        model: SentenceTransformer,
        max_results: int = MAX_RESULTS,
        score_threshold: float = SCORE_THRESHOLD,
        candidate_pool: int = CANDIDATE_POOL,
    ) -> None:
        self._collection = collection
        self._model = model
        self._max_results = max_results
        self._score_threshold = score_threshold
        self._candidate_pool = candidate_pool

    def search(self, query: str) -> list[SearchResult]:
        """
        Ordem obrigatória (RF-21): buscar candidatos -> calcular score real
        (cosine) -> filtrar `score > threshold` -> só então limitar a
        `max_results`. Nunca cortar para os 5 primeiros antes de filtrar —
        um candidato ruim entre os primeiros do ChromaDB não pode empurrar
        um candidato válido pra fora da resposta.
        """
        if not query or not query.strip():
            return []

        try:
            count = self._collection.count()
        except Exception:
            log.exception("Falha ao consultar tamanho da collection")
            return []

        if count == 0:
            return []

        query_embedding = self._model.encode(
            [query], show_progress_bar=False, convert_to_numpy=True
        )[0].tolist()

        n_candidates = min(self._candidate_pool, count)
        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_candidates,
            include=["documents", "metadatas", "embeddings"],
        )

        documents = raw.get("documents") or [[]]
        metadatas = raw.get("metadatas") or [[]]
        embeddings = raw.get("embeddings") or [[]]

        candidates = zip(documents[0], metadatas[0], embeddings[0])

        accepted: list[SearchResult] = []
        for document, metadata, doc_embedding in candidates:
            score = cosine_similarity(query_embedding, doc_embedding)
            if score <= self._score_threshold:
                continue
            accepted.append(
                SearchResult(text=document, score=score, metadata=dict(metadata))
            )

        accepted.sort(key=lambda result: result.score, reverse=True)
        return accepted[: self._max_results]
