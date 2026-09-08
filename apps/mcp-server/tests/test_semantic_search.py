"""
Testes de `SemanticSearchService` (RF-21 / RNF-45).

Nenhum teste baixa modelo real: um `SentenceTransformer` fake, controlado,
mapeia qualquer query para um embedding conhecido — permite calcular a
cosine similarity exata contra os documentos inseridos e testar a fronteira
do threshold (`score > 0.6`) com valores derivados da fórmula REAL de
`cosine_similarity`, não valores chutados. O ChromaDB usa sempre diretório
temporário (`tmp_path`), nunca `data/chroma` real — mesma convenção de
`test_index_manuals.py`.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import index_manuals as im  # noqa: E402
from semantic_search import SemanticSearchService, cosine_similarity  # noqa: E402


# ---------------------------------------------------------------------------
# Fake SentenceTransformer — sem download, sem rede.
# ---------------------------------------------------------------------------


class _FixedVectorModel:
    """Ignora o texto da query e sempre devolve `vector` — dá controle total
    sobre a similaridade até os documentos inseridos no teste. Registra as
    chamadas (`encode_calls`) pra provar que o modelo é reaproveitado, não
    recarregado por busca."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.encode_calls: list[str] = []

    def encode(
        self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
    ):
        self.encode_calls.extend(texts)
        return np.array([self.vector for _ in texts])


# ---------------------------------------------------------------------------
# Helpers de fixture — inserir chunks com embeddings/metadados conhecidos.
# ---------------------------------------------------------------------------


def _metadata(file_name: str, page: int = 1, chunk_index: int = 0) -> dict[str, Any]:
    return {
        "source": file_name,
        "file_name": file_name,
        "file_hash": "a" * 64,
        "page": page,
        "chunk_index": chunk_index,
        "embedding_model": "fake-model-for-tests",
    }


def _add_chunk(
    collection: Any, doc_id: str, text: str, vector: list[float], **metadata_kwargs: Any
) -> None:
    collection.add(
        ids=[doc_id],
        embeddings=[vector],
        documents=[text],
        metadatas=[_metadata(**metadata_kwargs)],
    )


# Query fixa em (1, 0) — vetor unitário no eixo X. Um documento em
# (cos(theta), sin(theta)) tem cosine_similarity == cos(theta) exatamente
# contra essa query, ENTÃO para atingir uma similaridade alvo `s`, o
# documento é colocado em (s, sqrt(1 - s**2)): mesma fórmula real usada em
# produção (`cosine_similarity`), não um valor mágico chutado. Funciona pra
# qualquer norma do vetor de documento (cosine é invariante à escala) —
# aqui usamos vetores unitários só por simplicidade.
QUERY_VECTOR = [1.0, 0.0]


def _vector_for_similarity(similarity: float) -> list[float]:
    return [similarity, math.sqrt(1.0 - similarity**2)]


# ---------------------------------------------------------------------------
# cosine_similarity — a conversão em si
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical_vectors_is_perfect_score() -> None:
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_minus_one() -> None:
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_invariant_to_vector_scale() -> None:
    """Empiricamente é exatamente por isso que a métrica default do
    ChromaDB (squared L2, sensível à norma) foi trocada por cosine — ver
    docstring de semantic_search.py."""
    sim_small = cosine_similarity([1.0, 0.0], [2.0, 1.0])
    sim_large = cosine_similarity(
        [1.0, 0.0], [200.0, 100.0]
    )  # mesma direção, norma 100x maior
    assert sim_small == pytest.approx(sim_large)


def test_cosine_similarity_zero_vector_returns_zero_not_exception() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


# ---------------------------------------------------------------------------
# 1. Query válida
# ---------------------------------------------------------------------------


def test_valid_query_returns_results_when_relevant_docs_exist(tmp_path: Path) -> None:
    collection = im.get_collection(tmp_path / "chroma")
    _add_chunk(
        collection,
        "c1",
        "Trocar oleo do compressor a cada 500 horas.",
        QUERY_VECTOR,
        file_name="manual.pdf",
    )

    service = SemanticSearchService(
        collection=collection, model=_FixedVectorModel(QUERY_VECTOR)
    )
    results = service.search("troca de oleo")

    assert len(results) == 1
    assert results[0].text == "Trocar oleo do compressor a cada 500 horas."
    assert results[0].score == pytest.approx(1.0)  # mesmo vetor -> score máximo


# ---------------------------------------------------------------------------
# 2. Máximo de 5
# ---------------------------------------------------------------------------


def test_never_returns_more_than_five_results(tmp_path: Path) -> None:
    collection = im.get_collection(tmp_path / "chroma")
    for i in range(8):
        # todas com similaridade bem acima do threshold (0.61..0.68)
        vector = _vector_for_similarity(0.61 + i * 0.01)
        _add_chunk(
            collection,
            f"c{i}",
            f"chunk numero {i}",
            vector,
            file_name="manual.pdf",
            chunk_index=i,
        )

    service = SemanticSearchService(
        collection=collection, model=_FixedVectorModel(QUERY_VECTOR)
    )
    results = service.search("consulta qualquer")

    assert len(results) == 5


# ---------------------------------------------------------------------------
# 3. Threshold — fronteira exata, derivada da fórmula real
# ---------------------------------------------------------------------------


class _FakeCollection:
    """Só pra este teste de fronteira exata: devolve os embeddings dos
    candidatos byte-a-byte como foram passados, sem round-trip por um
    ChromaDB real. Achado real desta task: o ChromaDB armazena embeddings
    como float32 internamente — um valor de similaridade construído para
    dar EXATAMENTE `0.60` em float64 pode virar `0.6000000...1` ou
    `0.59999...9` depois do round-trip float32, fazendo o teste de fronteira
    "flakar" por causa da precisão de armazenamento, não da lógica de
    filtro. Os outros testes desta suíte já cobrem o comportamento com o
    ChromaDB real (`im.get_collection`); este cobre só a lógica exata de
    `SemanticSearchService.search` (candidatos -> score -> filtro)."""

    def __init__(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        self._documents = documents
        self._metadatas = metadatas
        self._embeddings = embeddings

    def count(self) -> int:
        return len(self._documents)

    def query(
        self, query_embeddings: Any, n_results: int, include: Any
    ) -> dict[str, Any]:
        return {
            "documents": [self._documents[:n_results]],
            "metadatas": [self._metadatas[:n_results]],
            "embeddings": [self._embeddings[:n_results]],
        }


def test_threshold_boundary_strictly_greater_than_point_six() -> None:
    collection = _FakeCollection(
        documents=["abaixo do limite", "exatamente no limite", "acima do limite"],
        metadatas=[_metadata("a.pdf"), _metadata("b.pdf"), _metadata("c.pdf")],
        embeddings=[
            _vector_for_similarity(0.59),
            _vector_for_similarity(0.60),
            _vector_for_similarity(0.6001),
        ],
    )

    service = SemanticSearchService(
        collection=collection, model=_FixedVectorModel(QUERY_VECTOR)
    )
    results = service.search("consulta qualquer")

    texts = {r.text for r in results}
    assert "abaixo do limite" not in texts  # score 0.59 -> rejeitado
    assert (
        "exatamente no limite" not in texts
    )  # score == 0.60 -> rejeitado (regra é > 0.6, não >=)
    assert "acima do limite" in texts  # score 0.6001 -> aceito
    assert len(results) == 1
    assert results[0].score > 0.6


# ---------------------------------------------------------------------------
# 4. Ordenação
# ---------------------------------------------------------------------------


def test_results_ordered_most_relevant_first(tmp_path: Path) -> None:
    collection = im.get_collection(tmp_path / "chroma")
    # inseridos fora de ordem de propósito
    _add_chunk(
        collection,
        "mid",
        "relevancia media",
        _vector_for_similarity(0.75),
        file_name="a.pdf",
    )
    _add_chunk(
        collection,
        "high",
        "mais relevante",
        _vector_for_similarity(0.95),
        file_name="b.pdf",
    )
    _add_chunk(
        collection,
        "low",
        "menos relevante",
        _vector_for_similarity(0.65),
        file_name="c.pdf",
    )

    service = SemanticSearchService(
        collection=collection, model=_FixedVectorModel(QUERY_VECTOR)
    )
    results = service.search("consulta qualquer")

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].text == "mais relevante"
    assert results[-1].text == "menos relevante"


# ---------------------------------------------------------------------------
# 5. Metadados
# ---------------------------------------------------------------------------


def test_metadata_preserved_in_results(tmp_path: Path) -> None:
    collection = im.get_collection(tmp_path / "chroma")
    _add_chunk(
        collection,
        "c1",
        "conteudo do manual da bomba",
        QUERY_VECTOR,
        file_name="bomba.pdf",
        page=3,
        chunk_index=2,
    )

    service = SemanticSearchService(
        collection=collection, model=_FixedVectorModel(QUERY_VECTOR)
    )
    results = service.search("bomba")

    assert len(results) == 1
    metadata = results[0].metadata
    assert metadata["source"] == "bomba.pdf"
    assert metadata["file_name"] == "bomba.pdf"
    assert metadata["file_hash"] == "a" * 64
    assert metadata["page"] == 3
    assert metadata["chunk_index"] == 2


# ---------------------------------------------------------------------------
# 6. Query sem resultado
# ---------------------------------------------------------------------------


def test_no_relevant_chunk_returns_empty_list_not_exception(tmp_path: Path) -> None:
    collection = im.get_collection(tmp_path / "chroma")
    _add_chunk(
        collection,
        "c1",
        "totalmente irrelevante",
        _vector_for_similarity(0.1),
        file_name="a.pdf",
    )

    service = SemanticSearchService(
        collection=collection, model=_FixedVectorModel(QUERY_VECTOR)
    )
    results = service.search("consulta sem relacao nenhuma")

    assert results == []


def test_empty_collection_returns_empty_list_not_exception(tmp_path: Path) -> None:
    collection = im.get_collection(tmp_path / "chroma")  # nenhum chunk inserido

    service = SemanticSearchService(
        collection=collection, model=_FixedVectorModel(QUERY_VECTOR)
    )
    results = service.search("qualquer coisa")

    assert results == []


# ---------------------------------------------------------------------------
# 7. Modelo/query — não substitui a query internamente
# ---------------------------------------------------------------------------


def test_search_encodes_the_actual_query_text(tmp_path: Path) -> None:
    collection = im.get_collection(tmp_path / "chroma")
    _add_chunk(collection, "c1", "chunk", QUERY_VECTOR, file_name="a.pdf")
    model = _FixedVectorModel(QUERY_VECTOR)

    service = SemanticSearchService(collection=collection, model=model)
    service.search("vazamento de oleo no compressor")

    assert model.encode_calls == ["vazamento de oleo no compressor"]


# ---------------------------------------------------------------------------
# 8. Modelo carregado uma vez — encode chamado uma vez POR busca, nunca por
# candidato/chunk (os embeddings dos chunks já estão no ChromaDB).
# ---------------------------------------------------------------------------


def test_encode_called_once_per_search_not_per_candidate(tmp_path: Path) -> None:
    collection = im.get_collection(tmp_path / "chroma")
    for i in range(6):
        _add_chunk(
            collection,
            f"c{i}",
            f"chunk {i}",
            _vector_for_similarity(0.7),
            file_name="a.pdf",
            chunk_index=i,
        )
    model = _FixedVectorModel(QUERY_VECTOR)

    service = SemanticSearchService(collection=collection, model=model)
    service.search("consulta 1")
    service.search("consulta 2")
    service.search("consulta 3")

    assert (
        len(model.encode_calls) == 3
    )  # uma chamada de encode por busca, não por chunk (6) nem por candidato


# ---------------------------------------------------------------------------
# 9. ChromaDB temporário
# ---------------------------------------------------------------------------


def test_uses_temp_chroma_dir_not_dev_database(tmp_path: Path) -> None:
    chroma_dir = tmp_path / "chroma_isolado"
    collection = im.get_collection(chroma_dir)
    assert collection.count() == 0
    assert chroma_dir.exists()
    assert chroma_dir != im.CHROMA_DB_PATH


# ---------------------------------------------------------------------------
# Validação de entrada — string vazia / só espaço / extremamente grande
# ---------------------------------------------------------------------------


def test_empty_query_returns_no_results_without_calling_model(tmp_path: Path) -> None:
    collection = im.get_collection(tmp_path / "chroma")
    _add_chunk(collection, "c1", "chunk", QUERY_VECTOR, file_name="a.pdf")
    model = _FixedVectorModel(QUERY_VECTOR)

    service = SemanticSearchService(collection=collection, model=model)
    assert service.search("") == []
    assert model.encode_calls == []  # nunca gera embedding de uma query vazia


def test_whitespace_only_query_returns_no_results(tmp_path: Path) -> None:
    collection = im.get_collection(tmp_path / "chroma")
    _add_chunk(collection, "c1", "chunk", QUERY_VECTOR, file_name="a.pdf")

    service = SemanticSearchService(
        collection=collection, model=_FixedVectorModel(QUERY_VECTOR)
    )
    assert service.search("    \n\t  ") == []


def test_very_large_query_does_not_crash(tmp_path: Path) -> None:
    collection = im.get_collection(tmp_path / "chroma")
    _add_chunk(collection, "c1", "chunk", QUERY_VECTOR, file_name="a.pdf")

    service = SemanticSearchService(
        collection=collection, model=_FixedVectorModel(QUERY_VECTOR)
    )
    huge_query = "palavra " * 50_000  # ~400 KB de texto

    results = service.search(huge_query)  # não deve lançar exceção

    assert isinstance(results, list)
