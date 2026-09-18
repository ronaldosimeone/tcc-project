"""
PredictIQ — Armazenamento vetorial local embutido (RF-20 / RF-21 / RNF-62).

Substitui a dependência `chromadb` por um store mínimo sobre `sqlite3`
(stdlib) + `numpy` (já presente via `sentence-transformers`). Motivo — RNF-62,
segurança de produção:

    Toda versão publicada do `chromadb` (>= 0.4.17, incl. a mais recente
    1.5.9) carrega CVEs High/Critical SEM CORREÇÃO:
      • CVE-2026-45830 (GHSA-2wm9-hf6c-p5cr) — High 8.8
      • CVE-2026-45831 (GHSA-xph7-9rjv-w5fr) — High 8.8
      • CVE-2026-45833 (GHSA-36p7-vc44-83pf) — Critical
      • CVE-2026-45829 (GHSA-f4j7-r4q5-qw2c) — Critical (só 1.x)
    `patched: None` em todas — não existe versão do chromadb que passe no
    `pip-audit`. Os quatro estão no COMPONENTE SERVIDOR HTTP do chromadb
    (`chroma run` / FastAPI / RBAC) — que este serviço nunca executa (usa
    só o cliente embutido) —, mas o RNF-62 exige eliminar o High/Critical
    da árvore de dependências de produção, não apenas argumentar
    alcançabilidade. Ver README §17.

Contrato preservado (o resto do mcp-server não muda)
---------------------------------------------------
`index_manuals.py` e `semantic_search.py` continuam chamando exatamente os
mesmos métodos que chamavam no `chromadb`:

    client = PersistentClient(path=...)
    col = client.get_or_create_collection(name=..., metadata={...})
    col.add(ids=, embeddings=, documents=, metadatas=)
    col.upsert(ids=, embeddings=, documents=, metadatas=)
    col.count()
    col.query(query_embeddings=[v], n_results=n,
              include=["documents", "metadatas", "embeddings"])
    col.get(where={"file_name": x}, limit=1, include=["metadatas"])
    col.get(where={"file_name": x}, include=[])
    col.get(include=["metadatas"])
    col.delete(ids=[...])

Mesma dimensão de embedding (384, `paraphrase-multilingual-MiniLM-L12-v2`),
mesmos metadados (RF-20), mesma semântica de distância L2 para pré-seleção
de candidatos — o score final continua sendo a cosine similarity calculada
em `semantic_search.py` a partir dos embeddings brutos (invariante à norma).
Os embeddings são persistidos em `float32`, igual ao chromadb, para que os
scores medidos não mudem.

O corpus real (3 manuais → algumas centenas de chunks) torna a busca por
força bruta em `numpy` (`O(n_chunks * dim)`) instantânea — o índice ANN do
chromadb não trazia benefício nessa escala.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

_DB_FILENAME = "vector_store.sqlite3"
_COLUMNS = ("id", "document", "metadata", "embedding", "dim")


def _f32_bytes(vector: Sequence[float]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _f32_from_bytes(blob: bytes, dim: int) -> list[float]:
    return np.frombuffer(blob, dtype=np.float32, count=dim).astype(float).tolist()


class _Collection:
    """Uma collection nomeada dentro de um mesmo arquivo sqlite.

    Cada operação abre e fecha sua própria conexão sqlite — o serviço MCP
    guarda a collection como singleton de processo (RNF-45) e pode chamar
    `query` de forma concorrente (transporte async); conexões de vida curta
    evitam qualquer questão de thread-safety sem custo relevante nesta
    escala.
    """

    def __init__(self, db_path: Path, name: str) -> None:
        self._db_path = db_path
        self._name = name
        self._write_lock = threading.Lock()

    # -- infra ------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                collection TEXT NOT NULL,
                id         TEXT NOT NULL,
                document   TEXT,
                metadata   TEXT,
                embedding  BLOB NOT NULL,
                dim        INTEGER NOT NULL,
                PRIMARY KEY (collection, id)
            )
            """
        )
        return conn

    @staticmethod
    def _validate_lengths(
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str] | None,
        metadatas: Sequence[dict[str, Any]] | None,
    ) -> None:
        n = len(ids)
        if len(embeddings) != n:
            raise ValueError("ids e embeddings têm tamanhos diferentes")
        if documents is not None and len(documents) != n:
            raise ValueError("ids e documents têm tamanhos diferentes")
        if metadatas is not None and len(metadatas) != n:
            raise ValueError("ids e metadatas têm tamanhos diferentes")

    def _write(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str] | None,
        metadatas: Sequence[dict[str, Any]] | None,
    ) -> None:
        self._validate_lengths(ids, embeddings, documents, metadatas)
        rows = []
        for index, chunk_id in enumerate(ids):
            vector = np.asarray(embeddings[index], dtype=np.float32)
            if vector.ndim != 1:
                raise ValueError("cada embedding deve ser um vetor 1-D")
            document = documents[index] if documents is not None else None
            metadata = metadatas[index] if metadatas is not None else None
            rows.append(
                (
                    self._name,
                    str(chunk_id),
                    document,
                    (
                        json.dumps(metadata, ensure_ascii=False)
                        if metadata is not None
                        else None
                    ),
                    vector.tobytes(),
                    int(vector.shape[0]),
                )
            )
        with self._write_lock, self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO chunks "
                "(collection, id, document, metadata, embedding, dim) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )

    # -- API pública (mesma assinatura do chromadb) ---------------------
    def add(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str] | None = None,
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self._write(ids, embeddings, documents, metadatas)

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str] | None = None,
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self._write(ids, embeddings, documents, metadatas)

    def count(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE collection = ?", (self._name,)
            )
            return int(cursor.fetchone()[0])

    def delete(
        self,
        ids: Sequence[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> None:
        clauses = ["collection = ?"]
        params: list[Any] = [self._name]
        if ids is not None:
            placeholders = ",".join("?" for _ in ids)
            clauses.append(f"id IN ({placeholders})")
            params.extend(str(i) for i in ids)
        for key, value in (where or {}).items():
            clauses.append(f"json_extract(metadata, '$.{key}') = ?")
            params.append(value)
        with self._write_lock, self._connect() as conn:
            conn.execute(f"DELETE FROM chunks WHERE {' AND '.join(clauses)}", params)

    def get(
        self,
        ids: Sequence[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        include: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Leitura filtrada — devolve listas PLANAS (`{"ids": [...],
        "metadatas": [...], ...}`), igual ao `Collection.get` do chromadb.
        `include` default do chromadb é `["metadatas", "documents"]`;
        `include=[]` devolve só os ids."""
        want = _resolve_include(include, default=("metadatas", "documents"))

        clauses = ["collection = ?"]
        params: list[Any] = [self._name]
        if ids is not None:
            placeholders = ",".join("?" for _ in ids)
            clauses.append(f"id IN ({placeholders})")
            params.extend(str(i) for i in ids)
        for key, value in (where or {}).items():
            clauses.append(f"json_extract(metadata, '$.{key}') = ?")
            params.append(value)

        sql = (
            "SELECT id, document, metadata, embedding, dim FROM chunks "
            f"WHERE {' AND '.join(clauses)} ORDER BY rowid"
        )
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
            if offset is not None:
                sql += f" OFFSET {int(offset)}"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        result: dict[str, Any] = {"ids": [row[0] for row in rows]}
        if "documents" in want:
            result["documents"] = [row[1] for row in rows]
        if "metadatas" in want:
            result["metadatas"] = [
                json.loads(row[2]) if row[2] is not None else None for row in rows
            ]
        if "embeddings" in want:
            result["embeddings"] = [_f32_from_bytes(row[3], row[4]) for row in rows]
        return result

    def query(
        self,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        include: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Vizinhos mais próximos por distância L2 ao quadrado (mesmo espaço
        default do chromadb sem `hnsw:space`). Devolve listas ANINHADAS (uma
        por embedding de consulta), igual ao `Collection.query` do chromadb:
        `{"ids": [[...]], "documents": [[...]], ...}`."""
        want = _resolve_include(
            include, default=("metadatas", "documents", "distances")
        )

        clauses = ["collection = ?"]
        params: list[Any] = [self._name]
        for key, value in (where or {}).items():
            clauses.append(f"json_extract(metadata, '$.{key}') = ?")
            params.append(value)

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, document, metadata, embedding, dim FROM chunks "
                f"WHERE {' AND '.join(clauses)} ORDER BY rowid",
                params,
            ).fetchall()

        queries = np.asarray(query_embeddings, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[np.newaxis, :]

        ids_out: list[list[str]] = []
        docs_out: list[list[str | None]] = []
        metas_out: list[list[dict[str, Any] | None]] = []
        embs_out: list[list[list[float]]] = []
        dists_out: list[list[float]] = []

        if rows:
            dim = rows[0][4]
            matrix = np.frombuffer(
                b"".join(row[3] for row in rows), dtype=np.float32
            ).reshape(len(rows), dim)
        else:
            matrix = np.empty((0, queries.shape[1]), dtype=np.float32)

        top_k = max(0, min(int(n_results), len(rows)))

        for query_vector in queries:
            if top_k == 0:
                ids_out.append([])
                docs_out.append([])
                metas_out.append([])
                embs_out.append([])
                dists_out.append([])
                continue
            if query_vector.shape[0] != matrix.shape[1]:
                raise ValueError(
                    "dimensão do embedding de consulta "
                    f"({query_vector.shape[0]}) difere da dos chunks "
                    f"({matrix.shape[1]})"
                )
            distances = np.sum((matrix - query_vector) ** 2, axis=1)
            order = np.argsort(distances, kind="stable")[:top_k]

            ids_out.append([rows[i][0] for i in order])
            docs_out.append([rows[i][1] for i in order])
            metas_out.append(
                [
                    json.loads(rows[i][2]) if rows[i][2] is not None else None
                    for i in order
                ]
            )
            embs_out.append([_f32_from_bytes(rows[i][3], rows[i][4]) for i in order])
            dists_out.append([float(distances[i]) for i in order])

        result: dict[str, Any] = {"ids": ids_out}
        if "documents" in want:
            result["documents"] = docs_out
        if "metadatas" in want:
            result["metadatas"] = metas_out
        if "embeddings" in want:
            result["embeddings"] = embs_out
        if "distances" in want:
            result["distances"] = dists_out
        return result


def _resolve_include(
    include: Iterable[str] | None, default: tuple[str, ...]
) -> set[str]:
    if include is None:
        return set(default)
    return {str(item) for item in include}


class PersistentClient:
    """Cliente persistente — mesma interface mínima de
    `chromadb.PersistentClient` usada pelo mcp-server. Um arquivo sqlite
    (`vector_store.sqlite3`) por diretório `path`."""

    def __init__(self, path: str, **_ignored: Any) -> None:
        self._dir = Path(path)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._dir / _DB_FILENAME

    def get_or_create_collection(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
        **_ignored: Any,
    ) -> _Collection:
        return _Collection(self._db_path, name)

    # aliases — não usados hoje pelo mcp-server, mantidos por paridade
    def get_collection(self, name: str, **_ignored: Any) -> _Collection:
        return _Collection(self._db_path, name)

    def create_collection(
        self, name: str, metadata: dict[str, Any] | None = None, **_ignored: Any
    ) -> _Collection:
        return _Collection(self._db_path, name)

    def delete_collection(self, name: str) -> None:
        collection = _Collection(self._db_path, name)
        with collection._connect() as conn:  # noqa: SLF001 — mesmo módulo
            conn.execute("DELETE FROM chunks WHERE collection = ?", (name,))
