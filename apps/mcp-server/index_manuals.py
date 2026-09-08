"""
PredictIQ — Pipeline de ingestão de manuais de manutenção (RF-20 / RNF-44).

Lê PDFs de `MANUALS_DIR`, extrai texto (pypdf), divide em chunks com overlap,
gera embeddings localmente (sentence-transformers) e persiste tudo em um
ChromaDB local (`CHROMA_DB_PATH`), na collection `maintenance_manuals`.

Esta pipeline SÓ prepara a base de conhecimento. A ferramenta MCP
`search_maintenance_manual` (server.py, RF-19) continua stub nesta task —
ainda não consulta este ChromaDB. A próxima task implementa
query -> embedding -> ChromaDB -> resultados. Ver README "RF-20 — Ingestão
de manuais" para o diagrama completo.

Indexação incremental (RF-20)
------------------------------
Não existe arquivo de estado separado. A fonte de verdade é o próprio
ChromaDB: cada chunk carrega `file_hash` (SHA-256 do conteúdo do PDF) nos
metadados. Antes de reprocessar um arquivo, a pipeline consulta os chunks
já indexados para aquele `file_name`:

    hash novo == hash já indexado -> skip (idempotente, nenhum chunk criado)
    hash novo != hash já indexado -> remove os chunks antigos e reindexa
    arquivo nunca visto           -> indexa

Isso evita um segundo mecanismo de estado (ex.: JSON solto) que precisaria
ficar sincronizado com o ChromaDB — e que, se não estivesse em um volume
persistente, seria perdido a cada restart do container (o que a task
explicitamente pede para evitar). O ChromaDB já é o dado persistente único.

Segurança
---------
`MANUALS_DIR`/`CHROMA_DB_PATH` só vêm de variável de ambiente (configurado
pelo operador do serviço, nunca por entrada de usuário/rede em runtime — este
script não expõe endpoint algum). A varredura de arquivos usa `Path.rglob`
restrito à raiz resolvida de `MANUALS_DIR`; os metadados gravados usam apenas
o nome/caminho relativo do próprio arquivo (nunca um caminho vindo de fora),
então não há superfície para path traversal.

Uso
---
    python index_manuals.py                  # lê config do ambiente (.env)
    docker compose exec mcp-server python index_manuals.py
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("predictiq.index_manuals")

# ---------------------------------------------------------------------------
# Configuração — RNF-44. Tudo lido do ambiente uma única vez, no import do
# módulo (mesmo padrão de server.py — sem classe Settings/pydantic-settings
# nova: o mcp-server é standalone de propósito, RNF-43, e não compartilha
# config com o backend).
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent


def _resolve_dir(value: str) -> Path:
    """Resolve um valor de configuração de diretório relativo ao próprio
    módulo (não ao cwd do processo que o invoca) — assim `python
    index_manuals.py` funciona igual de dentro do container (WORKDIR /app)
    e localmente a partir de apps/mcp-server."""
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


# Diretório dos PDFs de origem. Vive dentro de apps/mcp-server (já montado
# inteiro como volume em docker-compose.yml::mcp-server -> ./apps/mcp-server:
# /app), então não precisa de uma linha de volume extra no compose — o
# host já enxerga este diretório diretamente.
MANUALS_DIR: Path = _resolve_dir(os.environ.get("MANUALS_DIR", "data/manuals"))

# ChromaDB persistente (PersistentClient) — mesma lógica de volume acima.
CHROMA_DB_PATH: Path = _resolve_dir(os.environ.get("CHROMA_DB_PATH", "data/chroma"))

COLLECTION_NAME = "maintenance_manuals"

# Modelo de embeddings — configurável, nunca hardcoded na função principal
# (RNF-44). Default multilíngue (suporta português dos manuais técnicos e
# das queries) escolhido após checar as dependências do projeto: não havia
# uso prévio de sentence-transformers no repo (grep em todo o projeto), e
# este é o modelo MiniLM multilíngue mais leve/estável da família — CPU-only,
# sem chave/API externa, compatível com o `sentence-transformers==3.3.1`
# fixado em requirements.txt.
EMBEDDING_MODEL: str = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Chunking — determinístico e simples (RF-20 pede para não complicar).
CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "150"))


# ---------------------------------------------------------------------------
# Estatísticas de uma execução
# ---------------------------------------------------------------------------


@dataclass
class IndexingStats:
    indexed: int = 0
    skipped: int = 0
    empty: list[str] = field(default_factory=list)  # PDFs sem texto extraível
    errors: list[str] = field(default_factory=list)  # arquivo -> motivo
    chunks_added: int = 0
    chunks_removed: int = 0

    def summary(self) -> str:
        return (
            f"indexed={self.indexed} skipped={self.skipped} "
            f"empty={len(self.empty)} errors={len(self.errors)} "
            f"chunks_added={self.chunks_added} chunks_removed={self.chunks_removed}"
        )


# ---------------------------------------------------------------------------
# Hash — RF-20. Sobre o CONTEÚDO do arquivo, nunca nome/data/tamanho.
# ---------------------------------------------------------------------------


def compute_file_hash(path: Path) -> str:
    """SHA-256 do conteúdo do arquivo, lido em blocos (não carrega o PDF
    inteiro na memória de uma vez)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Extração de texto — pypdf. Sem OCR (fora de escopo).
# ---------------------------------------------------------------------------


def extract_pdf_pages(path: Path) -> list[str]:
    """Retorna o texto de cada página (índice 0 = página 1), na ordem do
    documento. Página sem texto extraível vira string vazia — quem chama
    decide o que fazer (ignorar ao gerar chunks, ver `chunk_text`)."""
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pypdf pode falhar em página corrompida
            log.warning(
                "Falha ao extrair texto da página %d de %s: %s",
                page_number,
                path.name,
                exc,
            )
            text = ""
        pages.append(text)
    return pages


# ---------------------------------------------------------------------------
# Chunking — determinístico, com overlap, por página (mantém o metadado
# `page` correto sem cruzar limites de página).
# ---------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """mesmo texto + mesma config -> mesmos chunks, sempre (RF-20). Divide
    por índice de caractere — simples e reproduzível, sem dependência de
    tokenizer/algoritmo externo."""
    if chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP deve ser menor que CHUNK_SIZE")

    stripped = text.strip()
    if not stripped:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    start = 0
    length = len(stripped)
    while start < length:
        end = min(start + chunk_size, length)
        piece = stripped[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == length:
            break
        start += step
    return chunks


# ---------------------------------------------------------------------------
# IDs determinísticos — RF-20. file_hash + page + chunk_index.
# ---------------------------------------------------------------------------


def build_chunk_id(file_hash: str, page: int, chunk_index: int) -> str:
    return f"{file_hash}_p{page}_c{chunk_index}"


# ---------------------------------------------------------------------------
# Embeddings — sentence-transformers, modelo carregado uma vez por execução.
# ---------------------------------------------------------------------------


def load_embedding_model(model_name: str) -> SentenceTransformer:
    log.info("Carregando modelo de embeddings: %s", model_name)
    return SentenceTransformer(model_name)


def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """Um único `encode` em batch — nunca um `load`/`encode` por chunk."""
    if not texts:
        return []
    vectors = model.encode(
        texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
    )
    return vectors.tolist()


# ---------------------------------------------------------------------------
# ChromaDB — persistente, collection dedicada.
# ---------------------------------------------------------------------------


def get_collection(chroma_path: Path) -> Any:
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Chunks de manuais técnicos de manutenção (RF-20)."},
    )


def _existing_file_hash(collection: Any, file_name: str) -> str | None:
    """Hash já indexado para `file_name`, ou None se nunca foi indexado."""
    result = collection.get(
        where={"file_name": file_name}, limit=1, include=["metadatas"]
    )
    metadatas = result.get("metadatas") or []
    if not metadatas:
        return None
    return metadatas[0].get("file_hash")


def _delete_existing_chunks(collection: Any, file_name: str) -> int:
    result = collection.get(where={"file_name": file_name}, include=[])
    ids = result.get("ids") or []
    if ids:
        collection.delete(ids=ids)
    return len(ids)


# ---------------------------------------------------------------------------
# Indexação de um único PDF
# ---------------------------------------------------------------------------


def _build_chunks_for_pdf(
    path: Path,
    manuals_dir: Path,
    file_hash: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model_name: str,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Extrai + gera (ids, documents, metadatas) para todos os chunks do
    PDF. Não toca o ChromaDB — só monta os dados em memória, para que um
    erro aqui nunca deixe a base em estado parcial/inconsistente."""
    pages = extract_pdf_pages(path)
    source = path.relative_to(manuals_dir).as_posix()

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for page_number, page_text in enumerate(pages, start=1):
        page_chunks = chunk_text(page_text, chunk_size, chunk_overlap)
        for chunk_index, chunk in enumerate(page_chunks):
            ids.append(build_chunk_id(file_hash, page_number, chunk_index))
            documents.append(chunk)
            metadatas.append(
                {
                    "source": source,
                    "file_name": path.name,
                    "file_hash": file_hash,
                    "page": page_number,
                    "chunk_index": chunk_index,
                    "embedding_model": embedding_model_name,
                }
            )

    return ids, documents, metadatas


def index_manual(
    path: Path,
    manuals_dir: Path,
    collection: Any,
    model: SentenceTransformer,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model_name: str,
    stats: IndexingStats,
) -> None:
    file_name = path.name
    try:
        file_hash = compute_file_hash(path)
    except OSError as exc:
        log.error("Não foi possível ler %s: %s", file_name, exc)
        stats.errors.append(file_name)
        return

    existing_hash = _existing_file_hash(collection, file_name)
    if existing_hash == file_hash:
        log.info("Skip (hash inalterado): %s", file_name)
        stats.skipped += 1
        return

    try:
        ids, documents, metadatas = _build_chunks_for_pdf(
            path,
            manuals_dir,
            file_hash,
            chunk_size,
            chunk_overlap,
            embedding_model_name,
        )
        if not documents:
            log.warning(
                "PDF sem texto extraível, ignorado (sem OCR nesta task): %s", file_name
            )
            stats.empty.append(file_name)
            return

        embeddings = embed_texts(model, documents)
    except Exception as exc:
        # Falha na extração/chunking/embedding: NÃO mexe no ChromaDB. Se já
        # havia uma versão anterior indexada, ela permanece válida — o
        # documento simplesmente não é marcado como indexado com sucesso.
        log.error(
            "Falha ao processar %s, arquivo não foi (re)indexado: %s", file_name, exc
        )
        stats.errors.append(file_name)
        return

    removed = 0
    if existing_hash is not None:
        removed = _delete_existing_chunks(collection, file_name)

    collection.upsert(
        ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
    )

    stats.indexed += 1
    stats.chunks_added += len(ids)
    stats.chunks_removed += removed
    action = "Reindexado" if existing_hash is not None else "Indexado"
    log.info("%s: %s (%d chunks, hash=%s)", action, file_name, len(ids), file_hash[:12])


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------


def run_indexing(
    manuals_dir: Path,
    chroma_path: Path,
    embedding_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> IndexingStats:
    stats = IndexingStats()

    manuals_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(manuals_dir.rglob("*.pdf"))  # ordem determinística

    if not pdf_paths:
        log.warning(
            "Nenhum PDF encontrado em %s — coloque manuais lá e rode de novo.",
            manuals_dir,
        )
        return stats

    collection = get_collection(chroma_path)
    model = load_embedding_model(embedding_model_name)  # uma vez para toda a execução

    log.info("%d PDF(s) encontrado(s) em %s", len(pdf_paths), manuals_dir)
    for pdf_path in pdf_paths:
        index_manual(
            pdf_path,
            manuals_dir,
            collection,
            model,
            chunk_size,
            chunk_overlap,
            embedding_model_name,
            stats,
        )

    log.info("Concluído | modelo=%s | %s", embedding_model_name, stats.summary())
    return stats


def main() -> IndexingStats:
    log.info(
        "Iniciando indexação | manuals_dir=%s chroma_db_path=%s chunk_size=%d chunk_overlap=%d",
        MANUALS_DIR,
        CHROMA_DB_PATH,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )
    return run_indexing(
        MANUALS_DIR, CHROMA_DB_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
    )


if __name__ == "__main__":
    main()
