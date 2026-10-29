"""
Testes da pipeline de ingestão de manuais (RF-20 / RNF-44).

Nenhum teste aqui baixa modelo real nem depende de internet: o carregamento
do `SentenceTransformer` é mockado em todo o arquivo (`_FakeSentenceTransformer`)
e o ChromaDB usa sempre um diretório temporário (`tmp_path`), nunca o
`data/chroma` real de desenvolvimento. Os PDFs de teste são construídos em
memória com o próprio `pypdf` (baixo nível: fonte Helvetica base-14 + stream
de conteúdo `Tj`) — não precisa de uma lib de autoria de PDF a mais só para
os testes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import index_manuals as im  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture: PDF mínimo construído em memória
# ---------------------------------------------------------------------------


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _make_pdf_bytes(page_texts: list[str]) -> bytes:
    """PDF válido com uma página por item de `page_texts`. Página com string
    vazia vira página sem `/Contents` (sem texto extraível — cobre o caso
    "PDF sem texto"). Usa só objetos de baixo nível do pypdf (fonte Type1
    base-14 Helvetica, sem embutir fonte), então nenhuma lib extra de
    geração de PDF entra em requirements.txt só para o teste."""
    writer = PdfWriter()
    for text in page_texts:
        page = writer.add_blank_page(width=200, height=200)
        if not text:
            continue
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        font_ref = writer._add_object(font)
        resources = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
        )
        page[NameObject("/Resources")] = resources
        content = f"BT /F1 12 Tf 10 100 Td ({_escape_pdf_text(text)}) Tj ET".encode(
            "latin-1"
        )
        stream = DecodedStreamObject()
        stream.set_data(content)
        page[NameObject("/Contents")] = writer._add_object(stream)

    import io

    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _write_pdf(path: Path, page_texts: list[str]) -> None:
    path.write_bytes(_make_pdf_bytes(page_texts))


# ---------------------------------------------------------------------------
# Fake SentenceTransformer — sem download, sem rede.
# ---------------------------------------------------------------------------


class _FakeSentenceTransformer:
    """Registra o `model_name` recebido (verifica RNF-44 — modelo vem da
    config) e devolve embeddings determinísticos e baratos."""

    instances: list[str] = []

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        _FakeSentenceTransformer.instances.append(model_name)

    def encode(
        self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
    ):
        return np.array([[float(len(t) % 11), 1.0, 0.0] for t in texts])


@pytest.fixture(autouse=True)
def _no_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aplica o mock em TODOS os testes deste arquivo — nenhum teste baixa
    modelo real por engano."""
    _FakeSentenceTransformer.instances = []
    monkeypatch.setattr(im, "SentenceTransformer", _FakeSentenceTransformer)


def _index(manuals_dir: Path, chroma_dir: Path, **overrides: Any) -> im.IndexingStats:
    kwargs: dict[str, Any] = dict(
        manuals_dir=manuals_dir,
        chroma_path=chroma_dir,
        embedding_model_name="fake-model-for-tests",
        chunk_size=50,
        chunk_overlap=10,
    )
    kwargs.update(overrides)
    return im.run_indexing(**kwargs)


# ---------------------------------------------------------------------------
# 1. Hash
# ---------------------------------------------------------------------------


def test_hash_same_content_same_hash(tmp_path: Path) -> None:
    a = tmp_path / "a.pdf"
    a.write_bytes(b"conteudo identico")
    assert im.compute_file_hash(a) == im.compute_file_hash(a)


def test_hash_different_content_different_hash(tmp_path: Path) -> None:
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    a.write_bytes(b"conteudo A")
    b.write_bytes(b"conteudo B, bem diferente")
    assert im.compute_file_hash(a) != im.compute_file_hash(b)


def test_hash_ignores_filename_only_content_matters(tmp_path: Path) -> None:
    a = tmp_path / "manual-x.pdf"
    b = tmp_path / "manual-y-outro-nome.pdf"
    a.write_bytes(b"mesmo conteudo binario")
    b.write_bytes(b"mesmo conteudo binario")
    assert im.compute_file_hash(a) == im.compute_file_hash(b)


# ---------------------------------------------------------------------------
# 2. Extração
# ---------------------------------------------------------------------------


def test_extract_pdf_pages_returns_real_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "manual.pdf"
    _write_pdf(pdf_path, ["Trocar oleo do compressor a cada 500 horas."])

    pages = im.extract_pdf_pages(pdf_path)

    assert len(pages) == 1
    assert "Trocar oleo do compressor" in pages[0]


def test_extract_pdf_pages_preserves_page_order(tmp_path: Path) -> None:
    pdf_path = tmp_path / "manual.pdf"
    _write_pdf(pdf_path, ["pagina um aqui", "pagina dois aqui", "pagina tres aqui"])

    pages = im.extract_pdf_pages(pdf_path)

    assert len(pages) == 3
    assert "um" in pages[0]
    assert "dois" in pages[1]
    assert "tres" in pages[2]


# ---------------------------------------------------------------------------
# 3. Chunking
# ---------------------------------------------------------------------------


def test_chunk_text_generates_chunks_with_overlap() -> None:
    text = "abcdefghij" * 10  # 100 chars
    chunks = im.chunk_text(text, chunk_size=30, chunk_overlap=10)

    assert len(chunks) > 1
    # overlap real: o fim de um chunk aparece no começo do próximo
    assert chunks[0][-10:] == chunks[1][:10]


def test_chunk_text_does_not_lose_content() -> None:
    text = "Manual de manutencao do motor industrial trifasico. " * 3
    chunks = im.chunk_text(text, chunk_size=25, chunk_overlap=5)

    assert chunks
    # com overlap, a soma dos chunks só pode ser >= o texto original —
    # nunca menor (perder conteúdo significaria soma < len(text.strip())).
    assert sum(len(c) for c in chunks) >= len(text.strip())
    assert chunks[0].startswith(text.strip()[:5])
    assert chunks[-1].endswith(text.strip()[-5:])


def test_chunk_text_deterministic_same_input_same_output() -> None:
    text = "Manual de manutencao do motor industrial. " * 5
    c1 = im.chunk_text(text, chunk_size=40, chunk_overlap=8)
    c2 = im.chunk_text(text, chunk_size=40, chunk_overlap=8)
    assert c1 == c2


def test_chunk_text_empty_text_returns_no_chunks() -> None:
    assert im.chunk_text("   ", chunk_size=100, chunk_overlap=10) == []


def test_chunk_id_deterministic() -> None:
    id1 = im.build_chunk_id("abc123", page=2, chunk_index=0)
    id2 = im.build_chunk_id("abc123", page=2, chunk_index=0)
    assert id1 == id2
    assert id1 != im.build_chunk_id("abc123", page=2, chunk_index=1)
    assert id1 != im.build_chunk_id("abc999", page=2, chunk_index=0)


# ---------------------------------------------------------------------------
# 4. Incremental
# ---------------------------------------------------------------------------


def test_second_run_without_changes_skips_and_stays_idempotent(tmp_path: Path) -> None:
    manuals_dir = tmp_path / "manuals"
    manuals_dir.mkdir()
    chroma_dir = tmp_path / "chroma"
    _write_pdf(manuals_dir / "manual.pdf", ["Texto do manual de teste, " * 5])

    first = _index(manuals_dir, chroma_dir)
    assert first.indexed == 1
    assert first.skipped == 0

    collection = im.get_collection(chroma_dir)
    count_after_first = collection.count()
    assert count_after_first > 0

    second = _index(manuals_dir, chroma_dir)
    assert second.indexed == 0
    assert second.skipped == 1

    assert im.get_collection(chroma_dir).count() == count_after_first


# ---------------------------------------------------------------------------
# 5. Alteração do PDF
# ---------------------------------------------------------------------------


def test_changed_pdf_is_reindexed_without_leaving_stale_chunks(tmp_path: Path) -> None:
    manuals_dir = tmp_path / "manuals"
    manuals_dir.mkdir()
    chroma_dir = tmp_path / "chroma"
    pdf_path = manuals_dir / "manual.pdf"
    _write_pdf(pdf_path, ["Conteudo original do manual. " * 5])

    _index(manuals_dir, chroma_dir)
    hash_before = im.compute_file_hash(pdf_path)

    _write_pdf(pdf_path, ["Conteudo TOTALMENTE novo e maior do manual revisado. " * 8])
    hash_after = im.compute_file_hash(pdf_path)
    assert hash_before != hash_after

    result = _index(manuals_dir, chroma_dir)
    assert result.indexed == 1
    assert result.skipped == 0

    collection = im.get_collection(chroma_dir)
    all_docs = collection.get(where={"file_name": "manual.pdf"}, include=["metadatas"])
    hashes_present = {m["file_hash"] for m in all_docs["metadatas"]}
    assert hashes_present == {hash_after}  # nenhum chunk do hash antigo sobrou


# ---------------------------------------------------------------------------
# 6. Metadados
# ---------------------------------------------------------------------------


def test_chunk_metadata_has_required_fields(tmp_path: Path) -> None:
    manuals_dir = tmp_path / "manuals"
    manuals_dir.mkdir()
    chroma_dir = tmp_path / "chroma"
    _write_pdf(manuals_dir / "bomba.pdf", ["Manual da bomba centrifuga. " * 5])

    _index(manuals_dir, chroma_dir)

    collection = im.get_collection(chroma_dir)
    result = collection.get(include=["metadatas"])
    assert len(result["metadatas"]) > 0
    for metadata in result["metadatas"]:
        assert metadata["source"] == "bomba.pdf"
        assert metadata["file_name"] == "bomba.pdf"
        assert (
            isinstance(metadata["file_hash"], str) and len(metadata["file_hash"]) == 64
        )
        assert metadata["page"] == 1
        assert isinstance(metadata["chunk_index"], int)
        assert metadata["embedding_model"] == "fake-model-for-tests"


# ---------------------------------------------------------------------------
# 7. Modelo configurável
# ---------------------------------------------------------------------------


def test_embedding_model_comes_from_config_not_hardcoded(tmp_path: Path) -> None:
    manuals_dir = tmp_path / "manuals"
    manuals_dir.mkdir()
    chroma_dir = tmp_path / "chroma"
    _write_pdf(
        manuals_dir / "manual.pdf", ["Texto qualquer para gerar embedding. " * 3]
    )

    _index(manuals_dir, chroma_dir, embedding_model_name="modelo-customizado-xyz")

    assert "modelo-customizado-xyz" in _FakeSentenceTransformer.instances


# ---------------------------------------------------------------------------
# 8. ChromaDB em diretório temporário
# ---------------------------------------------------------------------------


def test_uses_temp_chroma_dir_not_dev_database(tmp_path: Path) -> None:
    manuals_dir = tmp_path / "manuals"
    manuals_dir.mkdir()
    chroma_dir = tmp_path / "chroma_isolado"
    _write_pdf(manuals_dir / "manual.pdf", ["Conteudo isolado de teste. " * 3])

    _index(manuals_dir, chroma_dir)

    assert chroma_dir.exists()
    assert chroma_dir != im.CHROMA_DB_PATH  # nunca aponta pro banco de dev real


# ---------------------------------------------------------------------------
# 9. PDF sem texto (sem OCR)
# ---------------------------------------------------------------------------


def test_pdf_without_text_does_not_break_pipeline(tmp_path: Path) -> None:
    manuals_dir = tmp_path / "manuals"
    manuals_dir.mkdir()
    chroma_dir = tmp_path / "chroma"
    _write_pdf(manuals_dir / "escaneado.pdf", [""])  # página sem /Contents
    _write_pdf(manuals_dir / "normal.pdf", ["Este manual tem texto normal. " * 5])

    stats = _index(manuals_dir, chroma_dir)

    assert "escaneado.pdf" in stats.empty
    assert stats.indexed == 1  # normal.pdf processado normalmente
    assert "normal.pdf" not in stats.empty
    assert stats.errors == []  # sem texto não é erro, execução não quebra


# ---------------------------------------------------------------------------
# 10. Perfil por manual (sidecar <nome>.meta.json) — curadoria do corpus
# ---------------------------------------------------------------------------


def _write_profile(pdf_path: Path, profile: dict[str, Any]) -> None:
    im.profile_path_for(pdf_path).write_text(json.dumps(profile), encoding="utf-8")


def test_profile_restricts_page_range_and_keeps_original_page_numbers(
    tmp_path: Path,
) -> None:
    manuals_dir = tmp_path / "manuals"
    manuals_dir.mkdir()
    chroma_dir = tmp_path / "chroma"
    pdf = manuals_dir / "loco.pdf"
    _write_pdf(
        pdf, ["Cabine e portas. " * 4, "Compressor principal. " * 4, "Pantografo. " * 4]
    )
    _write_profile(pdf, {"pages": [2, 2]})

    _index(manuals_dir, chroma_dir)

    result = im.get_collection(chroma_dir).get(include=["metadatas", "documents"])
    assert {m["page"] for m in result["metadatas"]} == {2}
    assert all("Cabine" not in d and "Pantografo" not in d for d in result["documents"])


def test_profile_adds_scalar_metadata_without_overriding_core_fields(
    tmp_path: Path,
) -> None:
    manuals_dir = tmp_path / "manuals"
    manuals_dir.mkdir()
    chroma_dir = tmp_path / "chroma"
    pdf = manuals_dir / "loco.pdf"
    _write_pdf(pdf, ["Compressor principal inspecao. " * 4])
    _write_profile(
        pdf,
        {"metadata": {"domain": "locomotive", "doc_type": "scheduled", "page": 999}},
    )

    _index(manuals_dir, chroma_dir)

    for meta in im.get_collection(chroma_dir).get(include=["metadatas"])["metadatas"]:
        assert meta["domain"] == "locomotive"
        assert meta["doc_type"] == "scheduled"
        assert meta["page"] == 1  # campo do núcleo nunca é sobrescrito pelo sidecar


def test_strip_cross_references_keeps_task_description() -> None:
    text = (
        "601.1W.02 Main compressor oil -Check Check the oil level in the compressor. "
        "Refer to Suppliers Documentation Volume F13, Section 8, Metcalfe Compressor "
        "Type 2A320D Maintenance lnstructions (A666). Section 7.4 "
        "603.4Y.03 Drain cock -Replace Replace the drain cocks. Refer to Suppliers "
        "Documentation Volume F11, Section 7, Isolating and Drain Cocks Instructions. (A34). "
        "602.1Y.04 Air drier cocks -Check Replace the cocks if defective. Refer to Volume D, "
        "Maintenance and Repair Manual, Chapter 6.2, Air Drier. 602.2Y.01 Desiccant"
    )
    cleaned = " ".join(im.strip_cross_references(text).split())
    assert "Refer to" not in cleaned
    assert "(A666)" not in cleaned and "(A34)" not in cleaned
    assert "Section 7.4" not in cleaned
    assert "Check the oil level in the compressor." in cleaned
    assert "Replace the cocks if defective." in cleaned
    assert cleaned.endswith("602.2Y.01 Desiccant")


def test_profile_drops_reference_only_chunks(tmp_path: Path) -> None:
    manuals_dir = tmp_path / "manuals"
    manuals_dir.mkdir()
    chroma_dir = tmp_path / "chroma"
    pdf = manuals_dir / "loco.pdf"
    _write_pdf(
        pdf,
        [
            "Refer to Suppliers Documentation Volume F13, Section 8, Compressor (A666).",
            "Check the oil level in the compressor and top up with the specified lubricant.",
        ],
    )
    _write_profile(pdf, {"strip_cross_references": True, "min_chunk_chars": 20})

    _index(manuals_dir, chroma_dir, chunk_size=500, chunk_overlap=50)

    result = im.get_collection(chroma_dir).get(include=["metadatas", "documents"])
    assert {m["page"] for m in result["metadatas"]} == {2}
    assert all("Refer to" not in d for d in result["documents"])


def test_profile_change_triggers_reindex(tmp_path: Path) -> None:
    manuals_dir = tmp_path / "manuals"
    manuals_dir.mkdir()
    chroma_dir = tmp_path / "chroma"
    pdf = manuals_dir / "loco.pdf"
    _write_pdf(pdf, ["Pagina um. " * 4, "Pagina dois. " * 4])

    first = _index(manuals_dir, chroma_dir)
    again = _index(manuals_dir, chroma_dir)
    _write_profile(pdf, {"pages": [2, 2]})
    changed = _index(manuals_dir, chroma_dir)

    assert first.indexed == 1 and again.skipped == 1
    assert changed.indexed == 1 and changed.chunks_removed > 0
    pages = {
        m["page"]
        for m in im.get_collection(chroma_dir).get(include=["metadatas"])["metadatas"]
    }
    assert pages == {2}


@pytest.mark.parametrize(
    "bad",
    [
        {"pages": [5, 2]},
        {"pages": [0, 3]},
        {"pages": "1-3"},
        {"metadata": {"nested": {"a": 1}}},
        {"min_chunk_chars": -1},
        [1, 2],
    ],
)
def test_invalid_profile_is_an_error_and_not_indexed(tmp_path: Path, bad: Any) -> None:
    manuals_dir = tmp_path / "manuals"
    manuals_dir.mkdir()
    chroma_dir = tmp_path / "chroma"
    pdf = manuals_dir / "loco.pdf"
    _write_pdf(pdf, ["Compressor. " * 5])
    im.profile_path_for(pdf).write_text(json.dumps(bad), encoding="utf-8")

    stats = _index(manuals_dir, chroma_dir)

    assert stats.errors == ["loco.pdf"] and stats.indexed == 0


# ---------------------------------------------------------------------------
# 11. Guias Markdown do projeto (GUIDES_DIR)
# ---------------------------------------------------------------------------


def test_markdown_sections_become_pages(tmp_path: Path) -> None:
    guide = tmp_path / "guia.md"
    guide.write_text(
        "# Titulo\nintro\n\n## Vazamento de ar\nTP3 cai\n\n## Oleo\nnivel baixo\n",
        encoding="utf-8",
    )
    pages = im.extract_markdown_sections(guide)
    assert len(pages) == 3
    assert pages[1].startswith("## Vazamento de ar")
    assert "TP3 cai" in pages[1]


def test_guides_dir_is_indexed_with_section_pages(tmp_path: Path) -> None:
    manuals_dir = tmp_path / "manuals"
    guides_dir = tmp_path / "guides"
    manuals_dir.mkdir()
    guides_dir.mkdir()
    chroma_dir = tmp_path / "chroma"
    _write_pdf(manuals_dir / "bomba.pdf", ["Manual da bomba centrifuga. " * 3])
    (guides_dir / "guia.md").write_text(
        "# Guia\n\n## Vazamento de ar\nTP3 cai com o compressor em alivio.\n",
        encoding="utf-8",
    )
    (guides_dir / "guia.meta.json").write_text(
        '{"metadata": {"doc_type": "diagnostic_guide"}}', encoding="utf-8"
    )

    stats = _index(manuals_dir, chroma_dir, extra_dirs=(guides_dir,))

    assert stats.indexed == 2 and not stats.errors
    metadatas = im.get_collection(chroma_dir).get(include=["metadatas"])["metadatas"]
    guide_meta = [m for m in metadatas if m["file_name"] == "guia.md"]
    assert guide_meta and all(m["source"] == "guia.md" for m in guide_meta)
    assert {m["page"] for m in guide_meta} == {1, 2}
    assert all(m["doc_type"] == "diagnostic_guide" for m in guide_meta)


def test_profile_chunk_size_keeps_a_guide_section_in_one_chunk(tmp_path: Path) -> None:
    guides_dir = tmp_path / "guides"
    guides_dir.mkdir()
    chroma_dir = tmp_path / "chroma"
    body = "Vazamento de ar: diagnostico, reparo e teste. " * 10  # > 50 chars
    (guides_dir / "guia.md").write_text(f"## Falha\n{body}\n", encoding="utf-8")
    (guides_dir / "guia.meta.json").write_text('{"chunk_size": 5000}', encoding="utf-8")

    _index(tmp_path / "manuals", chroma_dir, extra_dirs=(guides_dir,))

    docs = im.get_collection(chroma_dir).get(include=["documents"])["documents"]
    assert len(docs) == 1 and docs[0].endswith(body.strip())
