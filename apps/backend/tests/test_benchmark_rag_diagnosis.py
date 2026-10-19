"""
Deterministic pieces of the RAG diagnosis benchmark (benchmarks/benchmark_rag_diagnosis.py).

1. WAP-5 task codes are read through OCR noise (spaces, ``O`` for ``0``).
2. Identification is a rank-weighted vote; a tie is "unidentified".
3. The random-ranking baseline is the hypergeometric P(hit in top-k).
4. The operator symptom never names the failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import benchmark_rag_diagnosis as b  # noqa: E402


def _chunk(text: str) -> dict[str, object]:
    return {"text": text, "score": 0.7, "metadata": {"file_name": b.WAP5, "page": 173}}


def test_task_codes_survive_ocr_noise() -> None:
    assert b.task_codes("602.1 W.O1 Air drier -Inspect; 601.1M.02 oil") == {
        "602.1W.01",
        "601.1M.02",
    }
    assert b.chunk_family("602.1 W.O1 Air drier") == {"air_dryer"}


def test_identify_is_rank_weighted_and_ties_are_none() -> None:
    dryer, oil = _chunk("602.1M.01 Air drier"), _chunk("601.3M.06 oil change")
    assert b.identify([dryer, oil]) == "air_dryer"
    assert b.identify([oil, dryer, dryer, dryer]) == "air_dryer"  # 1/2+1/3+1/4 > 1
    assert b.identify([_chunk("no code here")]) is None
    both = _chunk("601.3M.06 and 602.1M.01")
    assert b.identify([both]) is None


def test_random_hit_baseline() -> None:
    assert b.random_hit_at_k(25, 0, 5) == 0.0
    assert b.random_hit_at_k(25, 25, 5) == 1.0
    assert b.random_hit_at_k(25, 1, 5) == pytest.approx(5 / 25)


def test_operator_symptom_does_not_name_failure() -> None:
    dev = [
        {"sensor": "TP2", "z": -3.0},
        {"sensor": "Motor_current", "z": 2.1},
        {"sensor": "Towers", "z": 0.5},
    ]
    text = b.operator_symptom(dev)
    assert "abaixo do normal" in text and "acima do normal" in text
    assert "vazamento" not in text and "leak" not in text
