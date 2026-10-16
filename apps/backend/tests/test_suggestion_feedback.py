"""
RF-38 — feedback do técnico (human-in-the-loop).

Garantias verificadas
---------------------
1.  Feedback persiste e os votos por trecho viram priors Laplace por equipamento.
2.  Um único voto move o prior só até ±1/3 (nenhum técnico domina sozinho).
3.  Priors de outro equipamento não vazam.
4.  O re-ranking reordena trechos que JÁ passaram τ, mas nunca resgata um
    trecho abaixo de τ (o status de fundamentação não muda por feedback).
5.  Falha ao ler o feedback não derruba a sugestão.
6.  Endpoint POST /v1/maintenance/feedback -> 201 e GET /feedback/priors.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.database import Base, get_db
from src.main import create_app
from src.schemas.maintenance import (
    MaintenanceSuggestionRequest,
    ReferenceVote,
    SuggestionFeedbackCreate,
)
from src.services.maintenance_suggestion_service import MaintenanceSuggestionService
from src.services.suggestion_feedback_service import (
    SuggestionFeedbackService,
    laplace_prior,
    rerank,
)

EQUIP = "Compressor de ar da APU"


@pytest.fixture()
async def session() -> AsyncGenerator[AsyncSession, None]:
    engine: AsyncEngine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )
    async with factory() as s:
        yield s
    await engine.dispose()


def _fb(
    useful: bool, votes: list[tuple[str, int, bool]], equipment: str = EQUIP
) -> SuggestionFeedbackCreate:
    return SuggestionFeedbackCreate(
        equipment_name=equipment,
        useful=useful,
        reference_votes=[
            ReferenceVote(file_name=f, page=p, helpful=h) for f, p, h in votes
        ],
    )


def test_laplace_prior_bounds() -> None:
    assert laplace_prior(0, 0) == 0.0
    assert laplace_prior(1, 0) == pytest.approx(1 / 3)
    assert laplace_prior(0, 1) == pytest.approx(-1 / 3)
    assert -1 < laplace_prior(0, 50) < laplace_prior(50, 0) < 1


async def test_priors_aggregate_votes_per_equipment(session: AsyncSession) -> None:
    svc = SuggestionFeedbackService(session)
    await svc.create(_fb(True, [("wap5.pdf", 174, True), ("bomba.pdf", 3, False)]))
    await svc.create(_fb(True, [("wap5.pdf", 174, True)]))
    await svc.create(_fb(False, [("wap5.pdf", 174, False)], equipment="Outro"))

    priors = {(p.file_name, p.page): p for p in await svc.reference_priors(EQUIP)}

    assert priors[("wap5.pdf", 174)].helpful_votes == 2
    assert priors[("wap5.pdf", 174)].unhelpful_votes == 0  # voto de "Outro" não vaza
    assert priors[("wap5.pdf", 174)].prior == pytest.approx(0.5)
    assert priors[("bomba.pdf", 3)].prior == pytest.approx(-1 / 3)


class _Chunk:
    def __init__(self, file_name: str, page: int, score: float) -> None:
        self.file_name, self.page, self.score = file_name, page, score


def test_rerank_breaks_near_ties_only() -> None:
    a, b = _Chunk("a.pdf", 1, 0.700), _Chunk("b.pdf", 2, 0.690)
    assert rerank([a, b], {("b.pdf", 2): 0.5})[0] is b  # 0.690 + 0.025 > 0.700
    far = _Chunk("c.pdf", 3, 0.500)
    assert rerank([a, far], {("c.pdf", 3): 0.99})[0] is a  # prior não supera gap grande


def _mcp(*chunks: tuple[str, int, float]) -> Any:
    mcp = AsyncMock()
    mcp.search_maintenance_manual = AsyncMock(
        return_value={
            "query": "q",
            "results": [
                {
                    "text": f"{f} p{p}",
                    "score": s,
                    "metadata": {"file_name": f, "page": p, "chunk_index": 0},
                }
                for f, p, s in chunks
            ],
        }
    )
    return mcp


def _ollama() -> Any:
    o = AsyncMock()
    o.generate = AsyncMock(
        return_value="# Plano\n\n## Referências\n- `b.pdf`, página 2\n"
    )
    return o


REQ = MaintenanceSuggestionRequest(failure_probability=0.9, equipment_name=EQUIP)


async def test_feedback_reorders_prompt_but_never_rescues_below_tau() -> None:
    async def priors(_: str) -> dict[tuple[str, int], float]:
        return {("b.pdf", 2): 0.5, ("low.pdf", 9): 0.99}

    ollama = _ollama()
    service = MaintenanceSuggestionService(
        _mcp(("a.pdf", 1, 0.70), ("b.pdf", 2, 0.69), ("low.pdf", 9, 0.55)),
        ollama,
        "m",
        grounding_threshold=0.6,
        prior_provider=priors,
    )

    resp = await service.suggest(REQ)

    assert [r.file_name for r in resp.references] == ["b.pdf", "a.pdf"]
    _, prompt = ollama.generate.await_args.args
    assert prompt.index("b.pdf") < prompt.index("a.pdf")
    assert "low.pdf" not in prompt
    assert resp.grounding is not None and resp.grounding.contexts_discarded == 1


async def test_feedback_failure_does_not_break_suggestion() -> None:
    async def broken(_: str) -> dict[tuple[str, int], float]:
        raise RuntimeError("db down")

    service = MaintenanceSuggestionService(
        _mcp(("a.pdf", 1, 0.70), ("b.pdf", 2, 0.69)),
        _ollama(),
        "m",
        prior_provider=broken,
    )
    resp = await service.suggest(REQ)
    assert [r.file_name for r in resp.references] == ["a.pdf", "b.pdf"]


async def test_feedback_endpoints(session: AsyncSession) -> None:
    app: FastAPI = create_app()

    async def _db() -> AsyncGenerator[AsyncSession, None]:
        yield session

    app.dependency_overrides[get_db] = _db
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        created = await client.post(
            "/v1/maintenance/feedback",
            json={
                "equipment_name": EQUIP,
                "useful": True,
                "grounding_status": "grounded",
                "reference_votes": [
                    {"file_name": "wap5.pdf", "page": 174, "helpful": True}
                ],
            },
        )
        bad = await client.post(
            "/v1/maintenance/feedback", json={"equipment_name": EQUIP}
        )
        priors = await client.get(
            "/v1/maintenance/feedback/priors", params={"equipment_name": EQUIP}
        )

    assert created.status_code == 201 and created.json()["id"] == 1
    assert bad.status_code == 422  # `useful` é obrigatório
    assert priors.json() == [
        {
            "file_name": "wap5.pdf",
            "page": 174,
            "helpful_votes": 1,
            "unhelpful_votes": 0,
            "prior": pytest.approx(1 / 3),
        }
    ]
