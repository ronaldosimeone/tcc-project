"""
History router — fleet, failure events, KPIs and work orders.

I/O only (Clean Arch §3): validation via Pydantic/Query, services injected
with ``Depends``. All rules (RN-03 episodes, KPI definitions, work-order
state machine) live in ``services/history_service.py`` and
``services/failure_event_service.py``.
"""

from __future__ import annotations

from datetime import datetime

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path, Query, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.database import get_db
from src.core.security import Actor, require_role
from src.schemas.history import (
    AssetOut,
    DailyEventCount,
    EventPage,
    FailureEventDetail,
    KpiReport,
    WorkOrderCreate,
    WorkOrderOut,
    WorkOrderUpdate,
)
from src.services.audit_service import AuditService
from src.services.failure_event_service import FailureEventService
from src.services.model_registry import ModelRegistry, get_model_registry
from src.services.history_service import HistoryService, resolve_window
from src.services.shadow_service import ShadowReport, shadow_report

router: APIRouter = APIRouter(prefix="/v1", tags=["History"])


class ErrorBody(BaseModel):
    """Corpo padronizado dos erros de domínio (handler global de AppError)."""

    error: str
    detail: str


# IDs são INTEGER (32 bits) no Postgres: acima disso a consulta estoura
# (achado do teste de contrato) — rejeitado na validação com 422.
_MAX_ID = 2**31 - 1

_ERROR_TEXT: dict[int, str] = {
    400: "Requisição malformada",
    401: "Não autenticado",
    403: "Papel insuficiente",
    404: "Recurso inexistente",
    409: "Conflito de estado",
}


def _errors(*codes: int) -> dict[int | str, dict[str, Any]]:
    """Declara no OpenAPI os erros que a rota pode devolver (contrato testado
    por tests/test_contract_schemathesis.py)."""
    return {c: {"model": ErrorBody, "description": _ERROR_TEXT[c]} for c in codes}


def get_history_service(db: AsyncSession = Depends(get_db)) -> HistoryService:
    return HistoryService(db)


@router.get(
    "/assets", response_model=list[AssetOut], summary="Frota (taxonomia ISO 14224)"
)
async def list_assets(
    service: HistoryService = Depends(get_history_service),
) -> list[AssetOut]:
    return await service.list_assets()


@router.get(
    "/events",
    response_model=EventPage,
    summary="Eventos de falha (episódios RN-03), paginados por cursor",
    responses=_errors(400),
)
async def list_events(
    asset_id: str | None = Query(default=None, max_length=64),
    severity: str | None = Query(default=None, pattern="^(alerta|critico)$"),
    event_status: str | None = Query(
        default=None, alias="status", pattern="^(open|closed)$"
    ),
    start: datetime | None = None,
    end: datetime | None = None,
    cursor: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=50, ge=1, le=500),
    service: HistoryService = Depends(get_history_service),
) -> EventPage:
    return await service.list_events(
        asset_id, severity, event_status, start, end, cursor, limit
    )


@router.get(
    "/events/daily",
    response_model=list[DailyEventCount],
    summary="Eventos por dia e severidade",
    responses=_errors(400),
)
async def daily_counts(
    asset_id: str | None = Query(default=None, max_length=64),
    start: datetime | None = None,
    end: datetime | None = None,
    service: HistoryService = Depends(get_history_service),
) -> list[DailyEventCount]:
    lo, hi = resolve_window(start, end)
    return await service.daily_counts(lo, hi, asset_id)


@router.get(
    "/events/{event_id}",
    response_model=FailureEventDetail,
    summary="Evento + causa raiz + OS + leituras reais",
    responses=_errors(401, 403, 404),
)
async def event_detail(
    event_id: Annotated[int, Path(ge=1, le=_MAX_ID)],
    actor: Actor = Depends(require_role("operator")),
    service: HistoryService = Depends(get_history_service),
    db: AsyncSession = Depends(get_db),
) -> FailureEventDetail:
    detail = await service.event_detail(event_id)
    # "Quem viu": leitura do detalhe de um alarme também entra na trilha.
    await AuditService(db).record(actor, "event.view", "failure_event", event_id)
    await db.commit()
    return detail


@router.post(
    "/events/rebuild/{asset_id}",
    summary="Recalcula os episódios de um ativo a partir das predições (admin)",
    responses=_errors(401, 403, 409),
)
async def rebuild_events(
    asset_id: str,
    actor: Actor = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
    registry: ModelRegistry = Depends(get_model_registry),
) -> dict[str, int]:
    created = await FailureEventService(threshold=settings.event_threshold).rebuild(
        db, asset_id, await registry.active_threshold()
    )
    await AuditService(db).record(
        actor, "events.rebuild", "asset", asset_id, {"events_created": created}
    )
    await db.commit()
    return {"events_created": created}


@router.get(
    "/kpis",
    response_model=KpiReport,
    summary="KPIs de confiabilidade (MTBF, MTTR, disponibilidade)",
    responses=_errors(400),
)
async def kpis(
    asset_id: str | None = Query(default=None, max_length=64),
    start: datetime | None = None,
    end: datetime | None = None,
    service: HistoryService = Depends(get_history_service),
) -> KpiReport:
    lo, hi = resolve_window(start, end)
    return await service.kpis(lo, hi, asset_id)


@router.get(
    "/work-orders", response_model=list[WorkOrderOut], summary="Ordens de serviço"
)
async def list_work_orders(
    asset_id: str | None = Query(default=None, max_length=64),
    wo_status: str | None = Query(
        default=None, alias="status", pattern="^(open|in_progress|closed|cancelled)$"
    ),
    service: HistoryService = Depends(get_history_service),
) -> list[WorkOrderOut]:
    return await service.list_work_orders(asset_id, wo_status)


@router.post(
    "/work-orders",
    response_model=WorkOrderOut,
    status_code=status.HTTP_201_CREATED,
    summary="Abre uma OS a partir de um evento",
    responses=_errors(401, 403, 404, 409),
)
async def create_work_order(
    payload: WorkOrderCreate,
    actor: Actor = Depends(require_role("technician")),
    service: HistoryService = Depends(get_history_service),
    db: AsyncSession = Depends(get_db),
) -> WorkOrderOut:
    wo = await service.create_work_order(payload)
    await AuditService(db).record(
        actor, "work_order.create", "work_order", wo.id, {"event_id": wo.event_id}
    )
    await db.commit()
    return wo


@router.patch(
    "/work-orders/{wo_id}",
    response_model=WorkOrderOut,
    summary="Atualiza OS / confirma a causa (ground truth)",
    responses=_errors(401, 403, 404, 409),
)
async def update_work_order(
    wo_id: Annotated[int, Path(ge=1, le=_MAX_ID)],
    payload: WorkOrderUpdate,
    actor: Actor = Depends(require_role("technician")),
    service: HistoryService = Depends(get_history_service),
    db: AsyncSession = Depends(get_db),
) -> WorkOrderOut:
    wo = await service.update_work_order(wo_id, payload)
    await AuditService(db).record(
        actor,
        "work_order.update",
        "work_order",
        wo_id,
        payload.model_dump(exclude_none=True, mode="json"),
    )
    await db.commit()
    return wo


@router.get(
    "/models/shadow",
    response_model=ShadowReport,
    summary="Champion × challenger em shadow: concordância nas últimas N horas (engenheiro+)",
    responses=_errors(401, 403),
)
async def shadow_stats(
    hours: int = Query(default=24, ge=1, le=24 * 30),
    _: Actor = Depends(require_role("engineer")),
    db: AsyncSession = Depends(get_db),
) -> ShadowReport:
    return await shadow_report(db, hours)
