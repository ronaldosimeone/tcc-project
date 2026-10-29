"""
Demo data for the "Histórico & Relatórios" screen.

Entry point (inside the api container):
    docker compose exec api python scripts/seed_demo_history.py

Deletes every failure event, work order and root-cause analysis, then writes
a fixed (seeded) set of FICTITIOUS events for the last 30 days across five
compressors, with work orders and sensor contributions, so the screen is not
empty in a demonstration. Nothing here is a model result: every work order
carries a "dados de demonstração" note, and the thesis results come only
from apps/ml (docs/results_freeze.md). Predictions are kept.

Back up first, e.g.:
    docker exec db pg_dump -U $POSTGRES_USER -d $POSTGRES_DB --data-only \\
        -t failure_events -t work_orders -t root_cause_analyses > backup.sql
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select

from src.core.database import AsyncSessionFactory
from src.models.asset import Asset
from src.models.failure_event import FailureEvent, RootCauseAnalysis, WorkOrder
from src.services.event_tracker import severity_for

SEED: int = 42
N_EVENTS: int = 42
DAYS: int = 30
MODEL_NAME: str = "random_forest_v3"
THRESHOLD: float = 0.694  # decision threshold of random_forest_v3 (model card)
CRITICAL_AT: float = 0.9  # EVENT_CRITICAL_AT default (risk scale)
DEMO_NOTE: str = "Dados de demonstração (scripts/seed_demo_history.py)."

MAIN_ASSET: str = "APU-Trem-042"
FLEET: dict[str, str] = {
    "APU-Trem-015": "Compressor de ar da APU — Trem 015",
    "APU-Trem-023": "Compressor de ar da APU — Trem 023",
    "APU-Trem-031": "Compressor de ar da APU — Trem 031",
    "APU-Trem-055": "Compressor de ar da APU — Trem 055",
}
ASSET_WEIGHTS: dict[str, float] = {
    MAIN_ASSET: 0.34,
    "APU-Trem-015": 0.14,
    "APU-Trem-023": 0.26,
    "APU-Trem-031": 0.10,
    "APU-Trem-055": 0.16,
}
TEAMS: list[str] = [
    "Equipe A — Oficina Norte",
    "Equipe B — Oficina Norte",
    "Equipe C — Oficina Sul",
]


@dataclass(frozen=True)
class CauseProfile:
    weight: float
    minutes: tuple[int, int]  # episode duration
    repair_hours: tuple[float, float]
    sensors: dict[str, float]  # base contribution per sensor
    parts: list[tuple[str, int]]


# None = false alarm (no failure found by the technician).
CAUSES: dict[str | None, CauseProfile] = {
    "air_leak": CauseProfile(
        0.32,
        (25, 240),
        (1.5, 6.0),
        {"TP3": 0.34, "Motor_current": 0.22, "DV_pressure": 0.15, "COMP": 0.12},
        [("Kit de anéis de vedação", 1), ("Mangueira pneumática 1/2 pol.", 1)],
    ),
    "oil_system": CauseProfile(
        0.14,
        (60, 600),
        (3.0, 8.0),
        {"Oil_temperature": 0.38, "Motor_current": 0.20, "TP2": 0.14, "H1": 0.08},
        [("Retentor do eixo", 1), ("Óleo do compressor (L)", 4), ("Filtro de óleo", 1)],
    ),
    "air_drier": CauseProfile(
        0.14,
        (15, 120),
        (2.0, 5.0),
        {"Towers": 0.31, "DV_pressure": 0.24, "TP3": 0.18, "Reservoirs": 0.10},
        [("Válvula de purga do secador", 1), ("Dessecante (kg)", 6)],
    ),
    "valve": CauseProfile(
        0.12,
        (10, 90),
        (1.0, 3.0),
        {"DV_eletric": 0.33, "COMP": 0.27, "TP2": 0.16, "MPG": 0.09},
        [("Reparo da válvula de admissão", 1)],
    ),
    "electrical_motor": CauseProfile(
        0.08,
        (5, 45),
        (2.0, 6.0),
        {"Motor_current": 0.46, "COMP": 0.14, "Oil_temperature": 0.11},
        [("Contator do motor", 1)],
    ),
    "sensor_fault": CauseProfile(
        0.06,
        (5, 30),
        (0.5, 2.0),
        {"H1": 0.41, "TP2": 0.19, "DV_pressure": 0.12},
        [("Transdutor de pressão", 1)],
    ),
    None: CauseProfile(
        0.14,
        (3, 20),
        (0.3, 1.0),
        {"TP2": 0.16, "Motor_current": 0.12, "TP3": 0.10},
        [],
    ),
}


def _contributions(
    rng: random.Random, base: dict[str, float]
) -> list[dict[str, str | float]]:
    scored = sorted(
        ((s, round(v * rng.uniform(0.8, 1.2), 3)) for s, v in base.items()),
        key=lambda sv: -sv[1],
    )
    return [{"sensor": s, "contribution": c} for s, c in scored]


async def _ensure_assets() -> None:
    async with AsyncSessionFactory() as db:
        main = await db.get(Asset, MAIN_ASSET)
        if main is None:
            raise SystemExit(f"{MAIN_ASSET} not registered — start the api once first.")
        for asset_id, name in FLEET.items():
            if await db.get(Asset, asset_id) is None:
                db.add(
                    Asset(
                        id=asset_id,
                        name=name,
                        installation=main.installation,
                        system=main.system,
                        equipment_unit=main.equipment_unit,
                        subunit=main.subunit,
                        manufacturer_model=main.manufacturer_model,
                        criticality=main.criticality,
                    )
                )
        await db.commit()


async def seed() -> None:
    rng = random.Random(SEED)
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    await _ensure_assets()

    async with AsyncSessionFactory() as db:
        await db.execute(delete(RootCauseAnalysis))
        await db.execute(delete(WorkOrder))
        await db.execute(delete(FailureEvent))

        starts = sorted(
            now - timedelta(minutes=rng.randint(30, DAYS * 24 * 60))
            for _ in range(N_EVENTS)
        )
        for i, started in enumerate(starts):
            asset_id = rng.choices(
                list(ASSET_WEIGHTS), weights=list(ASSET_WEIGHTS.values())
            )[0]
            cause = rng.choices(
                list(CAUSES), weights=[p.weight for p in CAUSES.values()]
            )[0]
            profile = CAUSES[cause]
            minutes = rng.randint(*profile.minutes)
            is_recent = i >= N_EVENTS - 2  # the two newest stay open
            peak = round(rng.uniform(0.72, 0.99), 3)
            if cause is None:
                peak = round(rng.uniform(0.70, 0.86), 3)
            event = FailureEvent(
                asset_id=asset_id,
                started_at=started,
                ended_at=None if is_recent else started + timedelta(minutes=minutes),
                status="open" if is_recent else "closed",
                severity=severity_for(peak, THRESHOLD, CRITICAL_AT),
                peak_probability=peak,
                mean_probability=round(rng.uniform(THRESHOLD, peak), 3),
                n_samples=minutes * 60,
                model_name=MODEL_NAME,
                threshold=THRESHOLD,
                created_at=started,
                updated_at=started + timedelta(minutes=minutes),
            )
            db.add(event)
            await db.flush()

            db.add(
                RootCauseAnalysis(
                    event_id=event.id,
                    created_at=started,
                    method="sensor_occlusion",
                    sensor_contributions=_contributions(rng, profile.sensors),
                    references=[],
                )
            )

            if rng.random() < 0.15 and not is_recent:
                continue  # some events never got a work order
            opened = started + timedelta(minutes=rng.randint(5, 40))
            age_days = (now - started).days
            if is_recent:
                status = "open"
            elif age_days < 2:
                status = "in_progress"
            elif rng.random() < 0.05:
                status = "cancelled"
            else:
                status = "closed"
            work_started = opened + timedelta(minutes=rng.randint(10, 90))
            closed = work_started + timedelta(hours=rng.uniform(*profile.repair_hours))
            done = status == "closed"
            db.add(
                WorkOrder(
                    event_id=event.id,
                    asset_id=asset_id,
                    status=status,
                    assignee=rng.choice(TEAMS),
                    opened_at=opened,
                    started_at=work_started if status != "open" else None,
                    closed_at=closed if done or status == "cancelled" else None,
                    parts=(
                        [{"part": p, "quantity": q} for p, q in profile.parts]
                        if done
                        else []
                    ),
                    confirmed_cause=cause if done else None,
                    is_false_alarm=(cause is None) if done else None,
                    notes=DEMO_NOTE,
                )
            )

        await db.commit()
        total = len((await db.execute(select(FailureEvent.id))).all())
        print(f"seeded {total} demo events")


if __name__ == "__main__":
    asyncio.run(seed())
