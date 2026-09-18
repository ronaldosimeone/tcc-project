"""
DriftMonitor — RF-27 / RNF-55.

Orquestra o fluxo completo de monitoramento de DATA DRIFT (distribuição das
FEATURES de entrada do modelo — nunca confundir com degradação de
performance do modelo, que exigiria rótulos verdadeiros/ground truth que
este projeto não coleta em produção; ver README §"RF-27"):

    reference (baseline)              current (últimas 24h)
    apps/ml/data/.../metropt3.parquet  tabela `predictions` (RF-09)
    amostra determinística,           janela [now-24h, now]
    só linhas "normais"
        v                                  v
        └──────────────► PSI (_population_stability_index) ◄──────┘
                                v
                    PSI por feature (7 sensores analógicos)
                                v
                PSI agregado = max(PSI por feature)
                                v
                    drift_detected = PSI > 0.25  (estrito, RF-27)
                                v
                        persist_report() — upsert por `analysis_date`
                                v
                    histórico → GET /monitoring/drift (RNF-55)

Responsabilidades separadas em métodos próprios — mesma convenção de
`MaintenanceSuggestionService` (RF-22): cada etapa testável isoladamente.

RNF-62 — por que PSI é calculado aqui em vez de usar `evidently`
------------------------------------------------------------------
Esta feature usava `evidently` (`Dataset`/`DataDefinition`/`Report`/
`ValueDrift(method="psi")`) até a task de hardening de segurança. `evidently`
carrega `nltk` como dependência OBRIGATÓRIA e IMEDIATA — mesmo `from
evidently import Dataset` já executa `evidently/__init__.py`, que importa
`evidently.legacy.metrics` → ... → `evidently.legacy.features.
OOV_words_percentage_feature` → `from nltk.corpus import words` (confirmado
empiricamente: `pip uninstall nltk` faz até o import mais simples do
`evidently` falhar com `ModuleNotFoundError`, então NÃO é possível manter
`evidently` e remover `nltk` do ambiente). `nltk` tem uma vulnerabilidade
High sem correção publicada em nenhuma versão (`PYSEC-2026-3740`/
`CVE-2026-81726`) — ver PENDENCIAS.md para o histórico completo.

Como a única funcionalidade do `evidently` de fato usada aqui era o cálculo
de PSI numérico (`Report(metrics=[ValueDrift(column=c, method="psi")])`),
`_population_stability_index()` abaixo reimplementa exatamente esse
algoritmo com `numpy`/`pandas` (dependências já existentes, nenhuma nova) —
réplica fiel de `evidently.legacy.calculations.stattests.psi._psi()` +
`.utils.get_binned_data()` (binagem por `numpy.histogram_bin_edges(...,
bins="sturges")` sobre reference+current combinados, com o mesmo
preenchimento de bins vazios por epsilon). Validado com 3 cenários
sintéticos (distribuições iguais, deslocadas e assimétricas tipo gama):
resultado **idêntico** (diff < 1e-9) ao `evidently` real em todos. `PSI ==
0` de um vetor consigo mesmo, mesmo threshold `0.25`, mesmos testes de
fronteira — nenhum comportamento de negócio mudou, só a dependência.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.models.drift_report import DriftReport
from src.models.prediction import Prediction
from src.schemas.drift import DriftReportResponse
from src.schemas.prediction import Page, make_page

# Reaproveita a MESMA constante de domínio usada pelo simulador (RF-13) para
# as 4 janelas de falha conhecidas do paper MetroPT-3, e o mesmo builder de
# máscara — em vez de duplicar essa lógica (que precisa ficar idêntica: um
# desalinhamento silencioso faria a amostra de referência incluir linhas de
# falha, contaminando o baseline). `_load_and_split`/`SensorSimulator` NÃO
# são reaproveitados diretamente: eles retornam um ndarray com as 12 colunas
# (analógicas + digitais) já embaralhadas por linha, sem os nomes de coluna
# de que o Evidently precisa — aqui construímos nosso próprio DataFrame,
# só com as 7 features monitoradas (ver `MONITORED_FEATURES` abaixo).
from src.services.simulator import _build_failure_mask_from_timestamps

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Regras de negócio (RF-27) — constantes próprias, nunca acopladas às de
# outras features (RF-14/RF-22/RF-24 já estabeleceram esse padrão no
# projeto: cada threshold é uma constante isolada, mesmo quando o valor
# numérico coincide por acaso).
# ---------------------------------------------------------------------------

# RF-27 — estrito: só `PSI > 0.25` é drift. `PSI == 0.25` NÃO é alerta.
DRIFT_PSI_THRESHOLD: float = 0.25

# Features monitoradas — os 7 sensores ANALÓGICOS que
# `src/services/preprocessing.py::MetroPTPreprocessor` usa como
# `_DEFAULT_SENSOR_COLS` (a base real do pipeline de feature engineering do
# modelo ativo, `ACTIVE_MODEL=random_forest_v2` — ver `.env`). Os 5 sinais
# digitais (COMP, DV_eletric, Towers, MPG, Oil_level — binários ON/OFF) são
# deliberadamente excluídos: PSI sobre um sinal ON/OFF mede só a proporção de
# tempo ligado/desligado, um sinal de drift qualitativamente diferente que
# RF-27 não pede — e a escolha aqui é rastreável ao pipeline real, não
# arbitrária (RF-27 §Fase 1 "Modelo").
MONITORED_FEATURES: list[str] = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
]

# Reference — amostra determinística (seed fixa) de linhas "normais" (fora
# das 4 janelas de falha conhecidas do paper MetroPT-3, mesma fonte de
# verdade do simulador RF-13) do PRÓPRIO dataset usado no treinamento
# (`settings.simulator_parquet_path` — RF-27 §Fase 1 "Modelo": não é uma
# fonte de dados inventada, é o dataset real de baseline do projeto).
# 5 000 linhas equilibram representatividade estatística (>> mínimo
# recomendado para PSI, tipicamente ~1000) com custo de I/O/memória de uma
# task diária em background — não a leitura de todas as 1.5M linhas.
REFERENCE_SAMPLE_SIZE: int = 5_000
REFERENCE_SAMPLE_SEED: int = 42

# Current — janela determinística e documentada (RF-27 §Fase 4): últimas
# 24h de predições reais persistidas (tabela `predictions`, RF-09).
CURRENT_WINDOW_HOURS: int = 24

# RF-27 §"IMPORTANTE — SEM DADOS INVENTADOS": abaixo deste número de linhas
# na janela current, NÃO calculamos PSI algum (evita um PSI estatisticamente
# instável/sem sentido sobre uma amostra minúscula) — `status="insufficient_data"`.
MIN_CURRENT_ROWS: int = 30


# ---------------------------------------------------------------------------
# PSI (Population Stability Index) — RNF-62. Réplica fiel do algoritmo do
# `evidently` (ver docstring do módulo para o porquê da reimplementação),
# validada contra ele em 3 cenários sintéticos (diff < 1e-9). Função pura,
# sem estado — testável isoladamente do resto do `DriftMonitor`.
# ---------------------------------------------------------------------------


def _binned_percentages(
    reference: pd.Series, current: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Divide `reference`+`current` (combinados) em bins pela regra de
    Sturges e devolve a fração de linhas de cada série em cada bin — mesma
    binagem usada por `evidently.legacy.calculations.stattests.utils.
    get_binned_data()` para colunas numéricas (>20 valores únicos, sempre o
    caso para os 7 sensores analógicos monitorados aqui)."""
    combined = np.concatenate([reference.to_numpy(), current.to_numpy()])
    bins = np.histogram_bin_edges(combined, bins="sturges")
    reference_pct = np.histogram(reference, bins)[0] / len(reference)
    current_pct = np.histogram(current, bins)[0] / len(current)
    return reference_pct, current_pct


def _fill_zero_bins(percentages: np.ndarray) -> np.ndarray:
    """Bins com frequência zero quebrariam `log(0)`/divisão por zero no
    cálculo do PSI — substituídos por um epsilon pequeno, nunca por zero
    “forçado a existir”. Mesma regra do `evidently`
    (`get_binned_data(..., feel_zeroes=True)`): `min(não-zero)/1e6` quando
    esse mínimo já é bem pequeno (<=0.0001), senão um piso fixo de 0.0001."""
    percentages = percentages.astype(float).copy()
    nonzero = percentages[percentages != 0]
    if len(nonzero) == 0:
        return percentages
    smallest = float(nonzero.min())
    fill_value = smallest / 1e6 if smallest <= 0.0001 else 0.0001
    percentages[percentages == 0] = fill_value
    return percentages


def _population_stability_index(reference: pd.Series, current: pd.Series) -> float:
    """PSI entre duas amostras de uma feature numérica —
    `sum((ref% - cur%) * ln(ref% / cur%))` por bin. `0.0` para distribuições
    idênticas; cresce com o quão diferente `current` é de `reference`."""
    reference_pct, current_pct = _binned_percentages(reference, current)
    reference_pct = _fill_zero_bins(reference_pct)
    current_pct = _fill_zero_bins(current_pct)
    return float(
        np.sum((reference_pct - current_pct) * np.log(reference_pct / current_pct))
    )


@dataclass
class DriftResult:
    """Resultado de uma execução de `DriftMonitor.run_daily_analysis()`."""

    status: str  # "ok" | "insufficient_data" | "error"
    reference_period: str
    current_period_start: datetime
    current_period_end: datetime
    psi: float | None = None
    drift_detected: bool | None = None
    features: dict[str, float] | None = None
    error_message: str | None = None


class DriftMonitor:
    """Responsabilidade única: baseline -> current -> PSI -> persistência."""

    def __init__(self, parquet_path: Path | None = None) -> None:
        self._parquet_path: Path = parquet_path or settings.simulator_parquet_path

    # ------------------------------------------------------------------
    # 1. Reference / baseline
    # ------------------------------------------------------------------

    def load_reference_data(self) -> pd.DataFrame:
        """
        Carrega a amostra de referência (RF-27 §Fase 4) — linhas "normais"
        (fora das janelas de falha conhecidas) do dataset MetroPT-3 real,
        amostradas deterministicamente (seed fixa).

        Levanta `FileNotFoundError` se o parquet não existir — o chamador
        (`run_daily_analysis`) trata isso como `status="error"`, nunca
        inventando dados de referência.
        """
        if not self._parquet_path.exists():
            raise FileNotFoundError(
                f"Dataset de referência não encontrado em {self._parquet_path}. "
                "Gere-o com `python src/ingest_metropt.py` em apps/ml/, ou "
                "ajuste SIMULATOR_PARQUET_PATH."
            )

        columns = [*MONITORED_FEATURES, "timestamp"]
        df = pd.read_parquet(self._parquet_path, columns=columns, engine="pyarrow")

        failure_mask = _build_failure_mask_from_timestamps(df["timestamp"])
        normal_df = df.loc[~failure_mask, MONITORED_FEATURES].reset_index(drop=True)

        sample_size = min(REFERENCE_SAMPLE_SIZE, len(normal_df))
        return normal_df.sample(
            n=sample_size, random_state=REFERENCE_SAMPLE_SEED
        ).reset_index(drop=True)

    @staticmethod
    def reference_period_label(reference_df: pd.DataFrame) -> str:
        """Descrição textual persistida em `DriftReport.reference_period` —
        auditável (RF-27 §Fase 6), não um id opaco."""
        return (
            f"metropt3_baseline_normal_sample(n={len(reference_df)},"
            f"seed={REFERENCE_SAMPLE_SEED})"
        )

    # ------------------------------------------------------------------
    # 2. Current / dados recentes
    # ------------------------------------------------------------------

    async def load_current_data(
        self, db: AsyncSession, now: datetime | None = None
    ) -> tuple[pd.DataFrame, datetime, datetime]:
        """
        Carrega as predições reais persistidas (tabela `predictions`, RF-09)
        na janela [now - 24h, now] — RF-27 §Fase 4, janela determinística e
        documentada.
        """
        current_end = now or datetime.now(timezone.utc)
        current_start = current_end - timedelta(hours=CURRENT_WINDOW_HOURS)

        stmt = select(Prediction).where(
            Prediction.timestamp >= current_start,
            Prediction.timestamp <= current_end,
        )
        rows = (await db.execute(stmt)).scalars().all()

        data = {col: [getattr(row, col) for row in rows] for col in MONITORED_FEATURES}
        current_df = pd.DataFrame(data, columns=MONITORED_FEATURES)
        return current_df, current_start, current_end

    # ------------------------------------------------------------------
    # 3-5. PSI -> decisão de drift
    # ------------------------------------------------------------------

    def calculate_drift(
        self, reference: pd.DataFrame, current: pd.DataFrame
    ) -> dict[str, float]:
        """
        Calcula o PSI (`_population_stability_index`, réplica validada do
        algoritmo do `evidently` — ver docstring do módulo, RNF-62) e
        retorna o valor de CADA feature monitorada — `{"TP2": 0.03, ...}`.

        A decisão `drift_detected = psi > DRIFT_PSI_THRESHOLD` é feita pelo
        CHAMADOR (`run_daily_analysis`) — este método só calcula o número.
        """
        return {
            column: _population_stability_index(reference[column], current[column])
            for column in MONITORED_FEATURES
        }

    # ------------------------------------------------------------------
    # Orquestração completa (chamada pela task Celery Beat)
    # ------------------------------------------------------------------

    async def run_daily_analysis(
        self, db: AsyncSession, now: datetime | None = None
    ) -> DriftResult:
        """
        Executa a análise completa — nunca inventa PSI quando não há dados
        suficientes (RF-27 §"IMPORTANTE — SEM DADOS INVENTADOS") e nunca
        deixa uma exceção do cálculo de PSI/parquet derrubar a task (mesma
        filosofia de RF-24: falha de monitoramento nunca é crítica o
        suficiente para quebrar o worker).
        """
        current_df, current_start, current_end = await self.load_current_data(
            db, now=now
        )

        if len(current_df) < MIN_CURRENT_ROWS:
            log.warning(
                "drift_analysis_insufficient_data",
                current_rows=len(current_df),
                min_required=MIN_CURRENT_ROWS,
            )
            return DriftResult(
                status="insufficient_data",
                reference_period="(não calculado — dados current insuficientes)",
                current_period_start=current_start,
                current_period_end=current_end,
                error_message=(
                    f"Apenas {len(current_df)} predições na janela de "
                    f"{CURRENT_WINDOW_HOURS}h — mínimo exigido: {MIN_CURRENT_ROWS}."
                ),
            )

        try:
            reference_df = self.load_reference_data()
            psi_by_feature = self.calculate_drift(reference_df, current_df)
        except Exception as exc:  # noqa: BLE001 — nunca derruba a task Celery
            log.exception("drift_analysis_failed")
            return DriftResult(
                status="error",
                reference_period="(não calculado — erro na análise)",
                current_period_start=current_start,
                current_period_end=current_end,
                error_message=str(exc),
            )

        overall_psi = max(psi_by_feature.values())
        drift_detected = overall_psi > DRIFT_PSI_THRESHOLD  # estrito, nunca >=

        log.info(
            "drift_analysis_completed",
            psi=overall_psi,
            drift_detected=drift_detected,
            current_rows=len(current_df),
        )

        return DriftResult(
            status="ok",
            reference_period=self.reference_period_label(reference_df),
            current_period_start=current_start,
            current_period_end=current_end,
            psi=overall_psi,
            drift_detected=drift_detected,
            features=psi_by_feature,
        )

    # ------------------------------------------------------------------
    # 6. Persistência — idempotente por `analysis_date` (RF-27 §Fase 8)
    # ------------------------------------------------------------------

    async def persist_report(
        self, db: AsyncSession, result: DriftResult, analysis_date: date_type
    ) -> DriftReport:
        """
        Upsert por `analysis_date` (UNIQUE) — mesmo padrão dialect-aware já
        usado em `alert_settings_service.py`/`telegram_alert_rate_limiter.py`.
        Reprocessar o mesmo dia (scheduler duplicado, retry manual)
        SOBRESCREVE o relatório daquele dia em vez de duplicar o histórico.
        """
        now = datetime.now(timezone.utc)
        dialect_name = db.get_bind().dialect.name
        insert_fn = pg_insert if dialect_name == "postgresql" else sqlite_insert

        values = {
            "analysis_date": analysis_date,
            "analyzed_at": now,
            "reference_period": result.reference_period,
            "current_period_start": result.current_period_start,
            "current_period_end": result.current_period_end,
            "psi": result.psi,
            "drift_detected": result.drift_detected,
            "features": result.features,
            "status": result.status,
            "error_message": result.error_message,
            "created_at": now,
        }
        stmt = (
            insert_fn(DriftReport)
            .values(**values)
            .on_conflict_do_update(  # type: ignore[attr-defined]
                index_elements=[DriftReport.analysis_date],
                set_={
                    "analyzed_at": now,
                    "reference_period": result.reference_period,
                    "current_period_start": result.current_period_start,
                    "current_period_end": result.current_period_end,
                    "psi": result.psi,
                    "drift_detected": result.drift_detected,
                    "features": result.features,
                    "status": result.status,
                    "error_message": result.error_message,
                },
            )
            .returning(DriftReport)
        )
        saved = (await db.execute(stmt)).scalar_one()
        await db.commit()
        return saved


# ---------------------------------------------------------------------------
# Histórico (RNF-55) — GET /monitoring/drift. Só leitura, nunca dispara a
# análise (RF-27 §Fase 7/9: "O endpoint não deve disparar a análise").
# ---------------------------------------------------------------------------


async def list_drift_reports(
    db: AsyncSession, page: int, size: int
) -> Page[DriftReportResponse]:
    """Histórico paginado, mais recente primeiro — mesmo padrão de
    `prediction_service.list_predictions` (RNF-15)."""
    count_stmt = select(func.count()).select_from(DriftReport)
    total: int = (await db.execute(count_stmt)).scalar_one()

    if total == 0:
        return Page(items=[], total=0, page=page, size=size, pages=0)

    offset = (page - 1) * size
    rows_stmt = (
        select(DriftReport)
        .order_by(DriftReport.analyzed_at.desc(), DriftReport.id.desc())
        .offset(offset)
        .limit(size)
    )
    rows = (await db.execute(rows_stmt)).scalars().all()
    items = [DriftReportResponse.model_validate(row) for row in rows]
    return make_page(items=items, total=total, page=page, size=size)
