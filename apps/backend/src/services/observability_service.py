"""
Observability service — RNF-77.

Consulta o Prometheus (rede Docker interna, nunca exposto ao browser — ver
RELATORIO-RNF-76-RNF-77.md §Fase 9) via sua HTTP API (`/api/v1/query`,
formato nativo do Prometheus) e traduz a taxa de erro 5xx num veredito
NORMAL/WARNING/CRITICAL para o Dashboard.

Por que um endpoint interno em vez do frontend falar direto com o
Prometheus: o Prometheus não tem autenticação própria por padrão — expô-lo
ao browser deixaria QUALQUER PromQL arbitrário acessível publicamente
(inclusive métricas de outros jobs, se algum dia existirem). Este endpoint
só expõe o resultado JÁ AGREGADO de UMA query fixa, sem repassar PromQL
arbitrário do cliente.

Definição de "taxa de erro" (RNF-77 Fase 4) — deliberadamente HTTP 5xx, não
4xx: 4xx (422 payload inválido, 429 rate limit) é o cliente "errando",
não a aplicação falhando — misturar os dois inflaria a métrica com tráfego
de rate-limit saudável (RNF-19) e mascararia problemas reais do servidor.
Erro de inferência (predictiq_inference_total{status="error"}) é OUTRO
sinal, deliberadamente separado (ver src/core/metrics.py) — não entra
nesta conta porque nem toda inferência passa por HTTP (pipeline contínuo).
"""

from __future__ import annotations

import math

import httpx
import structlog

from src.core.config import settings
from src.schemas.observability import ErrorRateResponse, ErrorRateStatus

log = structlog.get_logger(__name__)

_WINDOW = "5m"
_QUERY = (
    f'sum(rate(http_requests_total{{status=~"5.."}}[{_WINDOW}])) '
    f"/ "
    f"sum(rate(http_requests_total[{_WINDOW}]))"
)
_QUERY_TIMEOUT_SECONDS = 5.0


def _classify(error_rate: float) -> ErrorRateStatus:
    if error_rate >= settings.error_rate_critical_threshold:
        return "CRITICAL"
    if error_rate >= settings.error_rate_warning_threshold:
        return "WARNING"
    return "NORMAL"


async def get_error_rate_status() -> ErrorRateResponse:
    """
    Consulta o Prometheus e devolve o veredito atual.

    Nunca levanta exceção — qualquer falha (rede, parsing, Prometheus fora
    do ar) degrada para `status="NORMAL"`, `error_rate=0.0`,
    `prometheus_reachable=False`: o widget do Dashboard não pode travar a
    página só porque a stack de observabilidade está indisponível, e um
    dado ausente nunca deve virar um alerta falso.
    """
    try:
        async with httpx.AsyncClient(timeout=_QUERY_TIMEOUT_SECONDS) as client:
            response = await client.get(
                f"{settings.prometheus_url}/api/v1/query",
                params={"query": _QUERY},
            )
            response.raise_for_status()
            body = response.json()

        if body.get("status") != "success":
            raise ValueError(
                f"Prometheus query status != success: {body.get('status')}"
            )

        result = body["data"]["result"]
        if not result:
            # Sem série ainda (processo recém-subido, zero tráfego até agora
            # dentro da janela) — 0 requisições é 0% de erro, não "sem dado".
            error_rate = 0.0
        else:
            raw_value = float(result[0]["value"][1])
            # rate()/rate() com denominador 0 (nenhuma requisição na janela)
            # devolve NaN no Prometheus — trata como 0 erro, nunca propaga
            # NaN (JSON não tem NaN; um `status=CRITICAL` espúrio seria pior).
            error_rate = 0.0 if math.isnan(raw_value) else raw_value

        return ErrorRateResponse(
            status=_classify(error_rate),
            error_rate=round(error_rate, 6),
            window=_WINDOW,
            threshold_warning=settings.error_rate_warning_threshold,
            threshold_critical=settings.error_rate_critical_threshold,
            prometheus_reachable=True,
        )

    except Exception as exc:  # noqa: BLE001 — degradação segura, nunca propaga
        log.warning("observability_prometheus_query_failed", error=str(exc))
        return ErrorRateResponse(
            status="NORMAL",
            error_rate=0.0,
            window=_WINDOW,
            threshold_warning=settings.error_rate_warning_threshold,
            threshold_critical=settings.error_rate_critical_threshold,
            prometheus_reachable=False,
        )
