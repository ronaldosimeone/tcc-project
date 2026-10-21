"""
Application settings managed via pydantic-settings.

All values can be overridden through environment variables or a .env file
placed at the project root.  Variable names are case-insensitive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolved at import time: apps/backend/src/core/config.py → parents[3] = apps/
_ML_MODELS: Path = Path(__file__).resolve().parents[3] / "ml" / "models"


class Settings(BaseSettings):
    """Typed application configuration backed by environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Project metadata ──────────────────────────────────────────────────
    project_name: str = Field(default="Predictive Maintenance API")
    version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)

    # ── Database ──────────────────────────────────────────────────────────
    # Expected format: postgresql+asyncpg://user:password@host:port/dbname
    postgres_url: PostgresDsn = Field(
        default=PostgresDsn(
            "postgresql+asyncpg://postgres:postgres@localhost:5432/tcc_db"
        ),
        alias="DATABASE_URL",
    )

    # RNF-37 — pool tunável via .env para o load test (locust_streaming.py /
    # README §14.7). Defaults preservam o comportamento anterior (antes
    # hardcoded 10/20 diretamente em database.py).
    db_pool_size: int = Field(
        default=10,
        alias="DB_POOL_SIZE",
        description="SQLAlchemy async engine pool_size (conexões persistentes por worker).",
    )
    db_max_overflow: int = Field(
        default=20,
        alias="DB_MAX_OVERFLOW",
        description="SQLAlchemy async engine max_overflow (conexões extras sob pico).",
    )

    # ── Ollama (local LLM) — RF-22 / RNF-46 ─────────────────────────────────
    ollama_base_url: str = Field(
        default="http://host.docker.internal:11434",
        alias="OLLAMA_BASE_URL",
    )
    # Modelo usado por MaintenanceSuggestionService (src/services/ollama_client.py).
    # Default = Llama 3.2 3B (CLAUDE.md §4) — configurável sem redeploy.
    ollama_model: str = Field(
        default="llama3.2:3b",
        alias="OLLAMA_MODEL",
    )
    # Timeout do cliente HTTP do Ollama — generoso porque um Llama 3.2 3B
    # rodando em CPU pode levar dezenas de segundos para um plano completo.
    ollama_client_timeout_seconds: float = Field(
        default=120.0,
        alias="OLLAMA_CLIENT_TIMEOUT_SECONDS",
    )

    # RF-19 / RNF-43 — endereço do serviço MCP standalone (apps/mcp-server),
    # container e processo próprios (ver docker-compose.yml::mcp-server).
    # Consumida desde a RF-22 por MaintenanceSuggestionService
    # (src/services/mcp_client.py) — chama a tool `search_maintenance_manual`
    # via streamable-http (SDK oficial `mcp`, ver requirements.txt).
    mcp_server_url: str = Field(
        default="http://mcp-server:8100",
        alias="MCP_SERVER_URL",
    )
    # Timeout do cliente MCP — generoso porque a primeira chamada após o
    # mcp-server subir aciona o carregamento do SentenceTransformer
    # (~segundos a ~30s a frio, ver apps/mcp-server/semantic_search.py).
    mcp_client_timeout_seconds: float = Field(
        default=60.0,
        alias="MCP_CLIENT_TIMEOUT_SECONDS",
    )

    # RN-04 — fundamentação do plano de manutenção. Só trechos com cosine
    # >= τ entram no prompt. Default igual ao corte do próprio mcp-server
    # (0.6); o valor definitivo é calibrado no benchmark anotado (RF-36).
    # "block": sem trecho >= τ o LLM não é chamado (plano de "Limitações"
    # fixo); "flag": o LLM é chamado e a resposta sai marcada `ungrounded`.
    rag_grounding_threshold: float = Field(
        default=0.6,
        ge=-1.0,
        le=1.0,
        alias="RAG_GROUNDING_THRESHOLD",
    )
    rag_ungrounded_policy: Literal["block", "flag"] = Field(
        default="block",
        alias="RAG_UNGROUNDED_POLICY",
    )

    # ── Histórico / frota ────────────────────────────────────────────────
    # RN-03 aplicado online: evento abre com k amostras >= limiar e fecha
    # com k amostras abaixo; pico >= critical_at -> severidade "critico".
    # None (padrão) = limiar de decisão do modelo que gerou a predição (card,
    # escolhido sem vazamento). Um valor fixo só sobrepõe para experimentos —
    # com limiares OOF como 0,90/0,985 um 0,7 fixo abriria eventos falsos.
    event_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, alias="EVENT_THRESHOLD"
    )
    event_k_open: int = Field(default=6, ge=1, alias="EVENT_K_OPEN")
    event_k_close: int = Field(default=6, ge=1, alias="EVENT_K_CLOSE")
    event_critical_at: float = Field(
        default=0.9, ge=0.0, le=1.0, alias="EVENT_CRITICAL_AT"
    )
    # Gera o plano RAG (MCP + Ollama) em background quando um evento abre.
    rca_llm_enabled: bool = Field(default=True, alias="RCA_LLM_ENABLED")
    # Ativos simulados além do APU-Trem-042 (0 = frota desligada).
    fleet_size: int = Field(default=0, ge=0, le=1000, alias="FLEET_SIZE")
    fleet_noise: float = Field(default=0.02, ge=0.0, le=1.0, alias="FLEET_NOISE")
    fleet_seed: int = Field(default=42, alias="FLEET_SEED")

    # ── Autenticação / RBAC / auditoria ──────────────────────────────────
    # false (default) mantém o comportamento anterior (sem login); as ações
    # ainda entram na trilha de auditoria como "anonymous (auth disabled)".
    # Em produção: AUTH_ENABLED=true + JWT_SECRET forte (a app recusa subir
    # com o placeholder).
    auth_enabled: bool = Field(default=False, alias="AUTH_ENABLED")
    jwt_secret: str = Field(default="dev-only-jwt-secret-change-me", alias="JWT_SECRET")
    jwt_expire_minutes: int = Field(
        default=480, ge=5, le=7 * 24 * 60, alias="JWT_EXPIRE_MINUTES"
    )
    # Cria o primeiro admin no startup se não existir nenhum usuário.
    bootstrap_admin_username: str | None = Field(
        default=None, alias="BOOTSTRAP_ADMIN_USERNAME"
    )
    bootstrap_admin_password: str | None = Field(
        default=None, alias="BOOTSTRAP_ADMIN_PASSWORD"
    )

    # ── Telemetria industrial (MQTT) ─────────────────────────────────────
    # "inprocess" (default): a frota simulada é processada no próprio
    # processo. "mqtt": a frota PUBLICA no broker e o ingestor ASSINA
    # <prefix>/+/telemetry — o mesmo caminho de um gateway de campo real.
    telemetry_transport: Literal["inprocess", "mqtt", "opcua"] = Field(
        default="inprocess", alias="TELEMETRY_TRANSPORT"
    )
    mqtt_host: str = Field(default="mqtt", alias="MQTT_HOST")
    mqtt_port: int = Field(default=1883, alias="MQTT_PORT")
    mqtt_username: str | None = Field(default=None, alias="MQTT_USERNAME")
    mqtt_password: str | None = Field(default=None, alias="MQTT_PASSWORD")
    mqtt_topic_prefix: str = Field(default="predictiq/fleet", alias="MQTT_TOPIC_PREFIX")

    # "opcua": a frota escreve num gateway OPC UA (simulado, dentro do
    # backend) e o ingestor ASSINA os nós Sequence de cada ativo. Com
    # OPCUA_SIMULATED_GATEWAY=false o ingestor conecta a um gateway real.
    opcua_endpoint: str = Field(
        default="opc.tcp://0.0.0.0:4840/predictiq/", alias="OPCUA_ENDPOINT"
    )
    opcua_simulated_gateway: bool = Field(default=True, alias="OPCUA_SIMULATED_GATEWAY")
    opcua_username: str | None = Field(default=None, alias="OPCUA_USERNAME")
    opcua_password: str | None = Field(default=None, alias="OPCUA_PASSWORD")

    # ── Champion / challenger (shadow + canary) ──────────────────────────
    # CHALLENGER_MODEL definido → o desafiante pontua em SHADOW toda leitura
    # do ativo principal (sem agir) e age nos ativos da fração canary.
    challenger_model: str | None = Field(default=None, alias="CHALLENGER_MODEL")
    canary_fraction: float = Field(default=0.0, ge=0.0, le=1.0, alias="CANARY_FRACTION")

    # ── OpenTelemetry (tracing distribuído) ──────────────────────────────
    otel_enabled: bool = Field(default=False, alias="OTEL_ENABLED")
    otel_service_name: str = Field(default="predictiq-api", alias="OTEL_SERVICE_NAME")
    otel_exporter_otlp_endpoint: str = Field(
        default="http://jaeger:4318", alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )

    # ── Notificações críticas via Telegram — RF-24 / RNF-48 ─────────────────
    # `None` por padrão — sem token/chat_id configurados, TelegramNotificationAdapter
    # trata como configuração ausente (RF-24 §11 N/O) e nunca tenta enviar.
    telegram_bot_token: str | None = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str | None = Field(default=None, alias="TELEGRAM_CHAT_ID")
    telegram_api_base_url: str = Field(
        default="https://api.telegram.org",
        alias="TELEGRAM_API_BASE_URL",
    )
    telegram_client_timeout_seconds: float = Field(
        default=10.0,
        alias="TELEGRAM_CLIENT_TIMEOUT_SECONDS",
    )
    # URL absoluta e oficial do Dashboard — nunca aceita de payload/predição/LLM
    # (RF-24 §4, evita open redirect). Default aponta para o Nginx local.
    dashboard_url: str = Field(default="http://localhost", alias="DASHBOARD_URL")
    # Fuso (IANA) usado para exibir o horário nas mensagens de alerta
    # (Telegram/e-mail). O pipeline continua em UTC — só a apresentação muda.
    notification_timezone: str = Field(
        default="America/Sao_Paulo", alias="NOTIFICATION_TIMEZONE"
    )
    # 15 minutos — RF-24, TTL do rate limit por equipamento (não confundir
    # com MAINTENANCE_SUGGESTION_THRESHOLD/CRITICAL_FAILURE_THRESHOLD, que
    # são constantes de código, não configuráveis por env).
    critical_failure_rate_limit_seconds: int = Field(
        default=900,
        alias="CRITICAL_FAILURE_RATE_LIMIT_SECONDS",
    )
    # O pipeline real (InferencePipelineService) hoje simula um único ativo
    # (MetroPT-3) — reaproveita o mesmo ID já usado na rota do frontend
    # (`/sensors/APU-Trem-042`, ver Sidebar) em vez de inventar um novo.
    default_equipment_id: str = Field(
        default="APU-Trem-042", alias="DEFAULT_EQUIPMENT_ID"
    )
    default_equipment_name: str = Field(
        default="Compressor de Ar Industrial (MetroPT-3)",
        alias="DEFAULT_EQUIPMENT_NAME",
    )

    # ── Notificações críticas via e-mail (Resend) — RF-25 / RNF-49 ──────────
    # Auditoria explícita antes desta task: NENHUMA integração de e-mail
    # existia no projeto (nem Resend, nem SMTP, nem qualquer outro provedor)
    # — introduzida aqui pela primeira vez. `None` por padrão — sem API key,
    # EmailNotificationAdapter trata como configuração ausente e nunca tenta
    # enviar (mesmo padrão do TelegramNotificationAdapter acima).
    resend_api_key: str | None = Field(default=None, alias="RESEND_API_KEY")
    # Precisa ser um remetente de domínio verificado no Resend em produção;
    # o placeholder abaixo funciona apenas em modo sandbox/teste do Resend.
    resend_from_email: str = Field(
        default="PredictIQ <alerts@predictiq.dev>", alias="RESEND_FROM_EMAIL"
    )
    resend_api_base_url: str = Field(
        default="https://api.resend.com", alias="RESEND_API_BASE_URL"
    )
    resend_client_timeout_seconds: float = Field(
        default=10.0, alias="RESEND_CLIENT_TIMEOUT_SECONDS"
    )

    # ── Fila assíncrona de notificações — Celery + Redis (RNF-50/RNF-51) ────
    # Auditado antes desta task: nenhum Redis/Celery existia no projeto (o
    # rate limiter do RF-24 usa Postgres deliberadamente — ver
    # telegram_alert_rate_limiter.py). Redis aqui serve SÓ como broker do
    # Celery, nunca substitui o Postgres do rate limiter nem nenhum outro
    # dado. Default aponta para o serviço `redis` do docker-compose.yml —
    # nunca hardcoded no código, sempre lido via esta variável.
    celery_broker_url: str = Field(
        default="redis://redis:6379/0", alias="CELERY_BROKER_URL"
    )

    # ── Cache de inferência — Redis (RNF-70 / RNF-71) ───────────────────────
    # MESMA instância Redis do broker acima (serviço `redis` do
    # docker-compose.yml já existente) — logicamente isolada num DB Redis
    # diferente (db=1, broker fica em db=0) para um `FLUSHDB`/eviction do
    # cache nunca tocar a fila do Celery, e vice-versa. Consumida por
    # src/services/inference_cache.py — nunca hardcoded no código.
    redis_cache_url: str = Field(
        default="redis://redis:6379/1", alias="REDIS_CACHE_URL"
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        alias="ALLOWED_ORIGINS",
    )

    # ── Machine Learning ──────────────────────────────────────────────────
    # Directory with the served artefacts: ``<name>.onnx`` + ``<name>_card.json``
    # (one generation, ``*_v3``, written by apps/ml/src/train_production.py).
    # Override via MODELS_DIR in Docker or CI.
    models_dir: Path = Field(
        default=_ML_MODELS,
        alias="MODELS_DIR",
        description="Directory holding the served ONNX artefacts and model cards.",
    )

    # RF-10 — multi-model selection without redeploy.
    # Set ACTIVE_MODEL=xgboost to swap the inference engine at startup.
    active_model: str = Field(
        default="random_forest_v3",
        alias="ACTIVE_MODEL",
        description=(
            "Active inference model, one of the served v3 artefacts: "
            "random_forest_v3, xgboost_v3, mlp_v3, tcn_v3, bilstm_v3, "
            "patchtst_v3, autoencoder_v3."
        ),
    )

    # RF-11 — admin token for /models management endpoints.
    # MUST be overridden in production via the ADMIN_API_TOKEN environment variable.
    admin_api_token: str = Field(
        default="change-me-in-production",
        alias="ADMIN_API_TOKEN",
        description="Bearer token required for all /models admin endpoints.",
    )

    # ── Simulator data source ─────────────────────────────────────────────
    # Points to the processed MetroPT-3 parquet streamed by SensorSimulator.
    # Override via SIMULATOR_PARQUET_PATH env-var (e.g., in Docker Compose).
    simulator_parquet_path: Path = Field(
        default=Path(__file__).resolve().parents[3]
        / "ml"
        / "data"
        / "processed"
        / "metropt3.parquet",
        alias="SIMULATOR_PARQUET_PATH",
        description="Absolute path to the processed MetroPT-3 parquet file.",
    )

    # ── Observabilidade — Prometheus (RNF-76 / RNF-77) ──────────────────────
    # URL interna do Prometheus (rede Docker — nunca exposta ao browser, ver
    # RELATORIO-RNF-76-RNF-77.md §Fase 9). Consumida só por
    # `observability_service.py`, que consulta o PromQL de taxa de erro e
    # devolve um resumo pequeno (status + números) ao frontend — o Prometheus
    # em si nunca é alcançável fora da rede Docker interna.
    prometheus_url: str = Field(
        default="http://prometheus:9090", alias="PROMETHEUS_URL"
    )
    # RNF-77 — limiares de taxa de erro 5xx (rolling window de 5 min, ver
    # observability_service.py). Não são um número "escolhido pra passar no
    # teste": 1%/5% é a banda de alerta precoce comum em SRE (referência:
    # error budget de um SLO de 99% de disponibilidade deixa 1% de folga) —
    # dá margem de resposta ANTES da folga do SLO se esgotar. Configuráveis
    # via .env porque são uma decisão operacional, não uma constante de
    # código.
    error_rate_warning_threshold: float = Field(
        default=0.01, alias="ERROR_RATE_WARNING_THRESHOLD"
    )
    error_rate_critical_threshold: float = Field(
        default=0.05, alias="ERROR_RATE_CRITICAL_THRESHOLD"
    )


# Module-level singleton – import this directly instead of instantiating Settings elsewhere.
settings: Settings = Settings()
