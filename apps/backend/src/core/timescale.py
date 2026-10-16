"""
TimescaleDB para a série de predições — DDL compartilhado.

Usado pela migração 0009 e por ``benchmarks/benchmark_timescale.py``, para que
o benchmark meça exatamente o esquema que vai a produção.

Decisões
--------
* ``predictions`` vira hypertable particionada por ``timestamp`` em chunks de
  1 dia. Uma hypertable exige que toda chave única contenha a coluna de
  partição, por isso a PK passa de ``(id)`` para ``(id, timestamp)``; ``id``
  continua serial e o ORM continua endereçando a linha por ``id``. Os índices
  existentes (``timestamp`` e ``asset_id, timestamp``) já servem às consultas;
  ``create_default_indexes => false`` evita um índice duplicado.
* Compressão nativa segmentada por ``asset_id`` (as consultas do histórico
  sempre filtram por ativo) e ordenada por ``timestamp DESC``; chunks com mais
  de ``COMPRESS_AFTER`` são comprimidos pela política em background.
* Agregado contínuo ``predictions_hourly`` (contagem, média e pico de P(falha),
  nº de alarmes por ativo e hora) — alimenta o gráfico de frequência e os KPIs
  sem varrer a tabela bruta.
* Tudo é opcional: sem a extensão (Postgres puro, SQLite dos testes) a
  migração não faz nada e o sistema funciona igual, só sem o ganho.
"""

from __future__ import annotations

CHUNK_INTERVAL: str = "1 day"
COMPRESS_AFTER: str = "7 days"
CAGG_NAME: str = "predictions_hourly"

EXTENSION_AVAILABLE_SQL: str = (
    "SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb'"
)
IS_HYPERTABLE_SQL: str = (
    "SELECT 1 FROM timescaledb_information.hypertables "
    "WHERE hypertable_name = 'predictions'"
)

# Executados em ordem, dentro da transação da migração.
HYPERTABLE_DDL: tuple[str, ...] = (
    "CREATE EXTENSION IF NOT EXISTS timescaledb",
    "ALTER TABLE predictions DROP CONSTRAINT predictions_pkey",
    "ALTER TABLE predictions ADD PRIMARY KEY (id, timestamp)",
    "SELECT create_hypertable('predictions', 'timestamp', "
    f"chunk_time_interval => INTERVAL '{CHUNK_INTERVAL}', "
    "migrate_data => true, create_default_indexes => false, "
    "if_not_exists => true)",
    "ALTER TABLE predictions SET (timescaledb.compress, "
    "timescaledb.compress_segmentby = 'asset_id', "
    "timescaledb.compress_orderby = 'timestamp DESC')",
    f"SELECT add_compression_policy('predictions', INTERVAL '{COMPRESS_AFTER}', "
    "if_not_exists => true)",
)

# CREATE MATERIALIZED VIEW ... WITH (timescaledb.continuous) não pode rodar
# dentro de um bloco de transação: a migração usa autocommit_block().
CAGG_DDL: tuple[str, ...] = (
    f"CREATE MATERIALIZED VIEW IF NOT EXISTS {CAGG_NAME} "
    "WITH (timescaledb.continuous) AS "
    "SELECT time_bucket(INTERVAL '1 hour', timestamp) AS bucket, asset_id, "
    "count(*) AS n, avg(failure_probability) AS mean_probability, "
    "max(failure_probability) AS max_probability, "
    "sum(predicted_class) AS n_alarms "
    "FROM predictions GROUP BY bucket, asset_id WITH NO DATA",
    f"SELECT add_continuous_aggregate_policy('{CAGG_NAME}', "
    "start_offset => INTERVAL '3 days', end_offset => INTERVAL '1 hour', "
    "schedule_interval => INTERVAL '15 minutes', if_not_exists => true)",
)

CAGG_DROP_DDL: tuple[str, ...] = (f"DROP MATERIALIZED VIEW IF EXISTS {CAGG_NAME}",)

# Volta a tabela comum: hypertable não tem "un-create", então copia para uma
# tabela nova com o esquema original e troca os nomes. A sequência do id é
# desvinculada antes do DROP (senão cairia junto) e revinculada no fim.
REVERT_DDL: tuple[str, ...] = (
    "SELECT remove_compression_policy('predictions', if_exists => true)",
    "ALTER SEQUENCE predictions_id_seq OWNED BY NONE",
    "CREATE TABLE predictions_plain (LIKE predictions INCLUDING DEFAULTS)",
    "INSERT INTO predictions_plain SELECT * FROM predictions",
    "DROP TABLE predictions",
    "ALTER TABLE predictions_plain RENAME TO predictions",
    "ALTER TABLE predictions ADD CONSTRAINT predictions_pkey PRIMARY KEY (id)",
    "ALTER SEQUENCE predictions_id_seq OWNED BY predictions.id",
    "ALTER TABLE predictions ADD CONSTRAINT fk_predictions_asset "
    "FOREIGN KEY (asset_id) REFERENCES assets (id)",
    "CREATE INDEX ix_predictions_timestamp ON predictions (timestamp)",
    "CREATE INDEX ix_predictions_asset_timestamp ON predictions (asset_id, timestamp)",
)
