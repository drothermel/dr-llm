from __future__ import annotations

from enum import StrEnum
from importlib.resources import files
from typing import Any

import psycopg
import yaml
from psycopg import sql
from pydantic import BaseModel, ConfigDict, field_validator

from dr_llm.project.docker_psql import validate_pg_identifier
from dr_llm.project.errors import ProjectError
from dr_llm.project.project_service import get_project

DEFAULT_CONFIG_RESOURCE = "data/neon_publish.yml"
PUBLISHED_SCHEMA_VERSION = 1
NL_LATENTS_PROCESSOR_NAME = "nl_latents_samples_v1"
MANIFEST_GENERATED_AT_SQL = "now()"


class PublishProcessor(StrEnum):
    nl_latents_samples_v1 = NL_LATENTS_PROCESSOR_NAME


class PublishedPoolConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_pool: str
    processor: PublishProcessor
    summary_table: str

    @field_validator("source_pool", "summary_table")
    @classmethod
    def _validate_pg_name(cls, value: str) -> str:
        return validate_pg_identifier(value, "database identifier")


class NeonPublishConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int
    manifest_table: str
    pools: list[PublishedPoolConfig]

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: int) -> int:
        if value != 1:
            raise ValueError(
                "Only Neon publish config version 1 is supported."
            )
        return value

    @field_validator("manifest_table")
    @classmethod
    def _validate_manifest_table(cls, value: str) -> str:
        return validate_pg_identifier(value, "database identifier")

    @field_validator("pools")
    @classmethod
    def _validate_unique_tables(
        cls, value: list[PublishedPoolConfig]
    ) -> list[PublishedPoolConfig]:
        source_pools = [pool.source_pool for pool in value]
        summary_tables = [pool.summary_table for pool in value]
        if len(set(source_pools)) != len(source_pools):
            raise ValueError("Published pool source names must be unique.")
        if len(set(summary_tables)) != len(summary_tables):
            raise ValueError("Published pool summary tables must be unique.")
        return value

    @property
    def published_table_names(self) -> tuple[str, ...]:
        return (
            self.manifest_table,
            *(pool.summary_table for pool in self.pools),
        )


class PublishedPoolSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_pool: str
    processor: PublishProcessor
    summary_table: str
    source_row_count: int
    summary_row_count: int
    summary_schema_version: int = PUBLISHED_SCHEMA_VERSION


class ProjectNeonPublishResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    project_name: str
    manifest_table: str
    published_tables: list[str]
    pools: list[PublishedPoolSummary]


def load_neon_publish_config() -> NeonPublishConfig:
    config_path = files("dr_llm.project").joinpath(DEFAULT_CONFIG_RESOURCE)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ProjectError("Neon publish config must be a mapping.")
    try:
        return NeonPublishConfig(**raw)
    except ValueError as exc:
        raise ProjectError(f"Invalid Neon publish config: {exc}") from exc


def publish_project_for_neon(name: str) -> ProjectNeonPublishResult:
    config = load_neon_publish_config()
    project = get_project(name)
    if not project.running:
        raise ProjectError(
            f"Project {name!r} is {project.status}; start it first with "
            f"`uv run python -m dr_llm project start {name}`."
        )
    if project.dsn is None:
        raise ProjectError(f"Project {name!r} has no DSN; start it first.")

    try:
        with psycopg.connect(project.dsn) as conn:
            _ensure_publish_manifest(conn, config.manifest_table)
            summaries = [
                _publish_pool(conn, config.manifest_table, pool)
                for pool in config.pools
            ]
            conn.commit()
    except psycopg.OperationalError as exc:
        raise ProjectError(
            f"Could not connect to local project {name!r} at {project.dsn}. "
            "Start or restart the project before syncing to Neon."
        ) from exc

    return ProjectNeonPublishResult(
        project_name=name,
        manifest_table=config.manifest_table,
        published_tables=list(config.published_table_names),
        pools=summaries,
    )


def _publish_pool(
    conn: psycopg.Connection[Any],
    manifest_table: str,
    pool: PublishedPoolConfig,
) -> PublishedPoolSummary:
    if pool.processor is not PublishProcessor.nl_latents_samples_v1:
        raise ProjectError(
            f"Unsupported Neon publish processor: {pool.processor}"
        )
    return _publish_nl_latents_samples(conn, manifest_table, pool)


def _ensure_publish_manifest(
    conn: psycopg.Connection[Any], manifest_table: str
) -> None:
    conn.execute(
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {} (
                source_pool text PRIMARY KEY,
                processor text NOT NULL,
                summary_table text NOT NULL,
                source_row_count bigint NOT NULL,
                summary_row_count bigint NOT NULL,
                summary_schema_version integer NOT NULL,
                generated_at timestamptz NOT NULL DEFAULT now()
            )
            """
        ).format(sql.Identifier(manifest_table))
    )


def _publish_nl_latents_samples(
    conn: psycopg.Connection[Any],
    manifest_table: str,
    pool: PublishedPoolConfig,
) -> PublishedPoolSummary:
    source_table = validate_pg_identifier(
        f"pool_{pool.source_pool}_samples", "table name"
    )
    catalog_table = validate_pg_identifier(
        f"{pool.source_pool}_pool_catalog_entries", "table name"
    )
    build_table = validate_pg_identifier(
        f"{pool.summary_table}__build", "table name"
    )

    source_row_count = _table_row_count(conn, source_table)
    _drop_table(conn, build_table)
    _create_nl_latents_summary_table(
        conn,
        source_table=source_table,
        catalog_table=catalog_table,
        build_table=build_table,
    )
    _replace_table(conn, build_table, pool.summary_table)
    _create_nl_latents_summary_indexes(conn, pool.summary_table)
    summary_row_count = _table_row_count(conn, pool.summary_table)
    _upsert_manifest(
        conn,
        manifest_table=manifest_table,
        pool=pool,
        source_row_count=source_row_count,
        summary_row_count=summary_row_count,
    )
    return PublishedPoolSummary(
        source_pool=pool.source_pool,
        processor=pool.processor,
        summary_table=pool.summary_table,
        source_row_count=source_row_count,
        summary_row_count=summary_row_count,
    )


def _create_nl_latents_summary_table(
    conn: psycopg.Connection[Any],
    *,
    source_table: str,
    catalog_table: str,
    build_table: str,
) -> None:
    _ensure_try_parse_jsonb_function(conn)
    conn.execute(
        sql.SQL(
            """
            CREATE TABLE {} AS
            WITH prompt_configs AS (
                SELECT
                    replace(entry_id, 'prompt_config:', '') AS config_id,
                    (value_json->>'config_json')::jsonb AS prompt_config_json
                FROM {}
                WHERE namespace = 'legacy'
                  AND entry_id LIKE 'prompt_config:%'
            ),
            prompt_parts AS (
                SELECT
                    pc.config_id,
                    pc.prompt_config_json,
                    coalesce(
                        array_agg(part.value ORDER BY part.ord)
                            FILTER (WHERE part.value IS NOT NULL),
                        ARRAY[]::text[]
                    ) AS prompt_block_ids,
                    coalesce(
                        array_agg(initcap(replace(part.value, '_', ' '))
                            ORDER BY part.ord)
                            FILTER (WHERE part.value IS NOT NULL),
                        ARRAY[]::text[]
                    ) AS prompt_block_names
                FROM prompt_configs pc
                LEFT JOIN LATERAL (
                    SELECT
                        ord,
                        coalesce(
                            element->>'block_id',
                            element->>'template_id',
                            element->>'type'
                        ) AS value
                    FROM jsonb_array_elements(
                        CASE
                            WHEN jsonb_typeof(
                                pc.prompt_config_json->'elements'
                            ) = 'array'
                            THEN pc.prompt_config_json->'elements'
                            ELSE '[]'::jsonb
                        END
                    ) WITH ORDINALITY AS parts(element, ord)
                ) part ON true
                GROUP BY pc.config_id, pc.prompt_config_json
            ),
            payloads AS (
                SELECT
                    s.*,
                    coalesce(s.response_json, s.request_json) AS payload
                FROM {}
                s
            ),
            parsed AS (
                SELECT
                    p.*,
                    CASE
                        WHEN p.payload ? 'metadata_json'
                         AND nullif(p.payload->>'metadata_json', '') IS NOT NULL
                        THEN coalesce(
                            pg_temp.dr_llm_try_parse_jsonb(
                                p.payload->>'metadata_json'
                            ),
                            '{{}}'::jsonb
                        )
                        ELSE '{{}}'::jsonb
                    END AS response_metadata_json,
                    CASE
                        WHEN p.payload ? 'validation_json'
                         AND nullif(p.payload->>'validation_json', '') IS NOT NULL
                        THEN pg_temp.dr_llm_try_parse_jsonb(
                            p.payload->>'validation_json'
                        )
                        ELSE NULL::jsonb
                    END AS validation_json,
                    p.payload->>'enc_prompt' AS enc_prompt_text
                FROM payloads p
            ),
            display AS (
                SELECT
                    parsed.*,
                    (regexp_match(
                        parsed.enc_prompt_text,
                        '(?s)((?:def|class) .*)'
                    ))[1] AS input_code_text
                FROM parsed
            )
            SELECT
                d.sample_id,
                d.config_id,
                d.family,
                d.difficulty,
                d.difficulty::integer AS difficulty_level,
                d.split,
                d.language,
                d.budget,
                d.budget::integer AS budget_chars,
                d.task_id,
                d.task_data_version,
                d.enc_model,
                d.dec_model,
                replace(d.enc_model, 'openrouter:', '') AS enc_model_label,
                replace(d.dec_model, 'openrouter:', '') AS dec_model_label,
                d.enc_reasoning_effort,
                d.dec_reasoning_effort,
                d.call_id,
                d.status,
                d.sample_idx,
                d.run_id,
                d.finish_reason,
                d.attempt_count,
                d.created_at,
                pp.prompt_config_json,
                coalesce(pp.prompt_block_ids, ARRAY[]::text[])
                    AS prompt_block_ids,
                coalesce(pp.prompt_block_names, ARRAY[]::text[])
                    AS prompt_block_names,
                array_to_string(
                    coalesce(pp.prompt_block_names, ARRAY[]::text[]),
                    ' / '
                ) AS prompt_config_label,
                CASE
                    WHEN d.response_json IS NULL THEN 'pending'
                    WHEN d.response_json->>'passed' = 'true' THEN 'passed'
                    WHEN d.response_json->>'passed' = 'false' THEN 'failed'
                    ELSE 'pending'
                END AS result_state,
                CASE
                    WHEN d.response_json IS NULL THEN NULL::boolean
                    ELSE (d.response_json->>'passed')::boolean
                END AS passed,
                d.payload->>'failure_category' AS failure_category,
                CASE
                    WHEN d.payload->>'failure_category' IS NULL THEN NULL
                    ELSE lower(
                        regexp_replace(
                            d.payload->>'failure_category',
                            '^FailureCategory\\.',
                            ''
                        )
                    )
                END AS failure_category_normalized,
                d.payload->>'model_provenance_source'
                    AS model_provenance_source,
                (d.payload->>'budget_ok')::boolean AS budget_ok,
                (d.payload->>'actual_chars')::integer AS actual_chars,
                (d.response_metadata_json->>'enc_time_s')::double precision
                    AS enc_time_s,
                (d.response_metadata_json->>'dec_time_s')::double precision
                    AS dec_time_s,
                CASE
                    WHEN d.validation_json IS NULL THEN NULL::boolean
                    ELSE (d.validation_json->>'compiles')::boolean
                END AS validation_compiles,
                (d.validation_json->>'test_pass_rate')::double precision
                    AS validation_pass_rate,
                (d.validation_json->>'validation_time_seconds')
                    ::double precision AS validation_time_seconds,
                d.input_code_text AS input_code,
                d.enc_prompt_text AS enc_prompt,
                CASE
                    WHEN d.input_code_text IS NULL THEN d.enc_prompt_text
                    ELSE btrim(
                        left(
                            d.enc_prompt_text,
                            greatest(
                                position(d.input_code_text IN d.enc_prompt_text)
                                    - 1,
                                0
                            )
                        )
                    )
                END AS enc_prompt_instructions,
                d.payload->>'description' AS description,
                d.payload->>'dec_system' AS dec_system,
                d.payload->>'dec_task' AS dec_task,
                d.payload->>'decoded_code' AS decoded_code,
                d.payload->>'detail' AS error_detail,
                CASE
                    WHEN d.validation_json IS NULL THEN NULL::jsonb
                    ELSE jsonb_build_object(
                        'compiles', d.validation_json->'compiles',
                        'test_pass_rate',
                            d.validation_json->'test_pass_rate',
                        'validation_time_seconds',
                            d.validation_json->'validation_time_seconds',
                        'compile_error',
                            d.validation_json->'compile_error',
                        'has_code_fences',
                            d.validation_json->'has_code_fences',
                        'has_expected_function',
                            d.validation_json->'has_expected_function'
                    )
                END AS validation_summary_json
            FROM display d
            LEFT JOIN prompt_parts pp ON pp.config_id = d.config_id
            """
        ).format(
            sql.Identifier(build_table),
            sql.Identifier(catalog_table),
            sql.Identifier(source_table),
        )
    )


def _ensure_try_parse_jsonb_function(conn: psycopg.Connection[Any]) -> None:
    conn.execute(
        """
        CREATE OR REPLACE FUNCTION pg_temp.dr_llm_try_parse_jsonb(value text)
        RETURNS jsonb AS $$
        BEGIN
            RETURN value::jsonb;
        EXCEPTION WHEN others THEN
            RETURN NULL;
        END
        $$ LANGUAGE plpgsql IMMUTABLE;
        """
    )


def _create_nl_latents_summary_indexes(
    conn: psycopg.Connection[Any], table_name: str
) -> None:
    primary_key_name = validate_pg_identifier(
        f"{table_name}_pk", "constraint name"
    )
    conn.execute(
        sql.SQL(
            "ALTER TABLE {} ADD CONSTRAINT {} PRIMARY KEY (sample_id)"
        ).format(sql.Identifier(table_name), sql.Identifier(primary_key_name))
    )
    for suffix, columns in (
        (
            "list",
            "family, difficulty, split, enc_model, budget, created_at DESC",
        ),
        ("result_state", "result_state"),
        ("task_data_version", "task_data_version"),
        ("config_id", "config_id"),
    ):
        index_name = validate_pg_identifier(
            f"{table_name}_{suffix}_idx", "index name"
        )
        conn.execute(
            sql.SQL("CREATE INDEX {} ON {} ({})").format(
                sql.Identifier(index_name),
                sql.Identifier(table_name),
                sql.SQL(columns),
            )
        )


def _drop_table(conn: psycopg.Connection[Any], table_name: str) -> None:
    conn.execute(
        sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table_name))
    )


def _replace_table(
    conn: psycopg.Connection[Any], build_table: str, target_table: str
) -> None:
    _drop_table(conn, target_table)
    conn.execute(
        sql.SQL("ALTER TABLE {} RENAME TO {}").format(
            sql.Identifier(build_table),
            sql.Identifier(target_table),
        )
    )


def _upsert_manifest(
    conn: psycopg.Connection[Any],
    *,
    manifest_table: str,
    pool: PublishedPoolConfig,
    source_row_count: int,
    summary_row_count: int,
) -> None:
    conn.execute(
        sql.SQL(
            """
            INSERT INTO {} (
                source_pool,
                processor,
                summary_table,
                source_row_count,
                summary_row_count,
                summary_schema_version,
                generated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, {})
            ON CONFLICT (source_pool) DO UPDATE SET
                processor = excluded.processor,
                summary_table = excluded.summary_table,
                source_row_count = excluded.source_row_count,
                summary_row_count = excluded.summary_row_count,
                summary_schema_version = excluded.summary_schema_version,
                generated_at = excluded.generated_at
            """
        ).format(
            sql.Identifier(manifest_table),
            sql.SQL(MANIFEST_GENERATED_AT_SQL),
        ),
        [
            pool.source_pool,
            pool.processor.value,
            pool.summary_table,
            source_row_count,
            summary_row_count,
            PUBLISHED_SCHEMA_VERSION,
        ],
    )


def _table_row_count(conn: psycopg.Connection[Any], table_name: str) -> int:
    row = conn.execute(
        sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table_name))
    ).fetchone()
    if row is None or not isinstance(row[0], int):
        raise ProjectError(f"Could not count rows in table {table_name!r}.")
    return row[0]
