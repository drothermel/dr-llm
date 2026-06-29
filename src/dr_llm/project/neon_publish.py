from __future__ import annotations

from enum import StrEnum
from importlib.resources import files
from typing import Any

import psycopg
import yaml
from psycopg import sql
from pydantic import BaseModel, ConfigDict, Field, field_validator

from dr_llm.project.docker_psql import validate_pg_identifier
from dr_llm.project.errors import ProjectError
from dr_llm.project.project_service import get_project

DEFAULT_CONFIG_RESOURCE = "data/neon_publish.yml"
PUBLISHED_SCHEMA_VERSION = 3
MANIFEST_GENERATED_AT_SQL = "now()"
DEFAULT_PUBLISHED_SAMPLES_TABLE = "published_pool_samples"
PUBLISHED_SAMPLE_TEST_FAILURES_TABLE = "published_sample_test_failures"
PUBLISHED_TASKS_TABLE = "published_tasks"
NL_LATENTS_SUMMARY_TABLE = "published_nl_latents_samples"
NL_LATENTS_PROCESSOR_NAME = "nl_latents_samples_v1"
NL_LATENTS_ROUNDTRIP_PROCESSOR_NAME = "nl_latents_roundtrip_v1"
ENCODER_DESCRIPTION_PROCESSOR_NAME = "encoder_description_v1"
DECODER_CODE_PROCESSOR_NAME = "decoder_code_v1"


class PublishProcessor(StrEnum):
    nl_latents_samples_v1 = NL_LATENTS_PROCESSOR_NAME
    nl_latents_roundtrip_v1 = NL_LATENTS_ROUNDTRIP_PROCESSOR_NAME
    encoder_description_v1 = ENCODER_DESCRIPTION_PROCESSOR_NAME
    decoder_code_v1 = DECODER_CODE_PROCESSOR_NAME


class SampleRole(StrEnum):
    encoder_description = "encoder_description"
    decoder_code = "decoder_code"
    roundtrip_code = "roundtrip_code"


class PublishedPoolConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_pool: str
    processor: PublishProcessor
    summary_table: str = DEFAULT_PUBLISHED_SAMPLES_TABLE

    @field_validator("source_pool", "summary_table")
    @classmethod
    def _validate_pg_name(cls, value: str) -> str:
        return validate_pg_identifier(value, "database identifier")


class PublishedProjectConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    project_name: str
    pools: list[PublishedPoolConfig]

    @field_validator("project_name")
    @classmethod
    def _validate_project_name(cls, value: str) -> str:
        return validate_pg_identifier(value, "database identifier")

    @field_validator("pools")
    @classmethod
    def _validate_unique_pools(
        cls, value: list[PublishedPoolConfig]
    ) -> list[PublishedPoolConfig]:
        source_pools = [pool.source_pool for pool in value]
        if len(set(source_pools)) != len(source_pools):
            raise ValueError("Published pool source names must be unique.")
        return value


class NeonPublishConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int
    manifest_table: str
    published_samples_table: str = DEFAULT_PUBLISHED_SAMPLES_TABLE
    projects: list[PublishedProjectConfig] = Field(default_factory=list)
    pools: list[PublishedPoolConfig] = Field(default_factory=list)

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: int) -> int:
        if value != 1:
            raise ValueError(
                "Only Neon publish config version 1 is supported."
            )
        return value

    @field_validator("manifest_table", "published_samples_table")
    @classmethod
    def _validate_table_name(cls, value: str) -> str:
        return validate_pg_identifier(value, "database identifier")

    @field_validator("projects")
    @classmethod
    def _validate_unique_projects(
        cls, value: list[PublishedProjectConfig]
    ) -> list[PublishedProjectConfig]:
        project_names = [project.project_name for project in value]
        if len(set(project_names)) != len(project_names):
            raise ValueError("Published project names must be unique.")
        return value

    def project_config(self, project_name: str) -> PublishedProjectConfig:
        for project in self.projects:
            if project.project_name == project_name:
                return project
        if self.pools and project_name == "nl_latents":
            return PublishedProjectConfig(
                project_name=project_name, pools=self.pools
            )
        raise ProjectError(
            f"Project {project_name!r} is not configured for Neon publish."
        )

    def published_table_names_for(self, project_name: str) -> tuple[str, ...]:
        project = self.project_config(project_name)
        tables = {
            self.manifest_table,
            self.published_samples_table,
            PUBLISHED_SAMPLE_TEST_FAILURES_TABLE,
            PUBLISHED_TASKS_TABLE,
        }
        if any(
            pool.processor is PublishProcessor.nl_latents_roundtrip_v1
            or pool.processor is PublishProcessor.nl_latents_samples_v1
            for pool in project.pools
        ):
            tables.add(NL_LATENTS_SUMMARY_TABLE)
        return tuple(
            table
            for table in (
                self.manifest_table,
                self.published_samples_table,
                PUBLISHED_SAMPLE_TEST_FAILURES_TABLE,
                PUBLISHED_TASKS_TABLE,
                NL_LATENTS_SUMMARY_TABLE,
            )
            if table in tables
        )

    @property
    def published_table_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for project in self.projects:
            for table in self.published_table_names_for(project.project_name):
                if table not in names:
                    names.append(table)
        if not names and self.pools:
            return self.published_table_names_for("nl_latents")
        return tuple(names)


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


class PoolSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    project_name: str
    pool: PublishedPoolConfig
    source_table: str
    source_row_count: int
    table_columns: frozenset[str]


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
    project_config = config.project_config(name)
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
            _replace_publish_manifest(conn, config.manifest_table)
            sources = [
                _pool_source(conn, project_config.project_name, pool)
                for pool in project_config.pools
            ]
            _publish_common_samples(
                conn,
                sources=sources,
                summary_table=config.published_samples_table,
            )
            _publish_sample_test_failures(
                conn,
                sources=sources,
                failures_table=PUBLISHED_SAMPLE_TEST_FAILURES_TABLE,
            )
            _publish_tasks(
                conn,
                samples_table=config.published_samples_table,
                tasks_table=PUBLISHED_TASKS_TABLE,
            )
            summaries = [
                _published_pool_summary(
                    conn,
                    manifest_table=config.manifest_table,
                    summary_table=config.published_samples_table,
                    source=source,
                )
                for source in sources
            ]
            if _project_has_nl_latents(project_config):
                _publish_nl_latents_compat_table(
                    conn,
                    source_pool="nl_latents",
                    summary_table=NL_LATENTS_SUMMARY_TABLE,
                )
            conn.commit()
    except psycopg.OperationalError as exc:
        raise ProjectError(
            f"Could not connect to local project {name!r} at {project.dsn}. "
            "Start or restart the project before syncing to Neon."
        ) from exc

    return ProjectNeonPublishResult(
        project_name=name,
        manifest_table=config.manifest_table,
        published_tables=list(config.published_table_names_for(name)),
        pools=summaries,
    )


def _project_has_nl_latents(project: PublishedProjectConfig) -> bool:
    return any(
        pool.processor
        in {
            PublishProcessor.nl_latents_roundtrip_v1,
            PublishProcessor.nl_latents_samples_v1,
        }
        for pool in project.pools
    )


def _pool_source(
    conn: psycopg.Connection[Any],
    project_name: str,
    pool: PublishedPoolConfig,
) -> PoolSource:
    source_table = validate_pg_identifier(
        f"pool_{pool.source_pool}_samples", "table name"
    )
    return PoolSource(
        project_name=project_name,
        pool=pool,
        source_table=source_table,
        source_row_count=_table_row_count(conn, source_table),
        table_columns=frozenset(_table_columns(conn, source_table)),
    )


def _publish_common_samples(
    conn: psycopg.Connection[Any],
    *,
    sources: list[PoolSource],
    summary_table: str,
) -> None:
    build_table = validate_pg_identifier(
        f"{summary_table}__build", "table name"
    )
    _drop_table(conn, build_table)
    _ensure_try_parse_jsonb_function(conn)
    if not sources:
        _create_empty_common_samples_table(conn, build_table)
    else:
        selects = [_common_samples_select(source) for source in sources]
        conn.execute(
            sql.SQL("CREATE TABLE {} AS ").format(sql.Identifier(build_table))
            + sql.SQL(" UNION ALL ").join(selects)
        )
    _replace_table(conn, build_table, summary_table)
    _create_common_samples_indexes(conn, summary_table)


def _publish_sample_test_failures(
    conn: psycopg.Connection[Any],
    *,
    sources: list[PoolSource],
    failures_table: str,
) -> None:
    build_table = validate_pg_identifier(
        f"{failures_table}__build", "table name"
    )
    _drop_table(conn, build_table)
    _ensure_try_parse_jsonb_function(conn)
    selects = [_sample_test_failures_select(source) for source in sources]
    if not selects:
        _create_empty_sample_test_failures_table(conn, build_table)
    else:
        conn.execute(
            sql.SQL("CREATE TABLE {} AS ").format(sql.Identifier(build_table))
            + sql.SQL(" UNION ALL ").join(selects)
        )
    _replace_table(conn, build_table, failures_table)
    _create_sample_test_failures_indexes(conn, failures_table)


def _publish_tasks(
    conn: psycopg.Connection[Any],
    *,
    samples_table: str,
    tasks_table: str,
) -> None:
    build_table = validate_pg_identifier(f"{tasks_table}__build", "table name")
    _drop_table(conn, build_table)
    conn.execute(
        sql.SQL(
            """
            CREATE TABLE {} AS
            SELECT
                source_project,
                coalesce(dataset_id, task_id) AS dataset_id,
                coalesce(task_id, dataset_id) AS task_id,
                min(task_family) AS task_family,
                min(task_split) AS task_split,
                min(language) AS language,
                min(difficulty) AS difficulty,
                min(input_text) FILTER (
                    WHERE sample_role = 'encoder_description'
                      AND input_text IS NOT NULL
                ) AS prompt_text,
                NULL::text AS canonical_solution,
                NULL::text AS test_source,
                count(*)::bigint AS sample_count
            FROM {}
            WHERE dataset_id IS NOT NULL OR task_id IS NOT NULL
            GROUP BY source_project, dataset_id, task_id
            """
        ).format(sql.Identifier(build_table), sql.Identifier(samples_table))
    )
    _replace_table(conn, build_table, tasks_table)
    _create_tasks_indexes(conn, tasks_table)


def _published_pool_summary(
    conn: psycopg.Connection[Any],
    *,
    manifest_table: str,
    summary_table: str,
    source: PoolSource,
) -> PublishedPoolSummary:
    summary_row_count = _published_pool_row_count(
        conn,
        summary_table=summary_table,
        source_pool=source.pool.source_pool,
    )
    _upsert_manifest(
        conn,
        manifest_table=manifest_table,
        source=source,
        summary_table=summary_table,
        summary_row_count=summary_row_count,
    )
    return PublishedPoolSummary(
        source_pool=source.pool.source_pool,
        processor=source.pool.processor,
        summary_table=summary_table,
        source_row_count=source.source_row_count,
        summary_row_count=summary_row_count,
    )


def _replace_publish_manifest(
    conn: psycopg.Connection[Any], manifest_table: str
) -> None:
    _drop_table(conn, manifest_table)
    conn.execute(
        sql.SQL(
            """
            CREATE TABLE {} (
                source_project text NOT NULL,
                source_pool text NOT NULL,
                processor text NOT NULL,
                summary_table text NOT NULL,
                source_row_count bigint NOT NULL,
                summary_row_count bigint NOT NULL,
                summary_schema_version integer NOT NULL,
                generated_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (source_project, source_pool)
            )
            """
        ).format(sql.Identifier(manifest_table))
    )


def _publish_nl_latents_compat_table(
    conn: psycopg.Connection[Any],
    *,
    source_pool: str,
    summary_table: str,
) -> None:
    source_table = validate_pg_identifier(
        f"pool_{source_pool}_samples", "table name"
    )
    catalog_table = validate_pg_identifier(
        f"{source_pool}_pool_catalog_entries", "table name"
    )
    build_table = validate_pg_identifier(
        f"{summary_table}__build", "table name"
    )
    _drop_table(conn, build_table)
    _create_nl_latents_summary_table(
        conn,
        source_table=source_table,
        catalog_table=catalog_table,
        build_table=build_table,
    )
    _replace_table(conn, build_table, summary_table)
    _create_nl_latents_summary_indexes(conn, summary_table)


def _common_samples_select(source: PoolSource) -> sql.Composed:
    if source.pool.processor in {
        PublishProcessor.nl_latents_roundtrip_v1,
        PublishProcessor.nl_latents_samples_v1,
    }:
        return _nl_latents_common_select(source)
    if source.pool.processor is PublishProcessor.encoder_description_v1:
        return _encoder_common_select(source)
    if source.pool.processor is PublishProcessor.decoder_code_v1:
        return _decoder_common_select(source)
    raise ProjectError(
        f"Unsupported Neon publish processor: {source.pool.processor}"
    )


def _nl_latents_common_select(source: PoolSource) -> sql.Composed:
    table = sql.Identifier(source.source_table)
    payload = sql.SQL(
        "coalesce(s.response_json, s.request_json, jsonb_build_object())"
    )
    validation_json = _json_text_expr(payload, "validation_json")
    return sql.SQL(
        """
        SELECT
            {project_name}::text AS source_project,
            {pool_name}::text AS source_pool,
            {source_table}::text AS source_table,
            s.sample_id AS source_sample_id,
            s.sample_idx,
            s.run_id,
            s.created_at,
            s.status,
            s.attempt_count,
            s.finish_reason,
            {sample_role}::text AS sample_role,
            'decoded_code'::text AS output_kind,
            'response_json.decoded_code'::text AS output_json_path,
            {payload}->>'decoded_code' AS output_text,
            s.task_id AS dataset_id,
            s.task_id,
            s.family AS task_family,
            s.split AS task_split,
            s.language,
            s.difficulty,
            s.budget AS budget_label,
            nullif(s.budget, '')::integer AS budget_chars,
            s.config_id AS prompt_template_id,
            s.enc_model AS llm_config_id,
            split_part(replace(s.dec_model, 'openrouter:', ''), '/', 1)
                AS provider,
            replace(s.dec_model, 'openrouter:', '') AS model,
            s.config_id AS enc_prompt_template_id,
            s.enc_model AS enc_llm_config_id,
            s.call_id AS enc_sample_id,
            s.config_id AS dec_prompt_template_id,
            s.dec_model AS dec_llm_config_id,
            NULL::text AS upstream_project,
            NULL::text AS upstream_pool,
            NULL::text AS upstream_sample_id,
            NULL::integer AS upstream_sample_idx,
            {payload}->>'model_provenance_source' AS source_kind,
            {payload}->>'description' AS input_text,
            'payload.description'::text AS input_text_source,
            CASE
                WHEN s.response_json IS NULL THEN 'pending'
                WHEN s.response_json->>'passed' = 'true' THEN 'passed'
                WHEN s.response_json->>'passed' = 'false' THEN 'failed'
                ELSE 'pending'
            END AS result_state,
            CASE
                WHEN s.response_json IS NULL THEN NULL::boolean
                ELSE (s.response_json->>'passed')::boolean
            END AS passed,
            ({validation_json}->>'test_pass_rate')::double precision
                AS validation_pass_rate,
            {payload}->>'failure_category' AS failure_category,
            ({payload}->>'budget_ok')::boolean AS budget_ok,
            ({payload}->>'actual_chars')::integer AS actual_chars,
            NULL::text AS mode,
            NULL::integer AS warning_count,
            NULL::integer AS prompt_tokens,
            NULL::integer AS completion_tokens,
            NULL::integer AS reasoning_tokens,
            NULL::integer AS total_tokens,
            NULL::integer AS computed_total_tokens,
            NULL::numeric AS total_cost_usd,
            NULL::numeric AS prompt_cost_usd,
            NULL::numeric AS completion_cost_usd,
            NULL::numeric AS reasoning_cost_usd,
            NULL::text AS cost_currency,
            {payload}->>'detail' AS error_text,
            ({validation_json}->>'validation_time_seconds')
                ::double precision AS validation_time_seconds,
            ({validation_json}->>'compiles')::boolean AS compiles,
            {validation_json}->>'compile_error' AS compile_error,
            ({validation_json}->>'has_code_fences')::boolean
                AS has_code_fences,
            ({validation_json}->>'has_expected_function')::boolean
                AS has_expected_function
        FROM {table} s
        """
    ).format(
        project_name=sql.Literal(source.project_name),
        pool_name=sql.Literal(source.pool.source_pool),
        source_table=sql.Literal(source.source_table),
        sample_role=sql.Literal(SampleRole.roundtrip_code.value),
        payload=payload,
        validation_json=validation_json,
        table=table,
    )


def _encoder_common_select(source: PoolSource) -> sql.Composed:
    table = sql.Identifier(source.source_table)
    metadata = sql.SQL("coalesce(s.metadata_json, jsonb_build_object())")
    return sql.SQL(
        """
        SELECT
            {project_name}::text AS source_project,
            {pool_name}::text AS source_pool,
            {source_table}::text AS source_table,
            s.sample_id AS source_sample_id,
            s.sample_idx,
            s.run_id,
            s.created_at,
            {status} AS status,
            s.attempt_count,
            s.finish_reason,
            {sample_role}::text AS sample_role,
            'text'::text AS output_kind,
            'response_json.text'::text AS output_json_path,
            s.response_json->>'text' AS output_text,
            {data_sample_id} AS dataset_id,
            coalesce({metadata}->'task'->>'task_id', {data_sample_id})
                AS task_id,
            coalesce(
                {metadata}->'task'->>'family',
                split_part({data_sample_id}, '/', 1)
            ) AS task_family,
            {metadata}->'task'->>'split' AS task_split,
            coalesce({metadata}->'task'->>'language', 'python') AS language,
            {metadata}->'task'->>'difficulty' AS difficulty,
            {metadata}->'budgets'->>'budget_label' AS budget_label,
            pg_temp._dr_llm_first_int(
                {prompt_template_id},
                {metadata}->'budgets'->>'budget_chars'
            )
                AS budget_chars,
            {prompt_template_id} AS prompt_template_id,
            {llm_config_id} AS llm_config_id,
            s.response_json->>'provider' AS provider,
            s.response_json->>'model' AS model,
            {prompt_template_id} AS enc_prompt_template_id,
            {llm_config_id} AS enc_llm_config_id,
            s.sample_id AS enc_sample_id,
            NULL::text AS dec_prompt_template_id,
            NULL::text AS dec_llm_config_id,
            NULL::text AS upstream_project,
            NULL::text AS upstream_pool,
            NULL::text AS upstream_sample_id,
            NULL::integer AS upstream_sample_idx,
            {metadata}->>'source_kind' AS source_kind,
            {prompt_text} AS input_text,
            'request.prompt'::text AS input_text_source,
            CASE
                WHEN s.response_json IS NULL THEN 'pending'
                WHEN s.response_json ? 'error' THEN 'failed'
                ELSE 'completed'
            END AS result_state,
            NULL::boolean AS passed,
            NULL::double precision AS validation_pass_rate,
            {metadata}->>'failure_category' AS failure_category,
            ({metadata}->>'budget_ok')::boolean AS budget_ok,
            NULL::integer AS actual_chars,
            s.response_json->>'mode' AS mode,
            CASE
                WHEN jsonb_typeof(s.response_json->'warnings') = 'array'
                    THEN jsonb_array_length(s.response_json->'warnings')
                ELSE NULL::integer
            END AS warning_count,
            (s.response_json->'usage'->>'prompt_tokens')::integer
                AS prompt_tokens,
            (s.response_json->'usage'->>'completion_tokens')::integer
                AS completion_tokens,
            (s.response_json->'usage'->>'reasoning_tokens')::integer
                AS reasoning_tokens,
            (s.response_json->'usage'->>'total_tokens')::integer
                AS total_tokens,
            (s.response_json->'usage'->>'computed_total_tokens')::integer
                AS computed_total_tokens,
            (s.response_json->'cost'->>'total_cost_usd')::numeric
                AS total_cost_usd,
            (s.response_json->'cost'->>'prompt_cost_usd')::numeric
                AS prompt_cost_usd,
            (s.response_json->'cost'->>'completion_cost_usd')::numeric
                AS completion_cost_usd,
            (s.response_json->'cost'->>'reasoning_cost_usd')::numeric
                AS reasoning_cost_usd,
            s.response_json->'cost'->>'currency' AS cost_currency,
            s.response_json->>'error' AS error_text,
            NULL::double precision AS validation_time_seconds,
            NULL::boolean AS compiles,
            NULL::text AS compile_error,
            NULL::boolean AS has_code_fences,
            NULL::boolean AS has_expected_function
        FROM {table} s
        """
    ).format(
        project_name=sql.Literal(source.project_name),
        pool_name=sql.Literal(source.pool.source_pool),
        source_table=sql.Literal(source.source_table),
        status=_column_or_null(source, "status", "text"),
        sample_role=sql.Literal(SampleRole.encoder_description.value),
        data_sample_id=_column_or_null(source, "data_sample_id", "text"),
        prompt_template_id=_column_or_null(
            source, "prompt_template_id", "text"
        ),
        llm_config_id=_column_or_null(source, "llm_config_id", "text"),
        metadata=metadata,
        prompt_text=_prompt_text_expr(sql.SQL("s.request_json->'prompt'")),
        table=table,
    )


def _decoder_common_select(source: PoolSource) -> sql.Composed:
    table = sql.Identifier(source.source_table)
    metadata = sql.SQL("coalesce(s.metadata_json, jsonb_build_object())")
    data_sample_id = sql.SQL(
        "coalesce({metadata}->>'data_sample_id', {column})"
    ).format(
        metadata=metadata,
        column=_column_or_null(source, "data_sample_id", "text"),
    )
    source_sample_id = sql.SQL(
        "coalesce({metadata}->>'source_sample_id', {column})"
    ).format(
        metadata=metadata,
        column=_column_or_null(source, "source_sample_id", "text"),
    )
    return sql.SQL(
        """
        SELECT
            {project_name}::text AS source_project,
            {pool_name}::text AS source_pool,
            {source_table}::text AS source_table,
            s.sample_id AS source_sample_id,
            s.sample_idx,
            s.run_id,
            s.created_at,
            {status} AS status,
            s.attempt_count,
            s.finish_reason,
            {sample_role}::text AS sample_role,
            'code_text'::text AS output_kind,
            'response_json.text'::text AS output_json_path,
            s.response_json->>'text' AS output_text,
            {data_sample_id} AS dataset_id,
            coalesce({metadata}->'task'->>'task_id', {data_sample_id})
                AS task_id,
            coalesce(
                {metadata}->'task'->>'family',
                split_part({data_sample_id}, '/', 1)
            ) AS task_family,
            {metadata}->'task'->>'split' AS task_split,
            coalesce({metadata}->'task'->>'language', 'python') AS language,
            {metadata}->'task'->>'difficulty' AS difficulty,
            {metadata}->'budgets'->>'budget_label' AS budget_label,
            pg_temp._dr_llm_first_int(
                {dec_prompt_template_id},
                {metadata}->'budgets'->>'budget_chars'
            ) AS budget_chars,
            coalesce({dec_prompt_template_id}, {enc_prompt_template_id})
                AS prompt_template_id,
            coalesce({dec_llm_config_id}, s.response_json->>'model')
                AS llm_config_id,
            s.response_json->>'provider' AS provider,
            s.response_json->>'model' AS model,
            {enc_prompt_template_id} AS enc_prompt_template_id,
            coalesce({enc_llm_config_id}, {metadata}->>'enc_llm_config_id')
                AS enc_llm_config_id,
            {enc_sample_id} AS enc_sample_id,
            {dec_prompt_template_id} AS dec_prompt_template_id,
            {dec_llm_config_id} AS dec_llm_config_id,
            NULL::text AS upstream_project,
            {metadata}->>'source_pool_name' AS upstream_pool,
            {source_sample_id} AS upstream_sample_id,
            ({metadata}->>'source_sample_idx')::integer AS upstream_sample_idx,
            {metadata}->>'source_kind' AS source_kind,
            coalesce(
                {metadata}->>'source_text',
                {metadata}->'source_sample_payload'->>'text',
                {prompt_text}
            ) AS input_text,
            CASE
                WHEN {metadata} ? 'source_text' THEN 'metadata.source_text'
                WHEN {metadata}->'source_sample_payload' ? 'text'
                    THEN 'metadata.source_sample_payload.text'
                ELSE 'request.prompt'
            END AS input_text_source,
            CASE
                WHEN s.response_json IS NULL THEN 'pending'
                WHEN s.response_json ? 'error' THEN 'failed'
                ELSE 'completed'
            END AS result_state,
            NULL::boolean AS passed,
            NULL::double precision AS validation_pass_rate,
            {metadata}->>'failure_category' AS failure_category,
            ({metadata}->>'budget_ok')::boolean AS budget_ok,
            NULL::integer AS actual_chars,
            s.response_json->>'mode' AS mode,
            CASE
                WHEN jsonb_typeof(s.response_json->'warnings') = 'array'
                    THEN jsonb_array_length(s.response_json->'warnings')
                ELSE NULL::integer
            END AS warning_count,
            (s.response_json->'usage'->>'prompt_tokens')::integer
                AS prompt_tokens,
            (s.response_json->'usage'->>'completion_tokens')::integer
                AS completion_tokens,
            (s.response_json->'usage'->>'reasoning_tokens')::integer
                AS reasoning_tokens,
            (s.response_json->'usage'->>'total_tokens')::integer
                AS total_tokens,
            (s.response_json->'usage'->>'computed_total_tokens')::integer
                AS computed_total_tokens,
            (s.response_json->'cost'->>'total_cost_usd')::numeric
                AS total_cost_usd,
            (s.response_json->'cost'->>'prompt_cost_usd')::numeric
                AS prompt_cost_usd,
            (s.response_json->'cost'->>'completion_cost_usd')::numeric
                AS completion_cost_usd,
            (s.response_json->'cost'->>'reasoning_cost_usd')::numeric
                AS reasoning_cost_usd,
            s.response_json->'cost'->>'currency' AS cost_currency,
            s.response_json->>'error' AS error_text,
            NULL::double precision AS validation_time_seconds,
            NULL::boolean AS compiles,
            NULL::text AS compile_error,
            NULL::boolean AS has_code_fences,
            NULL::boolean AS has_expected_function
        FROM {table} s
        """
    ).format(
        project_name=sql.Literal(source.project_name),
        pool_name=sql.Literal(source.pool.source_pool),
        source_table=sql.Literal(source.source_table),
        status=_column_or_null(source, "status", "text"),
        sample_role=sql.Literal(SampleRole.decoder_code.value),
        data_sample_id=data_sample_id,
        metadata=metadata,
        dec_prompt_template_id=_column_or_null(
            source, "dec_prompt_template_id", "text"
        ),
        enc_prompt_template_id=_column_or_null(
            source, "enc_prompt_template_id", "text"
        ),
        dec_llm_config_id=_column_or_null(source, "dec_llm_config_id", "text"),
        enc_llm_config_id=_column_or_null(source, "enc_llm_config_id", "text"),
        enc_sample_id=_column_or_null(source, "enc_sample_id", "text"),
        source_sample_id=source_sample_id,
        prompt_text=_prompt_text_expr(sql.SQL("s.request_json->'prompt'")),
        table=table,
    )


def _sample_test_failures_select(source: PoolSource) -> sql.Composed:
    table = sql.Identifier(source.source_table)
    payload = sql.SQL(
        "coalesce(s.response_json, s.request_json, jsonb_build_object())"
    )
    validation_json = _json_text_expr(payload, "validation_json")
    return sql.SQL(
        """
        WITH parsed AS (
            SELECT
                s.sample_id AS source_sample_id,
                s.sample_idx,
                {validation_json} AS validation_json
            FROM {table} s
        ),
        cases AS (
            SELECT
                p.source_sample_id,
                p.sample_idx,
                NULL::text AS case_key,
                (array_case.ord - 1)::integer AS case_idx,
                array_case.value AS case_json
            FROM parsed p
            CROSS JOIN LATERAL jsonb_array_elements(
                CASE
                    WHEN jsonb_typeof(
                        p.validation_json->'test_case_results'
                    ) = 'array'
                    THEN p.validation_json->'test_case_results'
                    ELSE '[]'::jsonb
                END
            ) WITH ORDINALITY AS array_case(value, ord)
            UNION ALL
            SELECT
                p.source_sample_id,
                p.sample_idx,
                object_case.key AS case_key,
                NULL::integer AS case_idx,
                object_case.value AS case_json
            FROM parsed p
            CROSS JOIN LATERAL jsonb_each(
                CASE
                    WHEN jsonb_typeof(
                        p.validation_json->'test_case_results'
                    ) = 'object'
                    THEN p.validation_json->'test_case_results'
                    ELSE '{{}}'::jsonb
                END
            ) AS object_case(key, value)
        ),
        normalized AS (
            SELECT
                source_sample_id,
                sample_idx,
                case_key,
                case_idx,
                case_json,
                coalesce(
                    case_json->'input',
                    case_json->'inputs',
                    case_json->'args',
                    case_json->'arguments'
                ) AS input_json,
                coalesce(
                    case_json->'expected',
                    case_json->'expected_output',
                    case_json->'want'
                ) AS expected_json,
                coalesce(
                    case_json->'actual',
                    case_json->'actual_output',
                    case_json->'got',
                    case_json->'output'
                ) AS actual_json,
                coalesce(
                    case_json->>'error',
                    case_json->>'exception',
                    case_json->>'traceback',
                    case_json->>'message'
                ) AS error_text
            FROM cases
            WHERE pg_temp._dr_llm_test_case_failed(case_json)
        )
        SELECT
            {project_name}::text AS source_project,
            {pool_name}::text AS source_pool,
            {source_table}::text AS source_table,
            source_sample_id,
            sample_idx,
            case_key,
            case_idx,
            input_json,
            expected_json,
            actual_json,
            error_text,
            CASE
                WHEN input_json IS NULL
                 AND expected_json IS NULL
                 AND actual_json IS NULL
                 AND error_text IS NULL
                THEN case_json
                ELSE NULL::jsonb
            END AS failure_json
        FROM normalized
        """
    ).format(
        project_name=sql.Literal(source.project_name),
        pool_name=sql.Literal(source.pool.source_pool),
        source_table=sql.Literal(source.source_table),
        validation_json=validation_json,
        table=table,
    )


def _create_empty_common_samples_table(
    conn: psycopg.Connection[Any], build_table: str
) -> None:
    conn.execute(
        sql.SQL(
            """
            CREATE TABLE {} (
                source_project text,
                source_pool text,
                source_table text,
                source_sample_id text,
                sample_idx integer,
                run_id text,
                created_at timestamptz,
                status text,
                attempt_count integer,
                finish_reason text,
                sample_role text,
                output_kind text,
                output_json_path text,
                output_text text,
                dataset_id text,
                task_id text,
                task_family text,
                task_split text,
                language text,
                difficulty text,
                budget_label text,
                budget_chars integer,
                prompt_template_id text,
                llm_config_id text,
                provider text,
                model text,
                enc_prompt_template_id text,
                enc_llm_config_id text,
                enc_sample_id text,
                dec_prompt_template_id text,
                dec_llm_config_id text,
                upstream_project text,
                upstream_pool text,
                upstream_sample_id text,
                upstream_sample_idx integer,
                source_kind text,
                input_text text,
                input_text_source text,
                result_state text,
                passed boolean,
                validation_pass_rate double precision,
                failure_category text,
                budget_ok boolean,
                actual_chars integer,
                mode text,
                warning_count integer,
                prompt_tokens integer,
                completion_tokens integer,
                reasoning_tokens integer,
                total_tokens integer,
                computed_total_tokens integer,
                total_cost_usd numeric,
                prompt_cost_usd numeric,
                completion_cost_usd numeric,
                reasoning_cost_usd numeric,
                cost_currency text,
                error_text text,
                validation_time_seconds double precision,
                compiles boolean,
                compile_error text,
                has_code_fences boolean,
                has_expected_function boolean
            )
            """
        ).format(sql.Identifier(build_table))
    )


def _create_empty_sample_test_failures_table(
    conn: psycopg.Connection[Any], build_table: str
) -> None:
    conn.execute(
        sql.SQL(
            """
            CREATE TABLE {} (
                source_project text,
                source_pool text,
                source_table text,
                source_sample_id text,
                sample_idx integer,
                case_key text,
                case_idx integer,
                input_json jsonb,
                expected_json jsonb,
                actual_json jsonb,
                error_text text,
                failure_json jsonb
            )
            """
        ).format(sql.Identifier(build_table))
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
    conn.execute(
        """
        CREATE OR REPLACE FUNCTION pg_temp._dr_llm_first_int(
            VARIADIC input_values text[]
        )
        RETURNS integer AS $$
        DECLARE
            value text;
            match text[];
        BEGIN
            FOREACH value IN ARRAY input_values LOOP
                IF value IS NOT NULL THEN
                    match := regexp_match(value, '(?:budget|budget_chars|var_budget)=([0-9]+)');
                    IF match IS NOT NULL THEN
                        RETURN match[1]::integer;
                    END IF;
                    IF value ~ '^[0-9]+$' THEN
                        RETURN value::integer;
                    END IF;
                END IF;
            END LOOP;
            RETURN NULL;
        END
        $$ LANGUAGE plpgsql IMMUTABLE;
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE FUNCTION pg_temp._dr_llm_test_case_failed(
            case_json jsonb
        )
        RETURNS boolean AS $$
        DECLARE
            result_text text;
        BEGIN
            result_text := coalesce(
                case_json->>'passed',
                case_json->>'success',
                case_json->>'ok'
            );
            IF result_text IS NOT NULL THEN
                RETURN lower(result_text) IN (
                    'false',
                    '0',
                    'no',
                    'failed',
                    'fail'
                );
            END IF;
            RETURN case_json ? 'error'
                OR case_json ? 'exception'
                OR case_json ? 'traceback';
        END
        $$ LANGUAGE plpgsql IMMUTABLE;
        """
    )


def _create_common_samples_indexes(
    conn: psycopg.Connection[Any], table_name: str
) -> None:
    for suffix, columns in (
        ("source", "source_project, source_pool, source_sample_id"),
        ("list", "source_project, source_pool, sample_role, created_at DESC"),
        ("dataset", "dataset_id"),
        ("task_family", "task_family"),
        ("model", "model"),
        ("result_state", "result_state"),
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


def _create_sample_test_failures_indexes(
    conn: psycopg.Connection[Any], table_name: str
) -> None:
    for suffix, columns in (
        ("sample", "source_project, source_pool, source_sample_id"),
        ("source", "source_project, source_pool"),
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


def _create_tasks_indexes(
    conn: psycopg.Connection[Any], table_name: str
) -> None:
    primary_key_name = validate_pg_identifier(
        f"{table_name}_pk", "constraint name"
    )
    conn.execute(
        sql.SQL(
            """
            ALTER TABLE {} ADD CONSTRAINT {}
            PRIMARY KEY (source_project, dataset_id, task_id)
            """
        ).format(sql.Identifier(table_name), sql.Identifier(primary_key_name))
    )
    for suffix, columns in (
        ("family", "task_family"),
        ("dataset", "dataset_id"),
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
                            jsonb_build_object()
                        )
                        ELSE jsonb_build_object()
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
    source: PoolSource,
    summary_table: str,
    summary_row_count: int,
) -> None:
    conn.execute(
        sql.SQL(
            """
            INSERT INTO {} (
                source_project,
                source_pool,
                processor,
                summary_table,
                source_row_count,
                summary_row_count,
                summary_schema_version,
                generated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, {})
            ON CONFLICT (source_project, source_pool) DO UPDATE SET
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
            source.project_name,
            source.pool.source_pool,
            source.pool.processor.value,
            summary_table,
            source.source_row_count,
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


def _published_pool_row_count(
    conn: psycopg.Connection[Any],
    *,
    summary_table: str,
    source_pool: str,
) -> int:
    row = conn.execute(
        sql.SQL("SELECT count(*) FROM {} WHERE source_pool = %s").format(
            sql.Identifier(summary_table)
        ),
        [source_pool],
    ).fetchone()
    if row is None or not isinstance(row[0], int):
        raise ProjectError(
            f"Could not count published rows for pool {source_pool!r}."
        )
    return row[0]


def _table_columns(conn: psycopg.Connection[Any], table_name: str) -> set[str]:
    rows = conn.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        """,
        [table_name],
    ).fetchall()
    return {str(row[0]) for row in rows}


def _column_or_null(
    source: PoolSource, column_name: str, column_type: str
) -> sql.Composable:
    if column_name in source.table_columns:
        return sql.SQL("s.{}").format(sql.Identifier(column_name))
    column_types: dict[str, sql.SQL] = {
        "integer": sql.SQL("integer"),
        "text": sql.SQL("text"),
    }
    sql_type = column_types.get(column_type)
    if sql_type is None:
        raise ProjectError(f"Unsupported nullable SQL type: {column_type!r}")
    return sql.SQL("NULL::{}").format(sql_type)


def _json_text_expr(payload: sql.Composable, key: str) -> sql.Composed:
    return sql.SQL(
        """
        CASE
            WHEN {payload} ? {key}
             AND nullif({payload}->>{key}, '') IS NOT NULL
            THEN pg_temp.dr_llm_try_parse_jsonb({payload}->>{key})
            ELSE NULL::jsonb
        END
        """
    ).format(payload=payload, key=sql.Literal(key))


def _prompt_text_expr(prompt_json: sql.Composable) -> sql.Composed:
    return sql.SQL(
        """
        CASE
            WHEN jsonb_typeof({prompt_json}) = 'string'
                THEN {prompt_json} #>> ARRAY[]::text[]
            WHEN jsonb_typeof({prompt_json}) = 'array'
                THEN (
                    SELECT string_agg(
                        CASE
                            WHEN jsonb_typeof(part.value) = 'string'
                                THEN part.value #>> ARRAY[]::text[]
                            ELSE part.value->>'content'
                        END,
                        E'\n\n'
                        ORDER BY part.ord
                    )
                    FROM jsonb_array_elements({prompt_json})
                        WITH ORDINALITY AS part(value, ord)
                )
            ELSE NULL::text
        END
        """
    ).format(prompt_json=prompt_json)
