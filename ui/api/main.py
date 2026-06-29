"""FastAPI backend for dr-llm UI tools."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
import os
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import psycopg
from psycopg.rows import dict_row
from pydantic import BaseModel, ConfigDict, Field

from dr_llm.llm.catalog.models import ModelCatalogEntry
from dr_llm.llm.catalog.service import ModelCatalogService
from dr_llm.errors import ProviderSemanticError, ProviderTransportError
from dr_llm.llm import ControlMode, build_default_registry
from dr_llm.llm.providers.core.registry import ProviderRegistry
from dr_llm.project.neon_publish import load_neon_publish_config

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

UI_DOTENV_FILENAME = ".env"
DR_LLM_DATABASE_URL_ENV = "DR_LLM_DATABASE_URL"
DR_LLM_DATABASE_BASE_URL_ENV = "DR_LLM_DATABASE_BASE_URL"
POSTGRES_SYNC_ADMIN_URL_ENV = "DR_LLM_POSTGRES_SYNC_ADMIN_URL"
NL_LATENTS_DATABASE = "nl_latents"
NL_LATENTS_TABLE = "published_nl_latents_samples"
PUBLISHED_SAMPLES_TABLE = "published_pool_samples"
PUBLISHED_SAMPLE_TEST_FAILURES_TABLE = "published_sample_test_failures"
DR_LLM_PUBLISHED_PROJECTS_ENV = "DR_LLM_PUBLISHED_PROJECTS"
SMOKE_RUN_ID_PATTERN = "%smoke%"
DEFAULT_PAGE = 1
DEFAULT_LIMIT = 20
MAX_LIMIT = 50
JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None


class ProviderStatusResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    available: bool
    missing_env_vars: list[str] = Field(default_factory=list)
    missing_executables: list[str] = Field(default_factory=list)
    supports_structured_output: bool = False


class ModelEntryResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    display_name: str | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    control_mode: ControlMode | None = None
    supports_vision: bool | None = None
    source_quality: str = "live"


class ProviderModelsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    models: list[ModelEntryResponse]
    source: str  # "live" | "static" | "error"
    error: str | None = None


class SyncResultResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    success: bool
    model_count: int
    models: list[ModelEntryResponse]
    source: str
    error: str | None = None


class NlLatentsFiltersResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    families: list[str] = Field(default_factory=list)
    splits: list[str] = Field(default_factory=list)
    enc_models: list[str] = Field(default_factory=list)
    budgets: list[str] = Field(default_factory=list)
    difficulties: list[str] = Field(default_factory=list)
    data_versions: list[str] = Field(default_factory=list)


class NlLatentsSampleListRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_id: str
    family: str
    difficulty: str
    split: str
    language: str
    budget: str
    enc_model: str
    dec_model: str
    enc_model_label: str
    dec_model_label: str
    status: str
    attempt_count: int
    created_at: datetime
    result_state: str
    failure_category_normalized: str | None = None
    run_id: str | None = None
    prompt_config_label: str | None = None


class NlLatentsSamplesResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    samples: list[NlLatentsSampleListRow]
    total: int
    page: int
    limit: int
    total_pages: int


class NlLatentsSampleDetailResponse(NlLatentsSampleListRow):
    model_config = ConfigDict(frozen=True)

    config_id: str
    task_id: str
    task_data_version: str
    enc_reasoning_effort: str
    dec_reasoning_effort: str
    call_id: str
    sample_idx: int
    finish_reason: str | None = None
    prompt_block_ids: list[str] = Field(default_factory=list)
    prompt_block_names: list[str] = Field(default_factory=list)
    prompt_config_json: dict[str, Any] | None = None
    passed: bool | None = None
    failure_category: str | None = None
    model_provenance_source: str | None = None
    budget_ok: bool | None = None
    actual_chars: int | None = None
    enc_time_s: float | None = None
    dec_time_s: float | None = None
    validation_compiles: bool | None = None
    validation_pass_rate: float | None = None
    validation_time_seconds: float | None = None
    input_code: str | None = None
    enc_prompt: str | None = None
    enc_prompt_instructions: str | None = None
    description: str | None = None
    dec_system: str | None = None
    dec_task: str | None = None
    decoded_code: str | None = None
    error_detail: str | None = None
    validation_summary_json: dict[str, Any] | None = None


class PublishedFiltersResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    projects: list[str] = Field(default_factory=list)
    source_pools: list[str] = Field(default_factory=list)
    sample_roles: list[str] = Field(default_factory=list)
    task_families: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    result_states: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)


class PublishedSampleListRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_project: str
    source_pool: str
    source_sample_id: str
    sample_idx: int
    run_id: str | None = None
    created_at: datetime | None = None
    status: str | None = None
    attempt_count: int | None = None
    finish_reason: str | None = None
    sample_role: str
    output_kind: str
    dataset_id: str | None = None
    task_id: str | None = None
    task_family: str | None = None
    task_split: str | None = None
    language: str | None = None
    difficulty: str | None = None
    budget_label: str | None = None
    budget_chars: int | None = None
    provider: str | None = None
    model: str | None = None
    result_state: str
    passed: bool | None = None
    validation_pass_rate: float | None = None
    failure_category: str | None = None
    budget_ok: bool | None = None
    actual_chars: int | None = None
    output_text: str | None = None
    input_text: str | None = None


class PublishedSamplesResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    samples: list[PublishedSampleListRow]
    total: int
    page: int
    limit: int
    total_pages: int


class PublishedSampleTestFailure(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_project: str
    source_pool: str
    source_sample_id: str
    sample_idx: int | None = None
    case_key: str | None = None
    case_idx: int | None = None
    input_json: JsonValue = None
    expected_json: JsonValue = None
    actual_json: JsonValue = None
    error_text: str | None = None
    failure_json: dict[str, Any] | None = None


class PublishedSampleDetailResponse(PublishedSampleListRow):
    model_config = ConfigDict(frozen=True)

    source_table: str
    output_json_path: str
    prompt_template_id: str | None = None
    llm_config_id: str | None = None
    enc_prompt_template_id: str | None = None
    enc_llm_config_id: str | None = None
    enc_sample_id: str | None = None
    dec_prompt_template_id: str | None = None
    dec_llm_config_id: str | None = None
    upstream_project: str | None = None
    upstream_pool: str | None = None
    upstream_sample_id: str | None = None
    upstream_sample_idx: int | None = None
    source_kind: str | None = None
    input_text_source: str | None = None
    mode: str | None = None
    warning_count: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    computed_total_tokens: int | None = None
    total_cost_usd: float | None = None
    prompt_cost_usd: float | None = None
    completion_cost_usd: float | None = None
    reasoning_cost_usd: float | None = None
    cost_currency: str | None = None
    error_text: str | None = None
    validation_time_seconds: float | None = None
    compiles: bool | None = None
    compile_error: str | None = None
    has_code_fences: bool | None = None
    has_expected_function: bool | None = None
    test_failures: list[PublishedSampleTestFailure] = Field(
        default_factory=list
    )


def _entry_to_response(entry: ModelCatalogEntry) -> ModelEntryResponse:
    return ModelEntryResponse(
        provider=entry.provider,
        model=entry.model,
        display_name=entry.display_name,
        context_window=entry.context_window,
        max_output_tokens=entry.max_output_tokens,
        control_mode=entry.control_mode,
        supports_vision=entry.supports_vision,
        source_quality=entry.source_quality,
    )


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------


def _load_ui_dotenv() -> None:
    dotenv_path = find_dotenv(UI_DOTENV_FILENAME, usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)


_load_ui_dotenv()


def _get_registry(app: FastAPI) -> ProviderRegistry:
    registry = cast(
        ProviderRegistry | None, getattr(app.state, "registry", None)
    )
    if registry is None:
        raise HTTPException(
            status_code=503,
            detail="service unavailable: registry not initialized",
        )
    return registry


def _fallback_model_responses(
    provider: str, service: ModelCatalogService
) -> list[ModelEntryResponse]:
    entries, _raw = service.fallback_provider_models(provider)
    return [_entry_to_response(entry) for entry in entries]


def _parse_positive_int(value: int, fallback: int) -> int:
    return value if value > 0 else fallback


def _database_url_for_database(database_name: str) -> str:
    explicit_url = os.getenv(DR_LLM_DATABASE_URL_ENV)
    if explicit_url:
        return explicit_url
    base_url = os.getenv(DR_LLM_DATABASE_BASE_URL_ENV) or os.getenv(
        POSTGRES_SYNC_ADMIN_URL_ENV
    )
    if not base_url:
        raise HTTPException(
            status_code=503,
            detail=(
                f"{DR_LLM_DATABASE_URL_ENV}, "
                f"{DR_LLM_DATABASE_BASE_URL_ENV}, or "
                f"{POSTGRES_SYNC_ADMIN_URL_ENV} is not configured."
            ),
        )
    parts = urlsplit(base_url)
    if not parts.scheme or not parts.netloc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"{DR_LLM_DATABASE_BASE_URL_ENV} must include a scheme "
                "and host."
            ),
        )
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            f"/{database_name}",
            parts.query,
            parts.fragment,
        )
    )


def _connect_pool_database() -> psycopg.Connection[dict[str, Any]]:
    _load_ui_dotenv()
    dsn = _database_url_for_database(NL_LATENTS_DATABASE)
    return psycopg.connect(dsn, row_factory=dict_row)


def _connect_project_database(
    project_name: str,
) -> psycopg.Connection[dict[str, Any]]:
    _load_ui_dotenv()
    dsn = _database_url_for_database(project_name)
    return psycopg.connect(dsn, row_factory=dict_row)


def _published_projects() -> tuple[str, ...]:
    configured = os.getenv(DR_LLM_PUBLISHED_PROJECTS_ENV)
    if configured:
        names = tuple(
            part.strip() for part in configured.split(",") if part.strip()
        )
        if names:
            return names
    return tuple(
        project.project_name for project in load_neon_publish_config().projects
    )


def _fetch_string_options(column_name: str) -> list[str]:
    with _connect_pool_database() as conn:
        rows = conn.execute(
            f"SELECT DISTINCT {column_name} FROM {NL_LATENTS_TABLE} "
            f"ORDER BY {column_name}"
        ).fetchall()
    return [
        str(row[column_name]) for row in rows if row[column_name] is not None
    ]


def _fetch_nl_latents_filters() -> NlLatentsFiltersResponse:
    return NlLatentsFiltersResponse(
        families=_fetch_string_options("family"),
        splits=_fetch_string_options("split"),
        enc_models=_fetch_string_options("enc_model"),
        budgets=_fetch_string_options("budget"),
        difficulties=_fetch_string_options("difficulty"),
        data_versions=_fetch_string_options("task_data_version"),
    )


def _nl_latents_conditions(
    *,
    family: str | None,
    split: str | None,
    enc_model: str | None,
    budget: str | None,
    difficulty: str | None,
    data_version: str | None,
    result: str | None,
    hide_pending: bool,
    hide_smoke: bool,
) -> tuple[str, list[str]]:
    conditions: list[str] = []
    values: list[str] = []
    for column_name, value in (
        ("family", family),
        ("split", split),
        ("enc_model", enc_model),
        ("budget", budget),
        ("difficulty", difficulty),
        ("task_data_version", data_version),
        ("result_state", result),
    ):
        if value:
            values.append(value)
            conditions.append(f"{column_name} = %s")
    if hide_pending:
        conditions.append("result_state <> 'pending'")
    if hide_smoke:
        values.append(SMOKE_RUN_ID_PATTERN)
        conditions.append("coalesce(run_id, '') NOT ILIKE %s")
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    return where, values


def _fetch_nl_latents_samples(
    *,
    page: int,
    limit: int,
    family: str | None,
    split: str | None,
    enc_model: str | None,
    budget: str | None,
    difficulty: str | None,
    data_version: str | None,
    result: str | None,
    hide_pending: bool,
    hide_smoke: bool,
) -> NlLatentsSamplesResponse:
    page = _parse_positive_int(page, DEFAULT_PAGE)
    limit = min(MAX_LIMIT, _parse_positive_int(limit, DEFAULT_LIMIT))
    offset = (page - 1) * limit
    where, values = _nl_latents_conditions(
        family=family,
        split=split,
        enc_model=enc_model,
        budget=budget,
        difficulty=difficulty,
        data_version=data_version,
        result=result,
        hide_pending=hide_pending,
        hide_smoke=hide_smoke,
    )
    with _connect_pool_database() as conn:
        count_row = conn.execute(
            f"SELECT count(*) AS count FROM {NL_LATENTS_TABLE} {where}",
            values,
        ).fetchone()
        total = int(count_row["count"]) if count_row is not None else 0
        rows = conn.execute(
            f"""
            SELECT
                sample_id,
                family,
                difficulty,
                split,
                language,
                budget,
                enc_model,
                dec_model,
                enc_model_label,
                dec_model_label,
                status,
                attempt_count,
                created_at,
                result_state,
                failure_category_normalized,
                run_id,
                prompt_config_label
            FROM {NL_LATENTS_TABLE}
            {where}
            ORDER BY family, difficulty, split, enc_model, budget,
                     created_at DESC
            LIMIT %s OFFSET %s
            """,
            [*values, limit, offset],
        ).fetchall()
    return NlLatentsSamplesResponse(
        samples=[NlLatentsSampleListRow(**row) for row in rows],
        total=total,
        page=page,
        limit=limit,
        total_pages=(total + limit - 1) // limit if limit else 0,
    )


def _fetch_nl_latents_sample(
    sample_id: str,
) -> NlLatentsSampleDetailResponse | None:
    with _connect_pool_database() as conn:
        row = conn.execute(
            f"SELECT * FROM {NL_LATENTS_TABLE} WHERE sample_id = %s",
            [sample_id],
        ).fetchone()
    if row is None:
        return None
    return NlLatentsSampleDetailResponse(**row)


def _fetch_project_string_options(
    project_name: str, column_name: str
) -> list[str]:
    with _connect_project_database(project_name) as conn:
        rows = conn.execute(
            f"SELECT DISTINCT {column_name} FROM {PUBLISHED_SAMPLES_TABLE} "
            f"ORDER BY {column_name}"
        ).fetchall()
    return [
        str(row[column_name]) for row in rows if row[column_name] is not None
    ]


def _fetch_published_filters() -> PublishedFiltersResponse:
    projects = _published_projects()
    columns = {
        "source_pools": "source_pool",
        "sample_roles": "sample_role",
        "task_families": "task_family",
        "models": "model",
        "result_states": "result_state",
        "datasets": "dataset_id",
    }
    values: dict[str, set[str]] = {name: set() for name in columns}
    for project_name in projects:
        for field_name, column_name in columns.items():
            values[field_name].update(
                _fetch_project_string_options(project_name, column_name)
            )
    return PublishedFiltersResponse(
        projects=sorted(projects),
        source_pools=sorted(values["source_pools"]),
        sample_roles=sorted(values["sample_roles"]),
        task_families=sorted(values["task_families"]),
        models=sorted(values["models"]),
        result_states=sorted(values["result_states"]),
        datasets=sorted(values["datasets"]),
    )


def _published_conditions(
    *,
    project: str | None,
    source_pool: str | None,
    sample_role: str | None,
    task_family: str | None,
    model: str | None,
    result: str | None,
    dataset: str | None,
    hide_pending: bool,
    hide_smoke: bool,
) -> tuple[str, list[str]]:
    conditions: list[str] = []
    values: list[str] = []
    for column_name, value in (
        ("source_project", project),
        ("source_pool", source_pool),
        ("sample_role", sample_role),
        ("task_family", task_family),
        ("model", model),
        ("result_state", result),
        ("dataset_id", dataset),
    ):
        if value:
            values.append(value)
            conditions.append(f"{column_name} = %s")
    if hide_pending:
        conditions.append("result_state <> 'pending'")
    if hide_smoke:
        values.append(SMOKE_RUN_ID_PATTERN)
        conditions.append("coalesce(run_id, '') NOT ILIKE %s")
        values.append(SMOKE_RUN_ID_PATTERN)
        conditions.append("coalesce(source_pool, '') NOT ILIKE %s")
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    return where, values


def _target_published_projects(project: str | None) -> tuple[str, ...]:
    configured_projects = _published_projects()
    if project:
        if project not in configured_projects:
            raise HTTPException(
                status_code=404,
                detail=f"Published project not found: {project}",
            )
        return (project,)
    return configured_projects


def _fetch_published_samples(
    *,
    page: int,
    limit: int,
    project: str | None,
    source_pool: str | None,
    sample_role: str | None,
    task_family: str | None,
    model: str | None,
    result: str | None,
    dataset: str | None,
    hide_pending: bool,
    hide_smoke: bool,
) -> PublishedSamplesResponse:
    page = _parse_positive_int(page, DEFAULT_PAGE)
    limit = min(MAX_LIMIT, _parse_positive_int(limit, DEFAULT_LIMIT))
    offset = (page - 1) * limit
    where, values = _published_conditions(
        project=project,
        source_pool=source_pool,
        sample_role=sample_role,
        task_family=task_family,
        model=model,
        result=result,
        dataset=dataset,
        hide_pending=hide_pending,
        hide_smoke=hide_smoke,
    )
    total = 0
    rows: list[dict[str, Any]] = []
    per_project_limit = offset + limit
    for project_name in _target_published_projects(project):
        with _connect_project_database(project_name) as conn:
            count_row = conn.execute(
                f"SELECT count(*) AS count FROM {PUBLISHED_SAMPLES_TABLE} {where}",
                values,
            ).fetchone()
            total += int(count_row["count"]) if count_row is not None else 0
            rows.extend(
                conn.execute(
                    f"""
                    SELECT
                        source_project,
                        source_pool,
                        source_sample_id,
                        sample_idx,
                        run_id,
                        created_at,
                        status,
                        attempt_count,
                        finish_reason,
                        sample_role,
                        output_kind,
                        dataset_id,
                        task_id,
                        task_family,
                        task_split,
                        language,
                        difficulty,
                        budget_label,
                        budget_chars,
                        provider,
                        model,
                        result_state,
                        passed,
                        validation_pass_rate,
                        failure_category,
                        budget_ok,
                        actual_chars,
                        left(output_text, 500) AS output_text,
                        left(input_text, 500) AS input_text
                    FROM {PUBLISHED_SAMPLES_TABLE}
                    {where}
                    ORDER BY created_at DESC NULLS LAST, source_project,
                             source_pool, sample_idx
                    LIMIT %s
                    """,
                    [*values, per_project_limit],
                ).fetchall()
            )
    rows.sort(
        key=lambda row: (
            row["created_at"].timestamp()
            if row["created_at"] is not None
            else float("-inf"),
            row["source_project"],
            row["source_pool"],
            row["sample_idx"],
        ),
        reverse=True,
    )
    page_rows = rows[offset : offset + limit]
    return PublishedSamplesResponse(
        samples=[PublishedSampleListRow(**row) for row in page_rows],
        total=total,
        page=page,
        limit=limit,
        total_pages=(total + limit - 1) // limit if limit else 0,
    )


def _fetch_published_sample(
    project_name: str,
    source_pool: str,
    sample_id: str,
) -> PublishedSampleDetailResponse | None:
    if project_name not in _published_projects():
        raise HTTPException(
            status_code=404,
            detail=f"Published project not found: {project_name}",
        )
    with _connect_project_database(project_name) as conn:
        row = conn.execute(
            f"""
            SELECT *
            FROM {PUBLISHED_SAMPLES_TABLE}
            WHERE source_project = %s
              AND source_pool = %s
              AND source_sample_id = %s
            """,
            [project_name, source_pool, sample_id],
        ).fetchone()
        failure_rows = conn.execute(
            f"""
            SELECT
                source_project,
                source_pool,
                source_sample_id,
                sample_idx,
                case_key,
                case_idx,
                input_json,
                expected_json,
                actual_json,
                error_text,
                failure_json
            FROM {PUBLISHED_SAMPLE_TEST_FAILURES_TABLE}
            WHERE source_project = %s
              AND source_pool = %s
              AND source_sample_id = %s
            ORDER BY case_idx NULLS LAST, case_key NULLS LAST
            """,
            [project_name, source_pool, sample_id],
        ).fetchall()
    if row is None:
        return None
    return PublishedSampleDetailResponse(
        **row,
        test_failures=[
            PublishedSampleTestFailure(**failure_row)
            for failure_row in failure_rows
        ],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = build_default_registry()
    app.state.registry = registry
    try:
        yield
    finally:
        registry.close()
        if getattr(app.state, "registry", None) is registry:
            app.state.registry = None


app = FastAPI(title="dr-llm UI API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    cast(Any, CORSMiddleware),
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/providers", response_model=list[ProviderStatusResponse])
def list_providers(request: Request) -> list[ProviderStatusResponse]:
    """List all supported providers with availability status."""
    registry = _get_registry(request.app)
    statuses = registry.availability_statuses()
    return [
        ProviderStatusResponse(
            provider=s.provider,
            available=s.available,
            missing_env_vars=list(s.missing_env_vars),
            missing_executables=list(s.missing_executables),
            supports_structured_output=s.supports_structured_output,
        )
        for s in statuses
    ]


@app.get(
    "/api/providers/{provider}/models", response_model=ProviderModelsResponse
)
def get_provider_models(
    provider: str, request: Request
) -> ProviderModelsResponse:
    """Get models for a provider.  Uses static data if the provider is unavailable."""
    registry = _get_registry(request.app)
    try:
        orchestrator = registry.get(provider)
    except KeyError as err:
        raise HTTPException(
            status_code=404, detail=f"Unknown provider: {provider}"
        ) from err

    service = ModelCatalogService(registry=registry)
    is_available = orchestrator.availability_status().available

    if not is_available:
        static = _fallback_model_responses(provider, service)
        return ProviderModelsResponse(
            provider=provider,
            models=static,
            source="static",
        )

    try:
        entries, _raw = service.fetch_provider_models(provider)
    except (ProviderTransportError, ProviderSemanticError) as exc:
        static = _fallback_model_responses(provider, service)
        return ProviderModelsResponse(
            provider=provider,
            models=static,
            source="static",
            error=f"{type(exc).__name__}: {exc}",
        )
    models = [_entry_to_response(e) for e in entries]
    return ProviderModelsResponse(
        provider=provider,
        models=models,
        source="live",
    )


@app.post("/api/providers/{provider}/sync", response_model=SyncResultResponse)
def sync_provider_models(
    provider: str, request: Request
) -> SyncResultResponse:
    """Trigger a live model sync for a provider."""
    registry = _get_registry(request.app)
    try:
        registry.get(provider)
    except KeyError as err:
        raise HTTPException(
            status_code=404, detail=f"Unknown provider: {provider}"
        ) from err

    service = ModelCatalogService(registry=registry)
    try:
        entries, _raw = service.fetch_provider_models(provider)
    except (ProviderTransportError, ProviderSemanticError) as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        static = _fallback_model_responses(provider, service)
        return SyncResultResponse(
            provider=provider,
            success=False,
            model_count=len(static),
            models=static,
            source="static",
            error=error_msg,
        )
    models = [_entry_to_response(e) for e in entries]
    return SyncResultResponse(
        provider=provider,
        success=True,
        model_count=len(models),
        models=models,
        source="live",
    )


@app.get(
    "/api/nl-latents/filters",
    response_model=NlLatentsFiltersResponse,
)
def get_nl_latents_filters() -> NlLatentsFiltersResponse:
    return _fetch_nl_latents_filters()


@app.get(
    "/api/nl-latents/samples",
    response_model=NlLatentsSamplesResponse,
)
def list_nl_latents_samples(
    page: int = DEFAULT_PAGE,
    limit: int = DEFAULT_LIMIT,
    family: str | None = None,
    split: str | None = None,
    enc_model: str | None = None,
    budget: str | None = None,
    difficulty: str | None = None,
    data_version: str | None = None,
    result: str | None = None,
    hide_pending: bool = False,
    hide_smoke: bool = False,
) -> NlLatentsSamplesResponse:
    return _fetch_nl_latents_samples(
        page=page,
        limit=limit,
        family=family,
        split=split,
        enc_model=enc_model,
        budget=budget,
        difficulty=difficulty,
        data_version=data_version,
        result=result,
        hide_pending=hide_pending,
        hide_smoke=hide_smoke,
    )


@app.get(
    "/api/nl-latents/samples/{sample_id}",
    response_model=NlLatentsSampleDetailResponse,
)
def get_nl_latents_sample(sample_id: str) -> NlLatentsSampleDetailResponse:
    sample = _fetch_nl_latents_sample(sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample not found.")
    return sample


@app.get(
    "/api/published/filters",
    response_model=PublishedFiltersResponse,
)
def get_published_filters() -> PublishedFiltersResponse:
    return _fetch_published_filters()


@app.get(
    "/api/published/samples",
    response_model=PublishedSamplesResponse,
)
def list_published_samples(
    page: int = DEFAULT_PAGE,
    limit: int = DEFAULT_LIMIT,
    project: str | None = None,
    source_pool: str | None = None,
    sample_role: str | None = None,
    task_family: str | None = None,
    model: str | None = None,
    result: str | None = None,
    dataset: str | None = None,
    hide_pending: bool = False,
    hide_smoke: bool = False,
) -> PublishedSamplesResponse:
    return _fetch_published_samples(
        page=page,
        limit=limit,
        project=project,
        source_pool=source_pool,
        sample_role=sample_role,
        task_family=task_family,
        model=model,
        result=result,
        dataset=dataset,
        hide_pending=hide_pending,
        hide_smoke=hide_smoke,
    )


@app.get(
    "/api/published/samples/{project_name}/{source_pool}/{sample_id}",
    response_model=PublishedSampleDetailResponse,
)
def get_published_sample(
    project_name: str,
    source_pool: str,
    sample_id: str,
) -> PublishedSampleDetailResponse:
    sample = _fetch_published_sample(project_name, source_pool, sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample not found.")
    return sample
