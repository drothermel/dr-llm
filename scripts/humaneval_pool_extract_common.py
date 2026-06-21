from __future__ import annotations

import gzip
import json
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from enum import StrEnum
from itertools import islice
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field
from pydantic_core import to_jsonable_python
from sqlalchemy import text

from dr_llm.datetime_utils import UTC
from dr_llm.pool.admin.discovery import discover_pools
from dr_llm.pool.db.catalog import load_schema
from dr_llm.pool.db.names import PoolTableType, SampleColumn
from dr_llm.pool.db.runtime import DbConfig, DbRuntime
from dr_llm.pool.db.schema import PoolSchema
from dr_llm.project.docker_project_metadata import ContainerStatus
from dr_llm.project.project_info import ProjectInfo
from dr_llm.project.project_service import (
    get_project,
    start_project,
    stop_project,
)

DEFAULT_OUTPUT_ROOT = (
    Path.home() / "drotherm/data/code-comp/dr-llm-humaneval-pool-dumps"
)
MANIFEST_FILE_NAME = "manifest.json"
PREVIEW_FILE_NAME = "humaneval_code_attempts_preview.csv"
PARQUET_FILE_NAME = "humaneval_code_attempts.parquet"

HUMAN_EVAL_ID_RE = re.compile(
    r"(?<![A-Za-z0-9_])human_eval/(HumanEval/\d+)(?=/|$)"
)
HUMAN_EVAL_PRO_ID_RE = re.compile(
    r"(?<![A-Za-z0-9_])humaneval_pro/(HumanEvalPro/\d+)(?=/|$)"
)
DATASET_ID_COLUMNS = (
    "data_sample_id",
    "source_sample_id",
    "enc_sample_id",
    "task_id",
)


class OutputKind(StrEnum):
    CODE_TEXT = "code_text"
    DECODED_CODE = "decoded_code"
    NOT_CODE = "not_code"


class DecoderDescriptionSource(StrEnum):
    METADATA_SOURCE_TEXT = "metadata.source_text"
    SOURCE_SAMPLE_PAYLOAD_TEXT = "metadata.source_sample_payload.text"
    REQUEST_PROMPT = "request.prompt"
    HUMANEVAL_CACHE_PROMPT = "humaneval_cache.prompt"
    MISSING = "missing"


class CandidatePoolSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_name: str
    pool_names: list[str] | None = None


class ProjectLease(BaseModel):
    model_config = ConfigDict(frozen=True)

    project: ProjectInfo
    original_status: ContainerStatus
    temporarily_started: bool


class PoolTarget(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_name: str
    pool_name: str
    pool_schema: PoolSchema
    original_status: ContainerStatus
    temporarily_started: bool

    @computed_field
    @property
    def table_name(self) -> str:
        return self.pool_schema.table_name(PoolTableType.SAMPLES)


class DumpedPoolManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_name: str
    pool_name: str
    table_name: str
    file_name: str
    row_count: int
    dumped_row_count: int
    pool_schema_json: dict[str, Any]
    original_status: ContainerStatus
    temporarily_started: bool


class DumpManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: int = 1
    created_at: datetime
    output_dir: Path
    pools: list[DumpedPoolManifest] = Field(default_factory=list)


class DumpRowHints(BaseModel):
    model_config = ConfigDict(frozen=True)

    human_eval_task_id: str | None = None
    human_eval_pro_task_id: str | None = None
    output_kind: OutputKind
    output_json_path: str | None = None
    decoder_input_description_source: DecoderDescriptionSource


class DumpedPoolRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_name: str
    pool_name: str
    table_name: str
    sample_id: str
    key_values: dict[str, Any]
    sample_idx: int | None = None
    run_id: str | None = None
    request_json: dict[str, Any]
    response_json: dict[str, Any] | None = None
    finish_reason: str | None = None
    attempt_count: int = 0
    metadata_json: dict[str, Any]
    created_at: datetime | None = None
    hints: DumpRowHints


class CodeAttemptRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    attempt_id: str
    project_name: str
    pool_name: str
    pool_sample_id: str
    sample_idx: int | None
    run_id: str | None
    created_at: datetime | None
    human_eval_task_id: str
    data_sample_id: str | None = None
    source_sample_id: str | None = None
    prompt_template_id: str | None = None
    enc_prompt_template_id: str | None = None
    dec_prompt_template_id: str | None = None
    llm_config_id: str | None = None
    enc_llm_config_id: str | None = None
    dec_llm_config_id: str | None = None
    model: str | None = None
    provider: str | None = None
    finish_reason: str | None = None
    attempt_count: int
    output_json_path: str
    extraction_policy: str
    raw_code_output: str
    decoder_input_description: str | None
    decoder_input_description_source: DecoderDescriptionSource
    prompt_fingerprint: str | None = None
    source_pool_name: str | None = None
    source_kind: str | None = None


class PoolPolicySummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_name: str
    pool_name: str
    row_count: int
    key_columns: list[str]
    key_human_eval_count: int
    text_human_eval_sample_count: int
    text_human_eval_pro_sample_count: int
    decoder_candidate_sample_count: int
    decoder_description_sample_count: int
    sample_key_values: dict[str, list[str]]
    request_top_keys: list[str]
    response_top_keys: list[str]
    metadata_top_keys: list[str]


def default_candidate_specs() -> list[CandidatePoolSpec]:
    return [
        CandidatePoolSpec(project_name="code_comp_t1"),
        CandidatePoolSpec(project_name="code_comp_v0"),
        CandidatePoolSpec(
            project_name="nl_latents", pool_names=["nl_latents"]
        ),
    ]


def timestamped_output_dir(root: Path) -> Path:
    return root / datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


@contextmanager
def running_project(project_name: str) -> Iterator[ProjectLease]:
    project = get_project(project_name)
    original_status = project.status
    temporarily_started = False
    running = project
    try:
        if project.status != ContainerStatus.RUNNING:
            running = start_project(project_name)
            temporarily_started = True
        yield ProjectLease(
            project=running,
            original_status=original_status,
            temporarily_started=temporarily_started,
        )
    finally:
        if temporarily_started:
            stop_project(project_name)


def resolve_pool_targets(specs: list[CandidatePoolSpec]) -> list[PoolTarget]:
    targets: list[PoolTarget] = []
    for spec in specs:
        with running_project(spec.project_name) as lease:
            dsn = require_dsn(lease.project)
            runtime = DbRuntime(
                DbConfig(dsn=dsn, application_name="humaneval_extract")
            )
            try:
                pool_names = spec.pool_names or discover_pools(dsn)
                for pool_name in pool_names:
                    schema = load_schema(runtime, pool_name)
                    if schema is None:
                        raise RuntimeError(
                            f"Pool {spec.project_name}/{pool_name} has no catalog schema"
                        )
                    targets.append(
                        PoolTarget(
                            project_name=spec.project_name,
                            pool_name=pool_name,
                            pool_schema=schema,
                            original_status=lease.original_status,
                            temporarily_started=lease.temporarily_started,
                        )
                    )
            finally:
                runtime.close()
    return targets


def require_dsn(project: ProjectInfo) -> str:
    if project.dsn is None:
        raise RuntimeError(f"Project {project.name!r} has no DSN")
    return project.dsn


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sample_table_name(schema: PoolSchema) -> str:
    return schema.table_name(PoolTableType.SAMPLES)


def row_count(runtime: DbRuntime, table_name: str) -> int:
    with runtime.connect() as conn:
        return int(
            conn.execute(
                text(f"SELECT count(*) FROM {quote_ident(table_name)}")
            ).scalar_one()
        )


def stream_sample_rows(
    runtime: DbRuntime,
    *,
    table_name: str,
    batch_size: int,
) -> Iterator[Any]:
    stmt = text(f"SELECT * FROM {quote_ident(table_name)} ORDER BY sample_id")
    with runtime.connect() as conn:
        result = conn.execution_options(stream_results=True).execute(stmt)
        while True:
            batch = result.mappings().fetchmany(batch_size)
            if not batch:
                break
            yield from batch


def row_to_dumped_pool_row(
    *,
    project_name: str,
    pool_name: str,
    schema: PoolSchema,
    row: Any,
) -> DumpedPoolRow:
    row_dict = dict(row)
    key_values = {
        key: to_jsonable_python(row_dict.get(key))
        for key in schema.key_column_names
    }
    request_json = as_dict(row_dict.get(SampleColumn.REQUEST_JSON))
    response_json = as_optional_dict(row_dict.get(SampleColumn.RESPONSE_JSON))
    metadata_json = as_dict(row_dict.get(SampleColumn.METADATA_JSON))
    return DumpedPoolRow(
        project_name=project_name,
        pool_name=pool_name,
        table_name=sample_table_name(schema),
        sample_id=str(row_dict[SampleColumn.SAMPLE_ID]),
        key_values=key_values,
        sample_idx=row_dict.get(SampleColumn.SAMPLE_IDX),
        run_id=row_dict.get(SampleColumn.RUN_ID),
        request_json=request_json,
        response_json=response_json,
        finish_reason=row_dict.get(SampleColumn.FINISH_REASON),
        attempt_count=int(row_dict.get(SampleColumn.ATTEMPT_COUNT) or 0),
        metadata_json=metadata_json,
        created_at=row_dict.get(SampleColumn.CREATED_AT),
        hints=build_dump_row_hints(
            pool_name=pool_name,
            key_values=key_values,
            request_json=request_json,
            response_json=response_json,
            metadata_json=metadata_json,
        ),
    )


def build_dump_row_hints(
    *,
    pool_name: str,
    key_values: dict[str, Any],
    request_json: dict[str, Any],
    response_json: dict[str, Any] | None,
    metadata_json: dict[str, Any],
) -> DumpRowHints:
    combined = {
        "key_values": key_values,
        "request_json": request_json,
        "response_json": response_json,
        "metadata_json": metadata_json,
    }
    human_eval_task_id = find_human_eval_task_id(combined)
    human_eval_pro_task_id = find_human_eval_pro_task_id(combined)
    output_path, output_kind = output_path_and_kind(
        pool_name=pool_name,
        key_values=key_values,
        response_json=response_json,
    )
    _description, description_source = decoder_input_description(
        request_json=request_json,
        metadata_json=metadata_json,
    )
    return DumpRowHints(
        human_eval_task_id=human_eval_task_id,
        human_eval_pro_task_id=human_eval_pro_task_id,
        output_kind=output_kind,
        output_json_path=output_path,
        decoder_input_description_source=description_source,
    )


def as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return to_jsonable_python(value)
    return {}


def as_optional_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return to_jsonable_python(value)
    return None


def write_json_line(file: gzip.GzipFile, payload: BaseModel) -> None:
    line = json.dumps(payload.model_dump(mode="json"), sort_keys=True)
    file.write((line + "\n").encode("utf-8"))


def iter_dump_rows(path: Path) -> Iterator[DumpedPoolRow]:
    with gzip.open(path, mode="rt", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield DumpedPoolRow.model_validate_json(line)


def read_manifest(path: Path) -> DumpManifest:
    return DumpManifest.model_validate_json(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, manifest: DumpManifest) -> None:
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def find_human_eval_task_id(value: Any) -> str | None:
    for text_value in iter_strings(value):
        match = HUMAN_EVAL_ID_RE.search(text_value)
        if match is not None:
            return match.group(1)
    return None


def find_human_eval_pro_task_id(value: Any) -> str | None:
    for text_value in iter_strings(value):
        match = HUMAN_EVAL_PRO_ID_RE.search(text_value)
        if match is not None:
            return match.group(1)
    return None


def iter_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str):
                yield key
            yield from iter_strings(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from iter_strings(item)


def output_path_and_kind(
    *,
    pool_name: str,
    key_values: dict[str, Any],
    response_json: dict[str, Any] | None,
) -> tuple[str | None, OutputKind]:
    if response_json is None:
        return None, OutputKind.NOT_CODE
    if pool_name == "nl_latents" and isinstance(
        response_json.get("decoded_code"), str
    ):
        return "response_json.decoded_code", OutputKind.DECODED_CODE
    if not is_decoder_pool_row(pool_name=pool_name, key_values=key_values):
        return None, OutputKind.NOT_CODE
    if isinstance(response_json.get("text"), str):
        return "response_json.text", OutputKind.CODE_TEXT
    return None, OutputKind.NOT_CODE


def is_decoder_pool_row(*, pool_name: str, key_values: dict[str, Any]) -> bool:
    lowered_pool = pool_name.lower()
    if "decoder" in lowered_pool or lowered_pool.startswith("dec_"):
        return True
    return any(key.startswith("dec_") for key in key_values)


def raw_output_from_path(
    row: DumpedPoolRow, output_path: str | None
) -> str | None:
    if row.response_json is None or output_path is None:
        return None
    if output_path == "response_json.text":
        value = row.response_json.get("text")
    elif output_path == "response_json.decoded_code":
        value = row.response_json.get("decoded_code")
    else:
        value = None
    return value if isinstance(value, str) and value else None


def decoder_input_description(
    *,
    request_json: dict[str, Any],
    metadata_json: dict[str, Any],
) -> tuple[str | None, DecoderDescriptionSource]:
    source_text = metadata_json.get("source_text")
    if isinstance(source_text, str) and source_text:
        return source_text, DecoderDescriptionSource.METADATA_SOURCE_TEXT

    source_payload = metadata_json.get("source_sample_payload")
    if isinstance(source_payload, dict):
        payload_text = source_payload.get("text")
        if isinstance(payload_text, str) and payload_text:
            return (
                payload_text,
                DecoderDescriptionSource.SOURCE_SAMPLE_PAYLOAD_TEXT,
            )

    prompt_text = prompt_to_text(request_json.get("prompt"))
    if prompt_text:
        return prompt_text, DecoderDescriptionSource.REQUEST_PROMPT

    return None, DecoderDescriptionSource.MISSING


def prompt_to_text(prompt: Any) -> str | None:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts: list[str] = []
        for item in prompt:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return "\n\n".join(parts)
    return None


def data_sample_id(row: DumpedPoolRow) -> str | None:
    value = row.key_values.get("data_sample_id")
    if isinstance(value, str):
        return value
    value = row.metadata_json.get("data_sample_id")
    return value if isinstance(value, str) else None


def source_sample_id(row: DumpedPoolRow) -> str | None:
    value = row.key_values.get("source_sample_id")
    if isinstance(value, str):
        return value
    value = row.metadata_json.get("source_sample_id")
    return value if isinstance(value, str) else None


def first_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def model_value(row: DumpedPoolRow, key: str) -> str | None:
    if row.response_json is None:
        return None
    value = row.response_json.get(key)
    return value if isinstance(value, str) else None


def build_code_attempt(
    row: DumpedPoolRow,
    *,
    humaneval_prompts_by_task_id: Mapping[str, str] | None = None,
) -> CodeAttemptRow | None:
    if row.hints.human_eval_task_id is None:
        return None
    if row.hints.human_eval_pro_task_id is not None:
        return None
    raw_code_output = raw_output_from_path(row, row.hints.output_json_path)
    if raw_code_output is None or row.hints.output_json_path is None:
        return None
    description, description_source = decoder_input_description(
        request_json=row.request_json,
        metadata_json=row.metadata_json,
    )
    if (
        description is None
        and humaneval_prompts_by_task_id is not None
        and row.hints.human_eval_task_id in humaneval_prompts_by_task_id
    ):
        description = humaneval_prompts_by_task_id[
            row.hints.human_eval_task_id
        ]
        description_source = DecoderDescriptionSource.HUMANEVAL_CACHE_PROMPT
    prompt_fingerprint = first_string(
        row.key_values.get("prompt_template_id"),
        row.key_values.get("dec_prompt_template_id"),
        row.key_values.get("enc_prompt_template_id"),
    )
    return CodeAttemptRow(
        attempt_id=f"{row.project_name}:{row.pool_name}:{row.sample_id}",
        project_name=row.project_name,
        pool_name=row.pool_name,
        pool_sample_id=row.sample_id,
        sample_idx=row.sample_idx,
        run_id=row.run_id,
        created_at=row.created_at,
        human_eval_task_id=row.hints.human_eval_task_id,
        data_sample_id=data_sample_id(row),
        source_sample_id=source_sample_id(row),
        prompt_template_id=first_string(
            row.key_values.get("prompt_template_id")
        ),
        enc_prompt_template_id=first_string(
            row.key_values.get("enc_prompt_template_id"),
            row.metadata_json.get("enc_prompt_template_id"),
        ),
        dec_prompt_template_id=first_string(
            row.key_values.get("dec_prompt_template_id")
        ),
        llm_config_id=first_string(row.key_values.get("llm_config_id")),
        enc_llm_config_id=first_string(
            row.key_values.get("enc_llm_config_id"),
            row.metadata_json.get("enc_llm_config_id"),
        ),
        dec_llm_config_id=first_string(
            row.key_values.get("dec_llm_config_id")
        ),
        model=model_value(row, "model"),
        provider=model_value(row, "provider"),
        finish_reason=row.finish_reason,
        attempt_count=row.attempt_count,
        output_json_path=row.hints.output_json_path,
        extraction_policy=row.hints.output_kind.value,
        raw_code_output=raw_code_output,
        decoder_input_description=description,
        decoder_input_description_source=description_source,
        prompt_fingerprint=prompt_fingerprint,
        source_pool_name=first_string(
            row.metadata_json.get("source_pool_name")
        ),
        source_kind=first_string(row.metadata_json.get("source_kind")),
    )


def top_level_keys(values: list[Any]) -> list[str]:
    counts: dict[str, int] = {}
    for value in values:
        if isinstance(value, dict):
            for key in value:
                counts[key] = counts.get(key, 0) + 1
    return [
        key
        for key, _count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )[:20]
    ]


def collect_sample_key_values(
    rows: list[DumpedPoolRow],
    key_columns: list[str],
) -> dict[str, list[str]]:
    sample_values: dict[str, list[str]] = {}
    for key in key_columns:
        values = sorted(
            {
                str(row.key_values[key])
                for row in rows
                if row.key_values.get(key) is not None
            }
        )
        sample_values[key] = values[:5]
    return sample_values


def count_key_human_eval_rows(
    runtime: DbRuntime,
    *,
    table_name: str,
    key_columns: list[str],
) -> int:
    eligible_columns = [
        col for col in key_columns if col in DATASET_ID_COLUMNS
    ]
    if not eligible_columns:
        return 0
    clauses = [
        f"{quote_ident(column)} LIKE 'human_eval/HumanEval/%'"
        for column in eligible_columns
    ]
    stmt = text(
        f"SELECT count(*) FROM {quote_ident(table_name)} "
        f"WHERE {' OR '.join(clauses)}"
    )
    with runtime.connect() as conn:
        return int(conn.execute(stmt).scalar_one())


def build_pool_policy_summary(
    *,
    target: PoolTarget,
    runtime: DbRuntime,
    sample_limit: int,
) -> PoolPolicySummary:
    table_name = sample_table_name(target.pool_schema)
    rows = [
        row_to_dumped_pool_row(
            project_name=target.project_name,
            pool_name=target.pool_name,
            schema=target.pool_schema,
            row=row,
        )
        for row in islice(
            stream_sample_rows(
                runtime,
                table_name=table_name,
                batch_size=max(1, sample_limit),
            ),
            sample_limit,
        )
    ]
    return PoolPolicySummary(
        project_name=target.project_name,
        pool_name=target.pool_name,
        row_count=row_count(runtime, table_name),
        key_columns=target.pool_schema.key_column_names,
        key_human_eval_count=count_key_human_eval_rows(
            runtime,
            table_name=table_name,
            key_columns=target.pool_schema.key_column_names,
        ),
        text_human_eval_sample_count=sum(
            row.hints.human_eval_task_id is not None for row in rows
        ),
        text_human_eval_pro_sample_count=sum(
            row.hints.human_eval_pro_task_id is not None for row in rows
        ),
        decoder_candidate_sample_count=sum(
            row.hints.output_kind != OutputKind.NOT_CODE for row in rows
        ),
        decoder_description_sample_count=sum(
            row.hints.decoder_input_description_source
            != DecoderDescriptionSource.MISSING
            for row in rows
        ),
        sample_key_values=collect_sample_key_values(
            rows, target.pool_schema.key_column_names
        ),
        request_top_keys=top_level_keys([row.request_json for row in rows]),
        response_top_keys=top_level_keys(
            [
                row.response_json
                for row in rows
                if row.response_json is not None
            ]
        ),
        metadata_top_keys=top_level_keys([row.metadata_json for row in rows]),
    )


def manifest_path_for(output_dir: Path) -> Path:
    return output_dir / MANIFEST_FILE_NAME


def dump_file_name(project_name: str, pool_name: str) -> str:
    return f"{project_name}__{pool_name}.jsonl.gz"


def parquet_path_for(output_dir: Path) -> Path:
    return output_dir / PARQUET_FILE_NAME


def preview_path_for(output_dir: Path) -> Path:
    return output_dir / PREVIEW_FILE_NAME


__all__ = [
    "CandidatePoolSpec",
    "CodeAttemptRow",
    "DEFAULT_OUTPUT_ROOT",
    "DumpManifest",
    "DumpedPoolManifest",
    "DumpedPoolRow",
    "MANIFEST_FILE_NAME",
    "PARQUET_FILE_NAME",
    "PREVIEW_FILE_NAME",
    "PoolTarget",
    "build_code_attempt",
    "build_pool_policy_summary",
    "default_candidate_specs",
    "dump_file_name",
    "iter_dump_rows",
    "manifest_path_for",
    "parquet_path_for",
    "preview_path_for",
    "read_manifest",
    "require_dsn",
    "resolve_pool_targets",
    "row_count",
    "row_to_dumped_pool_row",
    "running_project",
    "sample_table_name",
    "stream_sample_rows",
    "timestamped_output_dir",
    "write_json_line",
    "write_manifest",
]
