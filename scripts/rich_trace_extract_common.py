from __future__ import annotations

import re
import gzip
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from humaneval_pool_extract_common import DumpedPoolRow, iter_dump_rows

RICH_TRACE_SPLIT_DIR_NAME = "rich_trace_split"
RICH_TRACES_DIR_NAME = "rich_traces"
CLEAN_DIR_NAME = "clean"
MESSY_DIR_NAME = "messy"
CLEAN_FILE_NAME = "full_encoder_chain.jsonl.gz"
MESSY_FILE_NAME = "messy_decoder_candidates.jsonl.gz"
SUMMARY_FILE_NAME = "summary.json"
MANIFEST_FILE_NAME = "manifest.json"
RICH_ATTEMPTS_FILE_NAME = "rich_trace_attempts.parquet"
BY_DATASET_DIR_NAME = "by_dataset"
PROMPT_KEY = "prompt"
TEXT_KEY = "text"
SOURCE_KIND_ENCODER_SAMPLE = "encoder_sample"
RICH_EXTRACTION_LEVEL_FULL_ENCODER_CHAIN = "full_encoder_chain"
ENCODER_SOURCE_ID_RE = re.compile(
    r"^encoder_pool/(?P<pool_name>[^/]+)/(?P<sample_id>[^/]+)$"
)


class RichTraceClassification(StrEnum):
    FULL_ENCODER_CHAIN = "full_encoder_chain"
    NON_DECODER_ROW = "non_decoder_row"
    MISSING_DECODER_OUTPUT = "missing_decoder_output"
    MISSING_DECODER_PROMPT = "missing_decoder_prompt"
    MISSING_ENCODER_SOURCE_KIND = "missing_encoder_source_kind"
    MISSING_ENCODER_SOURCE_ID = "missing_encoder_source_id"
    MALFORMED_ENCODER_SOURCE_ID = "malformed_encoder_source_id"
    MISSING_ENCODER_ROW = "missing_encoder_row"
    MISSING_ENCODER_PROMPT = "missing_encoder_prompt"
    MISSING_ENCODER_OUTPUT = "missing_encoder_output"


class EncoderReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_name: str
    pool_name: str
    sample_id: str
    raw_source_sample_id: str


class RichTraceCandidateRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    classification: RichTraceClassification
    classification_reasons: list[str] = Field(default_factory=list)
    decoder_row: DumpedPoolRow
    encoder_ref: EncoderReference | None = None
    encoder_row: DumpedPoolRow | None = None


class RichTraceSplitManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_dump_dir: Path
    output_dir: Path
    clean_file_name: str
    messy_file_name: str
    clean_count: int
    messy_count: int


class RichTraceSplitSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_dump_dir: Path
    output_dir: Path
    classification_counts: dict[str, int]
    decoder_pool_counts: dict[str, int]
    clean_dataset_counts: dict[str, int]
    clean_task_count: int


class RichTraceAttemptRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    attempt_id: str
    dataset: str
    task_id: str
    data_sample_id: str
    project_name: str
    decoder_pool_name: str
    decoder_sample_id: str
    encoder_pool_name: str
    encoder_sample_id: str
    sample_idx: int | None = None
    run_id: str | None = None
    created_at: str | None = None
    source_kind: str | None = None
    source_sample_id: str | None = None
    source_pool_name: str | None = None
    enc_prompt_template_id: str | None = None
    dec_prompt_template_id: str | None = None
    enc_llm_config_id: str | None = None
    dec_llm_config_id: str | None = None
    enc_prompt_json: list[dict[str, Any]] | str
    enc_prompt_text: str
    enc_input: str | None = None
    enc_output: str
    dec_prompt_json: list[dict[str, Any]] | str
    dec_prompt_text: str
    dec_input: str
    dec_output: str
    enc_provider: str | None = None
    enc_model: str | None = None
    enc_finish_reason: str | None = None
    enc_usage_json: dict[str, Any] | None = None
    enc_cost_json: dict[str, Any] | None = None
    enc_latency_ms: int | None = None
    dec_provider: str | None = None
    dec_model: str | None = None
    dec_finish_reason: str | None = None
    dec_usage_json: dict[str, Any] | None = None
    dec_cost_json: dict[str, Any] | None = None
    dec_latency_ms: int | None = None
    rich_extraction_level: str = RICH_EXTRACTION_LEVEL_FULL_ENCODER_CHAIN


class DatasetTaskSplitEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: str
    task_id: str
    row_count: int
    file_name: str


class DatasetSplitEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: str
    row_count: int
    all_file_name: str
    task_count: int
    tasks: list[DatasetTaskSplitEntry]


class RichTraceDatasetSplitManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_parquet: Path
    output_dir: Path
    total_rows: int
    datasets: list[DatasetSplitEntry]


def rich_trace_split_dir_for(dump_dir: Path) -> Path:
    return dump_dir / RICH_TRACE_SPLIT_DIR_NAME


def rich_traces_dir_for(dump_dir: Path) -> Path:
    return dump_dir / RICH_TRACES_DIR_NAME


def rich_attempts_path_for(dump_dir: Path) -> Path:
    return rich_traces_dir_for(dump_dir) / RICH_ATTEMPTS_FILE_NAME


def clean_records_path_for(split_dir: Path) -> Path:
    return split_dir / CLEAN_DIR_NAME / CLEAN_FILE_NAME


def messy_records_path_for(split_dir: Path) -> Path:
    return split_dir / MESSY_DIR_NAME / MESSY_FILE_NAME


def split_manifest_path_for(split_dir: Path) -> Path:
    return split_dir / MANIFEST_FILE_NAME


def split_summary_path_for(split_dir: Path) -> Path:
    return split_dir / SUMMARY_FILE_NAME


def by_dataset_dir_for(rich_traces_dir: Path) -> Path:
    return rich_traces_dir / BY_DATASET_DIR_NAME


def is_decoder_candidate(row: DumpedPoolRow) -> bool:
    pool_name = row.pool_name.lower()
    if (
        "decoder" in pool_name
        or pool_name.startswith("dec_")
        or "_dec_" in pool_name
    ):
        return True
    return any(key.startswith("dec_") for key in row.key_values)


def response_text(row: DumpedPoolRow) -> str | None:
    if row.response_json is None:
        return None
    value = row.response_json.get(TEXT_KEY)
    return value if isinstance(value, str) and value else None


def prompt_json(row: DumpedPoolRow) -> list[dict[str, Any]] | str | None:
    value = row.request_json.get(PROMPT_KEY)
    if isinstance(value, str) and value:
        return value
    if isinstance(value, list) and value:
        return value
    return None


def prompt_text(row: DumpedPoolRow) -> str | None:
    value = prompt_json(row)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return "\n\n".join(parts)
    return None


def encoder_reference_for(row: DumpedPoolRow) -> EncoderReference | None:
    raw_source_sample_id = row.metadata_json.get("source_sample_id")
    if not isinstance(raw_source_sample_id, str):
        return None
    match = ENCODER_SOURCE_ID_RE.fullmatch(raw_source_sample_id)
    if match is None:
        return None
    return EncoderReference(
        project_name=row.project_name,
        pool_name=match.group("pool_name"),
        sample_id=match.group("sample_id"),
        raw_source_sample_id=raw_source_sample_id,
    )


def classify_rich_trace_candidate(
    decoder_row: DumpedPoolRow,
    row_index: Mapping[tuple[str, str, str], DumpedPoolRow],
) -> RichTraceCandidateRecord | None:
    if not is_decoder_candidate(decoder_row):
        return None

    reasons: list[RichTraceClassification] = []
    if response_text(decoder_row) is None:
        reasons.append(RichTraceClassification.MISSING_DECODER_OUTPUT)
    if prompt_text(decoder_row) is None:
        reasons.append(RichTraceClassification.MISSING_DECODER_PROMPT)
    if (
        decoder_row.metadata_json.get("source_kind")
        != SOURCE_KIND_ENCODER_SAMPLE
    ):
        reasons.append(RichTraceClassification.MISSING_ENCODER_SOURCE_KIND)

    raw_source_sample_id = decoder_row.metadata_json.get("source_sample_id")
    encoder_ref = encoder_reference_for(decoder_row)
    if raw_source_sample_id is None:
        reasons.append(RichTraceClassification.MISSING_ENCODER_SOURCE_ID)
    elif encoder_ref is None:
        reasons.append(RichTraceClassification.MALFORMED_ENCODER_SOURCE_ID)

    encoder_row: DumpedPoolRow | None = None
    if encoder_ref is not None:
        encoder_row = row_index.get(
            (
                encoder_ref.project_name,
                encoder_ref.pool_name,
                encoder_ref.sample_id,
            )
        )
        if encoder_row is None:
            reasons.append(RichTraceClassification.MISSING_ENCODER_ROW)
        else:
            if prompt_text(encoder_row) is None:
                reasons.append(RichTraceClassification.MISSING_ENCODER_PROMPT)
            if response_text(encoder_row) is None:
                reasons.append(RichTraceClassification.MISSING_ENCODER_OUTPUT)

    if not reasons:
        classification = RichTraceClassification.FULL_ENCODER_CHAIN
    else:
        classification = reasons[0]

    return RichTraceCandidateRecord(
        classification=classification,
        classification_reasons=[reason.value for reason in reasons],
        decoder_row=decoder_row,
        encoder_ref=encoder_ref,
        encoder_row=encoder_row,
    )


def iter_dumped_pool_rows_from_manifest(
    dump_dir: Path,
) -> Iterator[DumpedPoolRow]:
    manifest = json.loads((dump_dir / MANIFEST_FILE_NAME).read_text())
    for pool in manifest.get("pools", []):
        file_name = pool.get("file_name")
        if not isinstance(file_name, str):
            continue
        yield from iter_dump_rows(dump_dir / file_name)


def index_dumped_rows(
    rows: Iterable[DumpedPoolRow],
) -> dict[tuple[str, str, str], DumpedPoolRow]:
    return {
        (row.project_name, row.pool_name, row.sample_id): row for row in rows
    }


def write_jsonl_gz(path: Path, rows: Iterable[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(path, "wt", encoding="utf-8") as file:
        for row in rows:
            file.write(row.model_dump_json() + "\n")
            count += 1
    return count


def iter_rich_trace_records(path: Path) -> Iterator[RichTraceCandidateRecord]:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield RichTraceCandidateRecord.model_validate_json(line)


def data_sample_id_for(row: DumpedPoolRow) -> str | None:
    value = row.metadata_json.get("data_sample_id")
    if isinstance(value, str):
        return value
    value = row.key_values.get("data_sample_id")
    return value if isinstance(value, str) else None


def dataset_and_task_id(data_sample_id: str) -> tuple[str, str]:
    parts = data_sample_id.split("/")
    if len(parts) >= 3:
        return parts[0], "/".join(parts[1:3])
    return "unknown", data_sample_id


def source_text_for(row: DumpedPoolRow) -> str | None:
    value = row.metadata_json.get("source_text")
    if isinstance(value, str) and value:
        return value
    source_payload = row.metadata_json.get("source_sample_payload")
    if isinstance(source_payload, dict):
        payload_text = source_payload.get(TEXT_KEY)
        if isinstance(payload_text, str) and payload_text:
            return payload_text
    return None


def first_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def int_value(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def dict_value(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def build_rich_trace_attempt(
    record: RichTraceCandidateRecord,
) -> RichTraceAttemptRow:
    if record.classification != RichTraceClassification.FULL_ENCODER_CHAIN:
        raise ValueError(
            "rich trace attempts can only be built from full encoder chain records"
        )
    if record.encoder_row is None:
        raise ValueError("full encoder chain record is missing encoder_row")
    if record.encoder_ref is None:
        raise ValueError("full encoder chain record is missing encoder_ref")

    decoder_row = record.decoder_row
    encoder_row = record.encoder_row
    data_sample_id = data_sample_id_for(decoder_row)
    if data_sample_id is None:
        raise ValueError(
            f"decoder row {decoder_row.sample_id!r} has no data_sample_id"
        )
    dataset, task_id = dataset_and_task_id(data_sample_id)
    enc_prompt_json = prompt_json(encoder_row)
    enc_prompt_text = prompt_text(encoder_row)
    enc_output = response_text(encoder_row)
    dec_prompt_json = prompt_json(decoder_row)
    dec_prompt_text = prompt_text(decoder_row)
    dec_input = source_text_for(decoder_row)
    dec_output = response_text(decoder_row)
    if enc_prompt_json is None:
        raise ValueError(
            f"full encoder chain record {decoder_row.sample_id!r} "
            "is missing enc_prompt_json"
        )
    if enc_prompt_text is None:
        raise ValueError(
            f"full encoder chain record {decoder_row.sample_id!r} "
            "is missing enc_prompt_text"
        )
    if enc_output is None:
        raise ValueError(
            f"full encoder chain record {decoder_row.sample_id!r} "
            "is missing enc_output"
        )
    if dec_prompt_json is None:
        raise ValueError(
            f"full encoder chain record {decoder_row.sample_id!r} "
            "is missing dec_prompt_json"
        )
    if dec_prompt_text is None:
        raise ValueError(
            f"full encoder chain record {decoder_row.sample_id!r} "
            "is missing dec_prompt_text"
        )
    if dec_input is None:
        raise ValueError(
            f"full encoder chain record {decoder_row.sample_id!r} "
            "is missing dec_input"
        )
    if dec_output is None:
        raise ValueError(
            f"full encoder chain record {decoder_row.sample_id!r} "
            "is missing dec_output"
        )
    encoder_response = encoder_row.response_json or {}
    decoder_response = decoder_row.response_json or {}
    created_at = (
        decoder_row.created_at.isoformat()
        if decoder_row.created_at is not None
        else None
    )
    return RichTraceAttemptRow(
        attempt_id=(
            f"{decoder_row.project_name}:{decoder_row.pool_name}:"
            f"{decoder_row.sample_id}"
        ),
        dataset=dataset,
        task_id=task_id,
        data_sample_id=data_sample_id,
        project_name=decoder_row.project_name,
        decoder_pool_name=decoder_row.pool_name,
        decoder_sample_id=decoder_row.sample_id,
        encoder_pool_name=encoder_row.pool_name,
        encoder_sample_id=encoder_row.sample_id,
        sample_idx=decoder_row.sample_idx,
        run_id=decoder_row.run_id,
        created_at=created_at,
        source_kind=first_string(decoder_row.metadata_json.get("source_kind")),
        source_sample_id=record.encoder_ref.raw_source_sample_id,
        source_pool_name=first_string(
            decoder_row.metadata_json.get("source_pool_name")
        ),
        enc_prompt_template_id=first_string(
            encoder_row.key_values.get("prompt_template_id"),
            decoder_row.metadata_json.get("enc_prompt_template_id"),
        ),
        dec_prompt_template_id=first_string(
            decoder_row.key_values.get("dec_prompt_template_id")
        ),
        enc_llm_config_id=first_string(
            encoder_row.key_values.get("llm_config_id"),
            decoder_row.metadata_json.get("enc_llm_config_id"),
        ),
        dec_llm_config_id=first_string(
            decoder_row.key_values.get("dec_llm_config_id")
        ),
        enc_prompt_json=enc_prompt_json,
        enc_prompt_text=enc_prompt_text,
        enc_input=None,
        enc_output=enc_output,
        dec_prompt_json=dec_prompt_json,
        dec_prompt_text=dec_prompt_text,
        dec_input=dec_input,
        dec_output=dec_output,
        enc_provider=first_string(encoder_response.get("provider")),
        enc_model=first_string(encoder_response.get("model")),
        enc_finish_reason=encoder_row.finish_reason
        or first_string(encoder_response.get("finish_reason")),
        enc_usage_json=dict_value(encoder_response.get("usage")),
        enc_cost_json=dict_value(encoder_response.get("cost")),
        enc_latency_ms=int_value(encoder_response.get("latency_ms")),
        dec_provider=first_string(decoder_response.get("provider")),
        dec_model=first_string(decoder_response.get("model")),
        dec_finish_reason=decoder_row.finish_reason
        or first_string(decoder_response.get("finish_reason")),
        dec_usage_json=dict_value(decoder_response.get("usage")),
        dec_cost_json=dict_value(decoder_response.get("cost")),
        dec_latency_ms=int_value(decoder_response.get("latency_ms")),
    )


def safe_path_part(value: str) -> str:
    return value.replace("/", "__").replace(" ", "_")


def write_dataframe_parquet(path: Path, rows: Sequence[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [row.model_dump(mode="json") for row in rows]
    pd.DataFrame(payload).to_parquet(path, index=False)
    return len(payload)
