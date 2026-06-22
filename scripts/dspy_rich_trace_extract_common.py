from __future__ import annotations

import gzip
import re
from collections.abc import Iterable, Iterator, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_DSPY_CORPUS_ROOT = Path(
    "/Users/daniellerothermel/drotherm/data/code-comp/dspy-exps/v0"
)
PARSED_EVAL_REPORTS_DIR_NAME = "parsed_eval_reports"
DSPY_RICH_TRACE_SPLIT_DIR_NAME = "dspy_rich_trace_split"
DSPY_RICH_TRACES_DIR_NAME = "dspy_rich_traces"
CLEAN_DIR_NAME = "clean"
MESSY_DIR_NAME = "messy"
CLEAN_EVAL_ATTEMPTS_FILE_NAME = "eval_attempts.jsonl.gz"
MESSY_EVAL_ATTEMPTS_FILE_NAME = "eval_attempts.jsonl.gz"
SUMMARY_FILE_NAME = "summary.json"
MANIFEST_FILE_NAME = "manifest.json"
DSPY_RICH_ATTEMPTS_FILE_NAME = "dspy_rich_trace_attempts.parquet"
BY_DATASET_DIR_NAME = "by_dataset"
DATASET_NAME = "human_eval"
PROJECT_NAME = "dspy_exps_v0"
SOURCE_KIND_EVAL_ATTEMPT = "dspy_eval_attempt"
GENERATION_TYPE_DIRECT = "direct"
GENERATION_TYPE_ENCDEC = "encdec"
PROMPT_KIND_DIRECT = "direct_code_from_stub"
PROMPT_KIND_ENCODE = "encode_code_spec"
PROMPT_KIND_DECODE = "decode_code_spec"
FIELD_CODE_SPEC = "code_spec"
FIELD_FUNCTION_STUB = "function_stub"
FIELD_COMPLETED_CODE = "completed_code"
FIELD_RE = re.compile(
    r"\[\[\s*##\s*(?P<field>[^#]+?)\s*##\s*\]\](?P<body>.*?)(?=\[\[\s*##|\Z)",
    re.DOTALL,
)


class DspyRichTraceClassification(StrEnum):
    DIRECT_EVAL_ATTEMPT = "direct_eval_attempt"
    ENCDEC_EVAL_ATTEMPT = "encdec_eval_attempt"
    SKIPPED_ATTEMPT = "skipped_attempt"
    MISSING_RAW_COMPLETED_CODE = "missing_raw_completed_code"
    MISSING_TEST_PASS_RATE = "missing_test_pass_rate"
    MISSING_GENERATION_CALL_IDS = "missing_generation_call_ids"
    MISSING_GENERATION_CALL_RECORD = "missing_generation_call_record"
    UNSUPPORTED_GENERATION_TYPE = "unsupported_generation_type"
    UNEXPECTED_PROMPT_KINDS = "unexpected_prompt_kinds"


class DspyGenerationFamily(StrEnum):
    DIRECT = "direct"
    ENCDEC = "encdec"


class DspyReportSession(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    session_id: str | None = None


class DspyEvalRun(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    source_relative_path: str | None = None
    run_format: str | None = None
    timestamp: str | None = None
    generation_type: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    model_names: list[str] = Field(default_factory=list)


class DspyEvalAttempt(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    run_id: str | None = None
    run_format: str | None = None
    source_relative_path: str | None = None
    timestamp: str | None = None
    generation_type: str | None = None
    dataset_index: int | None = None
    task_id: str
    repeat_index: int | None = None
    skipped: bool = False
    error: Any = None
    code_spec: str | None = None
    raw_completed_code: str | None = None
    extracted_code: str | None = None
    test_pass_rate: float | None = None
    test_case_results: list[dict[str, Any]] = Field(default_factory=list)
    generation_log_file: str | None = None
    generation_log_source_relative_path: str | None = None
    generation_call_ids: list[str] = Field(default_factory=list)


class DspyGenerationCall(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    source_relative_path: str | None = None
    record_index: int | None = None
    timestamp: str | None = None
    uuid: str | None = None
    model: str | None = None
    response_model: str | None = None
    model_type: str | None = None
    prompt_fingerprint: str | None = None
    prompt_kind: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    outputs: list[Any] = Field(default_factory=list)
    response: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    cost: Any = None
    attempt: dict[str, Any] | None = None


class DspyEvalReport(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    schema_version: str | None = None
    created_at: str | None = None
    session: DspyReportSession = Field(default_factory=DspyReportSession)
    runs: list[DspyEvalRun] = Field(default_factory=list)
    attempts: list[DspyEvalAttempt] = Field(default_factory=list)
    generation_calls: list[DspyGenerationCall] = Field(default_factory=list)


class DspyRichTraceCandidateRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    classification: DspyRichTraceClassification
    classification_reasons: list[str] = Field(default_factory=list)
    report_path: Path
    report_relative_path: str
    session_id: str
    attempt: DspyEvalAttempt
    run: DspyEvalRun | None = None
    generation_calls: list[DspyGenerationCall] = Field(default_factory=list)
    encoder_call: DspyGenerationCall | None = None
    decoder_call: DspyGenerationCall | None = None


class DspyRichTraceSplitManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_corpus_root: Path
    output_dir: Path
    clean_file_name: str
    messy_file_name: str
    clean_count: int
    messy_count: int


class DspyRichTraceSplitSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_corpus_root: Path
    output_dir: Path
    report_count: int
    classification_counts: dict[str, int]
    generation_type_counts: dict[str, int]
    clean_dataset_counts: dict[str, int]
    clean_task_count: int


class DspyRichTraceAttemptRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    attempt_id: str
    dataset: str
    task_id: str
    data_sample_id: str
    project_name: str = PROJECT_NAME
    decoder_pool_name: str | None = None
    decoder_sample_id: str | None = None
    encoder_pool_name: str | None = None
    encoder_sample_id: str | None = None
    sample_idx: int | None = None
    run_id: str | None = None
    created_at: str | None = None
    source_kind: str = SOURCE_KIND_EVAL_ATTEMPT
    source_sample_id: str | None = None
    source_pool_name: str | None = None
    enc_prompt_template_id: str | None = None
    dec_prompt_template_id: str | None = None
    enc_llm_config_id: str | None = None
    dec_llm_config_id: str | None = None
    enc_prompt_json: list[dict[str, Any]] | str | None = None
    enc_prompt_text: str | None = None
    enc_input: str | None = None
    enc_output: str | None = None
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
    rich_extraction_level: str
    dspy_session_id: str
    dspy_report_path: str
    dspy_attempt_id: str
    dspy_generation_type: str
    dspy_dataset_index: int | None = None
    dspy_repeat_index: int | None = None
    dspy_test_pass_rate: float
    dspy_raw_completed_code: str
    dspy_extracted_code: str | None = None
    dspy_run_format: str | None = None
    dspy_generation_log_source_relative_path: str | None = None
    dspy_generation_call_ids: list[str]
    dspy_encoder_call_id: str | None = None
    dspy_decoder_call_id: str | None = None
    dspy_encoder_call_output: str | None = None
    dspy_decoder_call_output: str | None = None


class DspyDatasetTaskSplitEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: str
    task_id: str
    row_count: int
    file_name: str


class DspyDatasetSplitEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: str
    row_count: int
    all_file_name: str
    task_count: int
    tasks: list[DspyDatasetTaskSplitEntry]


class DspyRichTraceDatasetSplitManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_parquet: Path
    output_dir: Path
    total_rows: int
    datasets: list[DspyDatasetSplitEntry]


def dspy_rich_trace_split_dir_for(corpus_root: Path) -> Path:
    return corpus_root / DSPY_RICH_TRACE_SPLIT_DIR_NAME


def dspy_rich_traces_dir_for(corpus_root: Path) -> Path:
    return corpus_root / DSPY_RICH_TRACES_DIR_NAME


def dspy_rich_attempts_path_for(corpus_root: Path) -> Path:
    return dspy_rich_traces_dir_for(corpus_root) / DSPY_RICH_ATTEMPTS_FILE_NAME


def clean_eval_attempts_path_for(split_dir: Path) -> Path:
    return split_dir / CLEAN_DIR_NAME / CLEAN_EVAL_ATTEMPTS_FILE_NAME


def messy_eval_attempts_path_for(split_dir: Path) -> Path:
    return split_dir / MESSY_DIR_NAME / MESSY_EVAL_ATTEMPTS_FILE_NAME


def split_manifest_path_for(split_dir: Path) -> Path:
    return split_dir / MANIFEST_FILE_NAME


def split_summary_path_for(split_dir: Path) -> Path:
    return split_dir / SUMMARY_FILE_NAME


def by_dataset_dir_for(rich_traces_dir: Path) -> Path:
    return rich_traces_dir / BY_DATASET_DIR_NAME


def iter_eval_report_paths(corpus_root: Path) -> Iterator[Path]:
    report_dir = corpus_root / PARSED_EVAL_REPORTS_DIR_NAME
    yield from sorted(report_dir.glob("*.eval_report.json"))


def read_eval_report(path: Path) -> DspyEvalReport:
    return DspyEvalReport.model_validate_json(path.read_text(encoding="utf-8"))


def iter_eval_reports(
    corpus_root: Path,
) -> Iterator[tuple[Path, DspyEvalReport]]:
    for path in iter_eval_report_paths(corpus_root):
        yield path, read_eval_report(path)


def report_relative_path(path: Path, corpus_root: Path) -> str:
    return path.relative_to(corpus_root).as_posix()


def prompt_text(messages: list[dict[str, Any]]) -> str | None:
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) and content:
            parts.append(content)
    if not parts:
        return None
    return "\n\n".join(parts)


def generation_call_output(call: DspyGenerationCall) -> str | None:
    for output in call.outputs:
        if isinstance(output, str) and output:
            return output
    response = call.response or {}
    choices = response.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content:
                    return content
    return None


def extract_dspy_field(text: str | None, field_name: str) -> str | None:
    if not text:
        return None
    for match in FIELD_RE.finditer(text):
        field = match.group("field").strip()
        if field == field_name:
            value = match.group("body").strip()
            return value if value else None
    return None


def extract_field_from_messages(
    messages: list[dict[str, Any]],
    field_name: str,
) -> str | None:
    for message in reversed(messages):
        content = message.get("content")
        value = extract_dspy_field(
            content if isinstance(content, str) else None,
            field_name,
        )
        if value is not None:
            return value
    return None


def provider_for_model(model: str | None) -> str | None:
    if not model:
        return None
    provider, separator, _ = model.partition("/")
    return provider if separator else None


def finish_reason_for(call: DspyGenerationCall | None) -> str | None:
    if call is None or call.response is None:
        return None
    choices = call.response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None
    finish_reason = first_choice.get("finish_reason")
    return finish_reason if isinstance(finish_reason, str) else None


def cost_json_for(call: DspyGenerationCall | None) -> dict[str, Any] | None:
    if call is None:
        return None
    if isinstance(call.cost, dict):
        return call.cost
    if isinstance(call.cost, int | float):
        return {"cost": float(call.cost)}
    usage_cost = (call.usage or {}).get("cost")
    if isinstance(usage_cost, int | float):
        return {"cost": float(usage_cost)}
    return None


def data_sample_id_for(task_id: str) -> str:
    return f"{DATASET_NAME}/{task_id}"


def run_generation_type(
    attempt: DspyEvalAttempt, run: DspyEvalRun | None
) -> str | None:
    if attempt.generation_type:
        return attempt.generation_type
    if run is not None and run.generation_type:
        return run.generation_type
    if run is not None and isinstance(run.config.get("generation_type"), str):
        return run.config["generation_type"]
    return None


def classify_dspy_eval_attempt(
    *,
    corpus_root: Path,
    report_path: Path,
    report: DspyEvalReport,
    attempt: DspyEvalAttempt,
) -> DspyRichTraceCandidateRecord:
    session_id = report.session.session_id or report_path.stem
    runs_by_id = {run.id: run for run in report.runs}
    calls_by_id = {call.id: call for call in report.generation_calls}
    run = runs_by_id.get(attempt.run_id or "")
    generation_type = run_generation_type(attempt, run)
    reasons: list[DspyRichTraceClassification] = []

    if attempt.skipped:
        reasons.append(DspyRichTraceClassification.SKIPPED_ATTEMPT)
    if not attempt.raw_completed_code:
        reasons.append(DspyRichTraceClassification.MISSING_RAW_COMPLETED_CODE)
    if attempt.test_pass_rate is None:
        reasons.append(DspyRichTraceClassification.MISSING_TEST_PASS_RATE)
    if generation_type not in {GENERATION_TYPE_DIRECT, GENERATION_TYPE_ENCDEC}:
        reasons.append(DspyRichTraceClassification.UNSUPPORTED_GENERATION_TYPE)

    generation_call_ids = attempt.generation_call_ids
    generation_calls: list[DspyGenerationCall] = []
    if not generation_call_ids:
        reasons.append(DspyRichTraceClassification.MISSING_GENERATION_CALL_IDS)
    else:
        for call_id in generation_call_ids:
            call = calls_by_id.get(call_id)
            if call is None:
                reasons.append(
                    DspyRichTraceClassification.MISSING_GENERATION_CALL_RECORD
                )
            else:
                generation_calls.append(call)

    encoder_call: DspyGenerationCall | None = None
    decoder_call: DspyGenerationCall | None = None
    if generation_calls and generation_type == GENERATION_TYPE_ENCDEC:
        encoder_calls = [
            call
            for call in generation_calls
            if call.prompt_kind == PROMPT_KIND_ENCODE
        ]
        decoder_calls = [
            call
            for call in generation_calls
            if call.prompt_kind == PROMPT_KIND_DECODE
        ]
        if encoder_calls and decoder_calls:
            encoder_call = encoder_calls[0]
            decoder_call = decoder_calls[-1]
        else:
            reasons.append(DspyRichTraceClassification.UNEXPECTED_PROMPT_KINDS)
    elif generation_calls and generation_type == GENERATION_TYPE_DIRECT:
        direct_calls = [
            call
            for call in generation_calls
            if call.prompt_kind == PROMPT_KIND_DIRECT
        ]
        if direct_calls:
            decoder_call = direct_calls[-1]
        else:
            reasons.append(DspyRichTraceClassification.UNEXPECTED_PROMPT_KINDS)

    if reasons:
        classification = reasons[0]
    elif generation_type == GENERATION_TYPE_ENCDEC:
        classification = DspyRichTraceClassification.ENCDEC_EVAL_ATTEMPT
    else:
        classification = DspyRichTraceClassification.DIRECT_EVAL_ATTEMPT

    return DspyRichTraceCandidateRecord(
        classification=classification,
        classification_reasons=[reason.value for reason in reasons],
        report_path=report_path,
        report_relative_path=report_relative_path(report_path, corpus_root),
        session_id=session_id,
        attempt=attempt,
        run=run,
        generation_calls=generation_calls,
        encoder_call=encoder_call,
        decoder_call=decoder_call,
    )


def write_jsonl_gz(path: Path, rows: Iterable[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(path, "wt", encoding="utf-8") as file:
        for row in rows:
            file.write(row.model_dump_json() + "\n")
            count += 1
    return count


def iter_dspy_rich_trace_records(
    path: Path,
) -> Iterator[DspyRichTraceCandidateRecord]:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield DspyRichTraceCandidateRecord.model_validate_json(line)


def build_dspy_rich_trace_attempt(
    record: DspyRichTraceCandidateRecord,
) -> DspyRichTraceAttemptRow:
    if record.classification not in {
        DspyRichTraceClassification.DIRECT_EVAL_ATTEMPT,
        DspyRichTraceClassification.ENCDEC_EVAL_ATTEMPT,
    }:
        raise ValueError("DSPy rich rows can only be built from clean records")
    if record.decoder_call is None:
        raise ValueError("clean DSPy record is missing decoder_call")
    raw_completed_code = record.attempt.raw_completed_code
    if raw_completed_code is None:
        raise ValueError("clean DSPy record is missing raw_completed_code")
    test_pass_rate = record.attempt.test_pass_rate
    if test_pass_rate is None:
        raise ValueError("clean DSPy record is missing test_pass_rate")

    decoder_output = generation_call_output(record.decoder_call)
    decoder_prompt_text = prompt_text(record.decoder_call.messages)
    if decoder_prompt_text is None:
        raise ValueError("clean DSPy record is missing decoder prompt text")
    decoder_input = (
        extract_field_from_messages(
            record.decoder_call.messages, FIELD_CODE_SPEC
        )
        or extract_field_from_messages(
            record.decoder_call.messages, FIELD_FUNCTION_STUB
        )
        or decoder_prompt_text
    )

    encoder_prompt_text: str | None = None
    encoder_output: str | None = None
    encoder_call_output: str | None = None
    encoder_messages: list[dict[str, Any]] | None = None
    encoder_input: str | None = None
    if record.encoder_call is not None:
        encoder_messages = record.encoder_call.messages
        encoder_prompt_text = prompt_text(encoder_messages)
        encoder_call_output = generation_call_output(record.encoder_call)
        encoder_output = (
            extract_dspy_field(encoder_call_output, FIELD_CODE_SPEC)
            or encoder_call_output
        )
        encoder_input = extract_field_from_messages(
            encoder_messages, "input_code"
        )

    generation_type = record.attempt.generation_type or (
        record.run.generation_type if record.run is not None else None
    )
    generation_type_text = generation_type or ""
    return DspyRichTraceAttemptRow(
        attempt_id=record.attempt.id,
        dataset=DATASET_NAME,
        task_id=record.attempt.task_id,
        data_sample_id=data_sample_id_for(record.attempt.task_id),
        decoder_pool_name=generation_type_text or None,
        decoder_sample_id=record.decoder_call.id,
        encoder_pool_name=(
            generation_type_text if record.encoder_call is not None else None
        ),
        encoder_sample_id=(
            record.encoder_call.id if record.encoder_call is not None else None
        ),
        sample_idx=record.attempt.dataset_index,
        run_id=record.attempt.run_id,
        created_at=record.attempt.timestamp,
        source_sample_id=record.attempt.id,
        source_pool_name=record.report_relative_path,
        enc_llm_config_id=(
            record.run.config.get("llm_config_id")
            if record.run is not None
            and isinstance(record.run.config.get("llm_config_id"), str)
            else None
        ),
        dec_llm_config_id=(
            record.run.config.get("llm_config_id")
            if record.run is not None
            and isinstance(record.run.config.get("llm_config_id"), str)
            else None
        ),
        enc_prompt_json=encoder_messages,
        enc_prompt_text=encoder_prompt_text,
        enc_input=encoder_input,
        enc_output=encoder_output,
        dec_prompt_json=record.decoder_call.messages,
        dec_prompt_text=decoder_prompt_text,
        dec_input=decoder_input,
        dec_output=raw_completed_code,
        enc_provider=provider_for_model(
            record.encoder_call.model
            if record.encoder_call is not None
            else None
        ),
        enc_model=record.encoder_call.model
        if record.encoder_call is not None
        else None,
        enc_finish_reason=finish_reason_for(record.encoder_call),
        enc_usage_json=record.encoder_call.usage
        if record.encoder_call
        else None,
        enc_cost_json=cost_json_for(record.encoder_call),
        dec_provider=provider_for_model(record.decoder_call.model),
        dec_model=record.decoder_call.model,
        dec_finish_reason=finish_reason_for(record.decoder_call),
        dec_usage_json=record.decoder_call.usage,
        dec_cost_json=cost_json_for(record.decoder_call),
        rich_extraction_level=record.classification.value,
        dspy_session_id=record.session_id,
        dspy_report_path=record.report_relative_path,
        dspy_attempt_id=record.attempt.id,
        dspy_generation_type=generation_type_text,
        dspy_dataset_index=record.attempt.dataset_index,
        dspy_repeat_index=record.attempt.repeat_index,
        dspy_test_pass_rate=float(test_pass_rate),
        dspy_raw_completed_code=raw_completed_code,
        dspy_extracted_code=record.attempt.extracted_code,
        dspy_run_format=record.attempt.run_format,
        dspy_generation_log_source_relative_path=(
            record.attempt.generation_log_source_relative_path
        ),
        dspy_generation_call_ids=record.attempt.generation_call_ids,
        dspy_encoder_call_id=(
            record.encoder_call.id if record.encoder_call is not None else None
        ),
        dspy_decoder_call_id=record.decoder_call.id,
        dspy_encoder_call_output=encoder_call_output,
        dspy_decoder_call_output=decoder_output,
    )


def safe_path_part(value: str) -> str:
    return value.replace("/", "__").replace(" ", "_")


def write_dataframe_parquet(path: Path, rows: Sequence[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [row.model_dump(mode="json") for row in rows]
    pd.DataFrame(payload).to_parquet(path, index=False)
    return len(payload)
