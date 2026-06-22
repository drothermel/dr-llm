from __future__ import annotations

from datetime import datetime

import pytest

from humaneval_pool_extract_common import (
    DumpRowHints,
    DumpedPoolRow,
    OutputKind,
)
from rich_trace_extract_common import (
    RichTraceClassification,
    build_rich_trace_attempt,
    classify_rich_trace_candidate,
    dataset_and_task_id,
    prompt_text,
    safe_path_part,
)


def _row(
    *,
    project_name: str = "code_comp_t1",
    pool_name: str,
    sample_id: str,
    key_values: dict[str, object] | None = None,
    request_prompt: object | None = None,
    response_text: str | None = None,
    metadata: dict[str, object] | None = None,
) -> DumpedPoolRow:
    request_json = {}
    if request_prompt is not None:
        request_json["prompt"] = request_prompt
    response_json = None
    if response_text is not None:
        response_json = {
            "text": response_text,
            "provider": "openrouter",
            "model": "test-model",
            "finish_reason": "stop",
            "usage": {"total_tokens": 12},
            "cost": {"total_cost_usd": 0.01},
            "latency_ms": 123,
        }
    return DumpedPoolRow(
        project_name=project_name,
        pool_name=pool_name,
        table_name=f"{pool_name}_samples",
        sample_id=sample_id,
        key_values=key_values or {},
        sample_idx=0,
        run_id="run-1",
        request_json=request_json,
        response_json=response_json,
        finish_reason="stop",
        attempt_count=0,
        metadata_json=metadata or {},
        created_at=datetime(2026, 1, 1),
        hints=DumpRowHints(
            output_kind=OutputKind.NOT_CODE,
            output_json_path=None,
            decoder_input_description_source="missing",
        ),
    )


def _message(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


def test_classifies_full_encoder_chain_and_builds_rich_row() -> None:
    encoder = _row(
        pool_name="budget_enc_v0_size6",
        sample_id="enc-1",
        key_values={
            "prompt_template_id": "enc-template",
            "llm_config_id": "enc-config",
        },
        request_prompt=_message("encode this code"),
        response_text="short description",
    )
    decoder = _row(
        pool_name="budget_dec_v0_size6",
        sample_id="dec-1",
        key_values={
            "dec_prompt_template_id": "dec-template",
            "dec_llm_config_id": "dec-config",
        },
        request_prompt=_message("decode short description"),
        response_text="def solution(): pass",
        metadata={
            "data_sample_id": "human_eval/HumanEval/0/gt_solution@abc",
            "source_kind": "encoder_sample",
            "source_sample_id": "encoder_pool/budget_enc_v0_size6/enc-1",
            "source_pool_name": "budget_enc_v0_size6",
            "source_text": "short description",
        },
    )
    record = classify_rich_trace_candidate(
        decoder,
        {("code_comp_t1", "budget_enc_v0_size6", "enc-1"): encoder},
    )

    assert record is not None
    assert record.classification == RichTraceClassification.FULL_ENCODER_CHAIN
    attempt = build_rich_trace_attempt(record)
    assert attempt.dataset == "human_eval"
    assert attempt.task_id == "HumanEval/0"
    assert attempt.enc_prompt_text == "encode this code"
    assert attempt.enc_output == "short description"
    assert attempt.dec_prompt_text == "decode short description"
    assert attempt.dec_input == "short description"
    assert attempt.dec_output == "def solution(): pass"
    assert attempt.enc_prompt_template_id == "enc-template"
    assert attempt.dec_prompt_template_id == "dec-template"


def test_missing_encoder_row_is_messy() -> None:
    decoder = _row(
        pool_name="budget_dec_v0_size6",
        sample_id="dec-1",
        request_prompt=_message("decode"),
        response_text="def solution(): pass",
        metadata={
            "source_kind": "encoder_sample",
            "source_sample_id": "encoder_pool/budget_enc_v0_size6/missing",
        },
    )

    record = classify_rich_trace_candidate(decoder, {})

    assert record is not None
    assert record.classification == RichTraceClassification.MISSING_ENCODER_ROW
    assert "missing_encoder_row" in record.classification_reasons


def test_missing_decoder_prompt_is_messy() -> None:
    encoder = _row(
        pool_name="budget_enc_v0_size6",
        sample_id="enc-1",
        request_prompt=_message("encode"),
        response_text="description",
    )
    decoder = _row(
        pool_name="budget_dec_v0_size6",
        sample_id="dec-1",
        response_text="def solution(): pass",
        metadata={
            "source_kind": "encoder_sample",
            "source_sample_id": "encoder_pool/budget_enc_v0_size6/enc-1",
        },
    )

    record = classify_rich_trace_candidate(
        decoder,
        {("code_comp_t1", "budget_enc_v0_size6", "enc-1"): encoder},
    )

    assert record is not None
    assert (
        record.classification == RichTraceClassification.MISSING_DECODER_PROMPT
    )


def test_non_decoder_row_is_ignored() -> None:
    encoder = _row(
        pool_name="budget_enc_v0_size6",
        sample_id="enc-1",
        request_prompt=_message("encode"),
        response_text="description",
    )

    assert classify_rich_trace_candidate(encoder, {}) is None


def test_build_rich_trace_attempt_rejects_messy_record() -> None:
    decoder = _row(
        pool_name="budget_dec_v0_size6",
        sample_id="dec-1",
        request_prompt=_message("decode"),
        response_text="def solution(): pass",
    )
    record = classify_rich_trace_candidate(decoder, {})

    assert record is not None
    with pytest.raises(ValueError, match="full encoder chain"):
        build_rich_trace_attempt(record)


def test_prompt_text_and_path_helpers() -> None:
    row = _row(
        pool_name="budget_dec_v0_size6",
        sample_id="dec-1",
        request_prompt=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        response_text="out",
    )

    assert prompt_text(row) == "system\n\nuser"
    assert dataset_and_task_id("human_eval/HumanEval/42/gt@abc") == (
        "human_eval",
        "HumanEval/42",
    )
    assert safe_path_part("HumanEval/42") == "HumanEval__42"
