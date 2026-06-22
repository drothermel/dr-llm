from __future__ import annotations

from pathlib import Path

import pytest

from dspy_rich_trace_extract_common import (
    DspyEvalAttempt,
    DspyEvalReport,
    DspyEvalRun,
    DspyGenerationCall,
    DspyReportSession,
    DspyRichTraceClassification,
    build_dspy_rich_trace_attempt,
    classify_dspy_eval_attempt,
    safe_path_part,
)


def _message(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


def _call(
    *,
    call_id: str,
    prompt_kind: str,
    messages: list[dict[str, str]],
    output: str,
) -> DspyGenerationCall:
    return DspyGenerationCall(
        id=call_id,
        model="openrouter/openai/gpt-5-nano",
        prompt_kind=prompt_kind,
        messages=messages,
        outputs=[output],
        response={"choices": [{"finish_reason": "stop"}]},
        usage={"total_tokens": 12},
        cost=0.01,
    )


def _attempt(
    *,
    generation_type: str = "encdec",
    generation_call_ids: list[str] | None = None,
    skipped: bool = False,
    raw_completed_code: str | None = "def solution(): pass",
) -> DspyEvalAttempt:
    return DspyEvalAttempt(
        id="run-1:attempt:000000",
        run_id="run-1",
        run_format="eval_report",
        timestamp="2026-05-15T00:00:00Z",
        generation_type=generation_type,
        dataset_index=0,
        task_id="HumanEval/0",
        repeat_index=0,
        skipped=skipped,
        raw_completed_code=raw_completed_code,
        extracted_code="def solution(): pass\n",
        test_pass_rate=1.0,
        generation_log_source_relative_path="raw/logs/generations.jsonl",
        generation_call_ids=(
            ["enc-1", "dec-1"]
            if generation_call_ids is None
            else generation_call_ids
        ),
    )


def _report(calls: list[DspyGenerationCall]) -> DspyEvalReport:
    return DspyEvalReport(
        session=DspyReportSession(session_id="session_000001"),
        runs=[DspyEvalRun(id="run-1", generation_type="encdec")],
        generation_calls=calls,
    )


def _report_path(tmp_path: Path) -> Path:
    return tmp_path / "parsed_eval_reports" / "session_000001.eval_report.json"


def test_classifies_encdec_attempt_and_builds_rich_row(tmp_path: Path) -> None:
    encoder = _call(
        call_id="enc-1",
        prompt_kind="encode_code_spec",
        messages=_message("[[ ## input_code ## ]]\ndef solution(): pass"),
        output="[[ ## code_spec ## ]]\nReturn nothing.",
    )
    decoder = _call(
        call_id="dec-1",
        prompt_kind="decode_code_spec",
        messages=_message(
            "[[ ## code_spec ## ]]\nReturn nothing.\n\n"
            "[[ ## function_stub ## ]]\ndef solution():"
        ),
        output="[[ ## completed_code ## ]]\ndef solution(): pass",
    )
    attempt = _attempt()
    report = _report([encoder, decoder])

    record = classify_dspy_eval_attempt(
        corpus_root=tmp_path,
        report_path=_report_path(tmp_path),
        report=report,
        attempt=attempt,
    )

    assert (
        record.classification
        == DspyRichTraceClassification.ENCDEC_EVAL_ATTEMPT
    )
    rich_row = build_dspy_rich_trace_attempt(record)
    assert rich_row.dataset == "human_eval"
    assert rich_row.task_id == "HumanEval/0"
    assert rich_row.enc_input == "def solution(): pass"
    assert rich_row.enc_output == "Return nothing."
    assert rich_row.dec_input == "Return nothing."
    assert rich_row.dec_output == "def solution(): pass"
    assert rich_row.dspy_decoder_call_output == (
        "[[ ## completed_code ## ]]\ndef solution(): pass"
    )
    assert rich_row.dspy_test_pass_rate == 1.0


def test_classifies_direct_attempt_with_null_encoder_fields(
    tmp_path: Path,
) -> None:
    direct = _call(
        call_id="direct-1",
        prompt_kind="direct_code_from_stub",
        messages=_message("[[ ## function_stub ## ]]\ndef solution():"),
        output="[[ ## completed_code ## ]]\ndef solution(): pass",
    )
    attempt = _attempt(
        generation_type="direct",
        generation_call_ids=["direct-1"],
    )
    report = DspyEvalReport(
        session=DspyReportSession(session_id="session_000001"),
        runs=[DspyEvalRun(id="run-1", generation_type="direct")],
        generation_calls=[direct],
    )

    record = classify_dspy_eval_attempt(
        corpus_root=tmp_path,
        report_path=_report_path(tmp_path),
        report=report,
        attempt=attempt,
    )

    assert (
        record.classification
        == DspyRichTraceClassification.DIRECT_EVAL_ATTEMPT
    )
    rich_row = build_dspy_rich_trace_attempt(record)
    assert rich_row.enc_prompt_text is None
    assert rich_row.enc_output is None
    assert rich_row.dec_input == "def solution():"
    assert rich_row.dec_output == "def solution(): pass"
    assert rich_row.rich_extraction_level == "direct_eval_attempt"


def test_skipped_attempt_is_messy(tmp_path: Path) -> None:
    attempt = _attempt(skipped=True)
    report = _report([])

    record = classify_dspy_eval_attempt(
        corpus_root=tmp_path,
        report_path=_report_path(tmp_path),
        report=report,
        attempt=attempt,
    )

    assert record.classification == DspyRichTraceClassification.SKIPPED_ATTEMPT
    assert "skipped_attempt" in record.classification_reasons


def test_missing_generation_call_ids_is_messy(tmp_path: Path) -> None:
    attempt = _attempt(generation_call_ids=[])
    report = _report([])

    record = classify_dspy_eval_attempt(
        corpus_root=tmp_path,
        report_path=_report_path(tmp_path),
        report=report,
        attempt=attempt,
    )

    assert (
        record.classification
        == DspyRichTraceClassification.MISSING_GENERATION_CALL_IDS
    )


def test_unexpected_prompt_kinds_is_messy(tmp_path: Path) -> None:
    bad_call = _call(
        call_id="bad-1",
        prompt_kind="other_prompt",
        messages=_message("prompt"),
        output="out",
    )
    attempt = _attempt(generation_call_ids=["bad-1"])
    report = _report([bad_call])

    record = classify_dspy_eval_attempt(
        corpus_root=tmp_path,
        report_path=_report_path(tmp_path),
        report=report,
        attempt=attempt,
    )

    assert (
        record.classification
        == DspyRichTraceClassification.UNEXPECTED_PROMPT_KINDS
    )


def test_build_rich_row_rejects_messy_record(tmp_path: Path) -> None:
    attempt = _attempt(generation_call_ids=[])
    report = _report([])
    record = classify_dspy_eval_attempt(
        corpus_root=tmp_path,
        report_path=_report_path(tmp_path),
        report=report,
        attempt=attempt,
    )

    with pytest.raises(ValueError, match="clean records"):
        build_dspy_rich_trace_attempt(record)


def test_safe_path_part() -> None:
    assert safe_path_part("HumanEval/42") == "HumanEval__42"
