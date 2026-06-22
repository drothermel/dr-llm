#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from dspy_rich_trace_extract_common import (
    DATASET_NAME,
    DEFAULT_DSPY_CORPUS_ROOT,
    DspyRichTraceClassification,
    DspyRichTraceSplitManifest,
    DspyRichTraceSplitSummary,
    classify_dspy_eval_attempt,
    clean_eval_attempts_path_for,
    dspy_rich_trace_split_dir_for,
    iter_eval_reports,
    messy_eval_attempts_path_for,
    split_manifest_path_for,
    split_summary_path_for,
    write_jsonl_gz,
)

app = typer.Typer(
    help="Split parsed DSPy eval attempts into clean and messy rich-trace records."
)


@app.command()
def main(
    corpus_root: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="DSPy corpus root containing parsed_eval_reports/.",
        ),
    ] = DEFAULT_DSPY_CORPUS_ROOT,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            file_okay=False,
            dir_okay=True,
            help="Output directory for clean/messy DSPy rich-trace split.",
        ),
    ] = None,
) -> None:
    split_dir = output_dir or dspy_rich_trace_split_dir_for(corpus_root)
    clean_records = []
    messy_records = []
    classification_counts: Counter[str] = Counter()
    generation_type_counts: Counter[str] = Counter()
    clean_dataset_counts: Counter[str] = Counter()
    clean_tasks: set[str] = set()
    report_count = 0

    for report_path, report in iter_eval_reports(corpus_root):
        report_count += 1
        for attempt in report.attempts:
            record = classify_dspy_eval_attempt(
                corpus_root=corpus_root,
                report_path=report_path,
                report=report,
                attempt=attempt,
            )
            classification_counts[record.classification.value] += 1
            generation_type_counts[attempt.generation_type or "<missing>"] += 1
            if record.classification in {
                DspyRichTraceClassification.DIRECT_EVAL_ATTEMPT,
                DspyRichTraceClassification.ENCDEC_EVAL_ATTEMPT,
            }:
                clean_records.append(record)
                clean_dataset_counts[DATASET_NAME] += 1
                clean_tasks.add(attempt.task_id)
            else:
                messy_records.append(record)

    clean_count = write_jsonl_gz(
        clean_eval_attempts_path_for(split_dir), clean_records
    )
    messy_count = write_jsonl_gz(
        messy_eval_attempts_path_for(split_dir), messy_records
    )
    manifest = DspyRichTraceSplitManifest(
        source_corpus_root=corpus_root,
        output_dir=split_dir,
        clean_file_name=clean_eval_attempts_path_for(split_dir)
        .relative_to(split_dir)
        .as_posix(),
        messy_file_name=messy_eval_attempts_path_for(split_dir)
        .relative_to(split_dir)
        .as_posix(),
        clean_count=clean_count,
        messy_count=messy_count,
    )
    split_manifest_path_for(split_dir).write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    summary = DspyRichTraceSplitSummary(
        source_corpus_root=corpus_root,
        output_dir=split_dir,
        report_count=report_count,
        classification_counts=dict(sorted(classification_counts.items())),
        generation_type_counts=dict(sorted(generation_type_counts.items())),
        clean_dataset_counts=dict(sorted(clean_dataset_counts.items())),
        clean_task_count=len(clean_tasks),
    )
    split_summary_path_for(split_dir).write_text(
        summary.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    typer.echo(
        f"Wrote {clean_count} clean and {messy_count} messy DSPy rich trace "
        f"records to {split_dir}"
    )


if __name__ == "__main__":
    app()
