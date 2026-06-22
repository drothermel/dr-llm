#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from rich_trace_extract_common import (
    RichTraceClassification,
    RichTraceSplitManifest,
    RichTraceSplitSummary,
    classify_rich_trace_candidate,
    clean_records_path_for,
    data_sample_id_for,
    dataset_and_task_id,
    index_dumped_rows,
    iter_dumped_pool_rows_from_manifest,
    messy_records_path_for,
    rich_trace_split_dir_for,
    split_manifest_path_for,
    split_summary_path_for,
    write_jsonl_gz,
)

app = typer.Typer(
    help="Split dumped pool rows into clean and messy rich-trace candidates."
)


@app.command()
def main(
    dump_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Directory containing manifest.json and per-pool JSONL dumps.",
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            file_okay=False,
            dir_okay=True,
            help="Output directory for clean/messy rich trace split.",
        ),
    ] = None,
) -> None:
    split_dir = output_dir or rich_trace_split_dir_for(dump_dir)
    rows = list(iter_dumped_pool_rows_from_manifest(dump_dir))
    row_index = index_dumped_rows(rows)
    clean_records = []
    messy_records = []
    classification_counts: Counter[str] = Counter()
    decoder_pool_counts: Counter[str] = Counter()
    clean_dataset_counts: Counter[str] = Counter()
    clean_tasks: set[tuple[str, str]] = set()

    for row in rows:
        record = classify_rich_trace_candidate(row, row_index)
        if record is None:
            continue
        classification_counts[record.classification.value] += 1
        decoder_pool_counts[f"{row.project_name}/{row.pool_name}"] += 1
        if record.classification == RichTraceClassification.FULL_ENCODER_CHAIN:
            clean_records.append(record)
            data_sample_id = data_sample_id_for(row)
            if data_sample_id is not None:
                dataset, task_id = dataset_and_task_id(data_sample_id)
                clean_dataset_counts[dataset] += 1
                clean_tasks.add((dataset, task_id))
        else:
            messy_records.append(record)

    clean_count = write_jsonl_gz(
        clean_records_path_for(split_dir), clean_records
    )
    messy_count = write_jsonl_gz(
        messy_records_path_for(split_dir), messy_records
    )
    manifest = RichTraceSplitManifest(
        source_dump_dir=dump_dir,
        output_dir=split_dir,
        clean_file_name=clean_records_path_for(split_dir)
        .relative_to(split_dir)
        .as_posix(),
        messy_file_name=messy_records_path_for(split_dir)
        .relative_to(split_dir)
        .as_posix(),
        clean_count=clean_count,
        messy_count=messy_count,
    )
    split_manifest_path_for(split_dir).write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    summary = RichTraceSplitSummary(
        source_dump_dir=dump_dir,
        output_dir=split_dir,
        classification_counts=dict(sorted(classification_counts.items())),
        decoder_pool_counts=dict(sorted(decoder_pool_counts.items())),
        clean_dataset_counts=dict(sorted(clean_dataset_counts.items())),
        clean_task_count=len(clean_tasks),
    )
    split_summary_path_for(split_dir).write_text(
        summary.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    typer.echo(
        f"Wrote {clean_count} clean and {messy_count} messy rich trace "
        f"records to {split_dir}"
    )


if __name__ == "__main__":
    app()
