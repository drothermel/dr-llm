#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from rich_trace_extract_common import (
    DatasetSplitEntry,
    DatasetTaskSplitEntry,
    RichTraceDatasetSplitManifest,
    by_dataset_dir_for,
    rich_attempts_path_for,
    rich_traces_dir_for,
    safe_path_part,
)

app = typer.Typer(help="Split rich trace attempts by dataset and task id.")

DATASET_COLUMN = "dataset"
TASK_ID_COLUMN = "task_id"


@app.command()
def main(
    dump_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Directory containing rich_traces/rich_trace_attempts.parquet.",
        ),
    ],
    input_path: Annotated[
        Path | None,
        typer.Option(
            "--input-path",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Rich trace Parquet path.",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            file_okay=False,
            dir_okay=True,
            help="Directory for dataset/task split output.",
        ),
    ] = None,
) -> None:
    parquet_path = input_path or rich_attempts_path_for(dump_dir)
    destination = output_dir or by_dataset_dir_for(
        rich_traces_dir_for(dump_dir)
    )
    destination.mkdir(parents=True, exist_ok=True)
    attempts = pd.read_parquet(parquet_path)
    validate_attempts(attempts, parquet_path)

    dataset_entries: list[DatasetSplitEntry] = []
    for dataset, dataset_rows in attempts.groupby(
        DATASET_COLUMN, sort=True, dropna=False
    ):
        dataset_text = validate_group_key(dataset, DATASET_COLUMN)
        dataset_dir = destination / safe_path_part(dataset_text)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        all_file_name = "all.parquet"
        dataset_rows.to_parquet(dataset_dir / all_file_name, index=False)
        task_entries: list[DatasetTaskSplitEntry] = []
        for task_id, task_rows in dataset_rows.groupby(
            TASK_ID_COLUMN, sort=True, dropna=False
        ):
            task_text = validate_group_key(task_id, TASK_ID_COLUMN)
            task_file_name = f"{safe_path_part(task_text)}.parquet"
            task_rows.to_parquet(dataset_dir / task_file_name, index=False)
            task_entries.append(
                DatasetTaskSplitEntry(
                    dataset=dataset_text,
                    task_id=task_text,
                    row_count=len(task_rows),
                    file_name=(
                        f"{safe_path_part(dataset_text)}/{task_file_name}"
                    ),
                )
            )
        dataset_entries.append(
            DatasetSplitEntry(
                dataset=dataset_text,
                row_count=len(dataset_rows),
                all_file_name=f"{safe_path_part(dataset_text)}/{all_file_name}",
                task_count=len(task_entries),
                tasks=task_entries,
            )
        )

    manifest = RichTraceDatasetSplitManifest(
        source_parquet=parquet_path,
        output_dir=destination,
        total_rows=len(attempts),
        datasets=dataset_entries,
    )
    (destination / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    typer.echo(
        f"Wrote {len(attempts)} rows across {len(dataset_entries)} datasets "
        f"to {destination}"
    )


def validate_attempts(attempts: pd.DataFrame, parquet_path: Path) -> None:
    for column in [DATASET_COLUMN, TASK_ID_COLUMN]:
        if column not in attempts.columns:
            raise typer.BadParameter(
                f"{parquet_path} does not contain required column {column!r}"
            )
        if attempts[column].isna().any():
            raise typer.BadParameter(
                f"{parquet_path} contains rows with missing {column!r}"
            )


def validate_group_key(value: object, column_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise typer.BadParameter(
            f"Expected non-empty string {column_name!r}; got {value!r}"
        )
    return value


if __name__ == "__main__":
    app()
