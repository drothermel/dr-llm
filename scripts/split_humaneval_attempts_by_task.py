#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from pydantic import BaseModel, ConfigDict

app = typer.Typer(
    help="Split a unified HumanEval attempts Parquet file by task id."
)

TASK_ID_COLUMN = "human_eval_task_id"
CODE_OUTPUT_COLUMN = "raw_code_output"
DEFAULT_OUTPUT_DIR_NAME = "per_elem"
TASK_FILE_SUFFIX = "decode"
TASK_FILE_EXTENSION = ".parquet"
DEDUP_FILE_SUFFIX = "decode-dedup"
DEDUP_FILE_EXTENSION = ".jsonl"
TASK_ID_RE = re.compile(r"^HumanEval/(?P<index>\d+)$")


class TaskSplitManifestEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    human_eval_task_id: str
    row_count: int
    unique_output_count: int
    file_name: str
    dedup_file_name: str


class DeduplicatedOutputRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    out: str
    count: int


class TaskSplitManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_parquet: Path
    output_dir: Path
    task_id_column: str
    total_rows: int
    tasks: list[TaskSplitManifestEntry]


@app.command()
def main(
    parquet_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Unified HumanEval attempts Parquet file.",
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            file_okay=False,
            dir_okay=True,
            help="Directory for one Parquet file per HumanEval task.",
        ),
    ] = None,
) -> None:
    destination = output_dir or parquet_path.parent / DEFAULT_OUTPUT_DIR_NAME
    destination.mkdir(parents=True, exist_ok=True)

    attempts = pd.read_parquet(parquet_path)
    validate_attempts(attempts, parquet_path)

    manifest_entries: list[TaskSplitManifestEntry] = []
    grouped = attempts.groupby(TASK_ID_COLUMN, sort=False, dropna=False)
    for task_id, task_attempts in grouped:
        task_id_text = validate_task_id(task_id)
        task_file = task_file_name(task_id_text)
        dedup_file = dedup_file_name(task_id_text)
        task_path = destination / task_file
        dedup_path = destination / dedup_file
        task_attempts.to_parquet(task_path, index=False)
        dedup_row_count = write_deduplicated_outputs(dedup_path, task_attempts)
        manifest_entries.append(
            TaskSplitManifestEntry(
                human_eval_task_id=task_id_text,
                row_count=len(task_attempts),
                unique_output_count=dedup_row_count,
                file_name=task_file,
                dedup_file_name=dedup_file,
            )
        )
        typer.echo(
            f"{task_id_text}: wrote {len(task_attempts)} rows to {task_path} "
            f"and {dedup_row_count} deduped outputs to {dedup_path}"
        )

    manifest = TaskSplitManifest(
        source_parquet=parquet_path,
        output_dir=destination,
        task_id_column=TASK_ID_COLUMN,
        total_rows=len(attempts),
        tasks=sorted(
            manifest_entries,
            key=lambda entry: task_index(entry.human_eval_task_id),
        ),
    )
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    typer.echo(
        f"Wrote {len(manifest.tasks)} task files and manifest to {destination}"
    )


def validate_attempts(attempts: pd.DataFrame, parquet_path: Path) -> None:
    if TASK_ID_COLUMN not in attempts.columns:
        raise typer.BadParameter(
            f"{parquet_path} does not contain required column "
            f"{TASK_ID_COLUMN!r}"
        )
    if CODE_OUTPUT_COLUMN not in attempts.columns:
        raise typer.BadParameter(
            f"{parquet_path} does not contain required column "
            f"{CODE_OUTPUT_COLUMN!r}"
        )
    if attempts.empty:
        raise typer.BadParameter(f"{parquet_path} contains no rows")
    if attempts[TASK_ID_COLUMN].isna().any():
        raise typer.BadParameter(
            f"{parquet_path} contains rows with missing {TASK_ID_COLUMN!r}"
        )
    if attempts[CODE_OUTPUT_COLUMN].isna().any():
        raise typer.BadParameter(
            f"{parquet_path} contains rows with missing {CODE_OUTPUT_COLUMN!r}"
        )


def validate_task_id(task_id: object) -> str:
    if not isinstance(task_id, str):
        raise typer.BadParameter(
            f"Expected string task id in {TASK_ID_COLUMN!r}; got {task_id!r}"
        )
    if TASK_ID_RE.fullmatch(task_id) is None:
        raise typer.BadParameter(
            f"Expected task id shaped like 'HumanEval/0'; got {task_id!r}"
        )
    return task_id


def task_file_name(task_id: str) -> str:
    return f"human_eval-{task_index(task_id)}-{TASK_FILE_SUFFIX}{TASK_FILE_EXTENSION}"


def dedup_file_name(task_id: str) -> str:
    return (
        f"human_eval-{task_index(task_id)}-"
        f"{DEDUP_FILE_SUFFIX}{DEDUP_FILE_EXTENSION}"
    )


def task_index(task_id: str) -> int:
    match = TASK_ID_RE.fullmatch(task_id)
    if match is None:
        raise ValueError(f"Invalid HumanEval task id: {task_id!r}")
    return int(match.group("index"))


def write_deduplicated_outputs(path: Path, task_attempts: pd.DataFrame) -> int:
    output_counts = (
        task_attempts[CODE_OUTPUT_COLUMN]
        .value_counts(dropna=False)
        .rename_axis(CODE_OUTPUT_COLUMN)
        .reset_index(name="count")
        .sort_values(
            ["count", CODE_OUTPUT_COLUMN],
            ascending=[False, True],
            kind="stable",
        )
    )
    with path.open("w", encoding="utf-8") as file:
        for output, count in output_counts.itertuples(index=False, name=None):
            if not isinstance(output, str):
                raise typer.BadParameter(
                    f"Expected string {CODE_OUTPUT_COLUMN!r}; got {output!r}"
                )
            row = DeduplicatedOutputRow(out=output, count=int(count))
            file.write(row.model_dump_json() + "\n")
    return len(output_counts)


if __name__ == "__main__":
    app()
