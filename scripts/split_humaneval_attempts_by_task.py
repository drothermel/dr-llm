#!/usr/bin/env python3
from __future__ import annotations

import json
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
DEFAULT_OUTPUT_DIR_NAME = "per_elem"
TASK_FILE_SUFFIX = "decode"
TASK_FILE_EXTENSION = ".parquet"
TASK_ID_RE = re.compile(r"^HumanEval/(?P<index>\d+)$")


class TaskSplitManifestEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    human_eval_task_id: str
    row_count: int
    file_name: str


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
        task_path = destination / task_file
        task_attempts.to_parquet(task_path, index=False)
        manifest_entries.append(
            TaskSplitManifestEntry(
                human_eval_task_id=task_id_text,
                row_count=len(task_attempts),
                file_name=task_file,
            )
        )
        typer.echo(
            f"{task_id_text}: wrote {len(task_attempts)} rows to {task_path}"
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
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
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
    if attempts.empty:
        raise typer.BadParameter(f"{parquet_path} contains no rows")
    if attempts[TASK_ID_COLUMN].isna().any():
        raise typer.BadParameter(
            f"{parquet_path} contains rows with missing {TASK_ID_COLUMN!r}"
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


def task_index(task_id: str) -> int:
    match = TASK_ID_RE.fullmatch(task_id)
    if match is None:
        raise ValueError(f"Invalid HumanEval task id: {task_id!r}")
    return int(match.group("index"))


if __name__ == "__main__":
    app()
