#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer

from humaneval_pool_extract_common import (
    CodeAttemptRow,
    build_code_attempt,
    iter_dump_rows,
    manifest_path_for,
    parquet_path_for,
    preview_path_for,
    read_manifest,
)

app = typer.Typer(
    help="Extract HumanEval code attempts from pool JSONL dumps."
)

DEFAULT_HUMANEVAL_CACHE_PAYLOAD = (
    Path.home()
    / ".cache/nl-code/datasets/evalplus__humanevalplus/test/v3/payload.json.gz"
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
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Parquet output path."),
    ] = None,
    preview_rows: Annotated[
        int,
        typer.Option("--preview-rows", min=0, help="CSV preview row count."),
    ] = 200,
    humaneval_cache_payload: Annotated[
        Path | None,
        typer.Option(
            "--humaneval-cache-payload",
            help=(
                "nl-code parsed HumanEval payload.json.gz used to backfill "
                "official prompt descriptions."
            ),
        ),
    ] = DEFAULT_HUMANEVAL_CACHE_PAYLOAD,
) -> None:
    manifest = read_manifest(manifest_path_for(dump_dir))
    humaneval_prompts = load_humaneval_prompts(humaneval_cache_payload)
    attempts: list[CodeAttemptRow] = []
    for pool in manifest.pools:
        path = dump_dir / pool.file_name
        pool_attempts = 0
        for dumped_row in iter_dump_rows(path):
            attempt = build_code_attempt(
                dumped_row,
                humaneval_prompts_by_task_id=humaneval_prompts,
            )
            if attempt is None:
                continue
            attempts.append(attempt)
            pool_attempts += 1
        typer.echo(
            f"{pool.project_name}/{pool.pool_name}: "
            f"extracted {pool_attempts} code attempts"
        )

    rows = [attempt.model_dump(mode="json") for attempt in attempts]
    output_path = output or parquet_path_for(dump_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    typer.echo(f"Wrote {len(rows)} attempts to {output_path}")

    if preview_rows > 0:
        write_preview(
            preview_path_for(output_path.parent), rows[:preview_rows]
        )
        typer.echo(f"Wrote preview to {preview_path_for(output_path.parent)}")


def write_preview(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    field_names = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def load_humaneval_prompts(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as file:
        payload = json.load(file)
    prompts: dict[str, str] = {}
    raw_samples = payload.get("raw_samples") or {}
    if not isinstance(raw_samples, dict):
        return prompts
    for task_id, raw_sample in raw_samples.items():
        if not isinstance(task_id, str) or not isinstance(raw_sample, dict):
            continue
        source = raw_sample.get("source")
        if not isinstance(source, dict):
            continue
        prompt = source.get("prompt")
        if isinstance(prompt, str) and prompt:
            prompts[task_id] = prompt
    return prompts


if __name__ == "__main__":
    app()
