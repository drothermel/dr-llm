#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from dspy_rich_trace_extract_common import (
    DEFAULT_DSPY_CORPUS_ROOT,
    build_dspy_rich_trace_attempt,
    clean_eval_attempts_path_for,
    dspy_rich_attempts_path_for,
    dspy_rich_trace_split_dir_for,
    iter_dspy_rich_trace_records,
    write_dataframe_parquet,
)

app = typer.Typer(
    help="Extract rich trace rows from clean parsed DSPy eval attempt records."
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
            help="DSPy corpus root containing dspy_rich_trace_split/.",
        ),
    ] = DEFAULT_DSPY_CORPUS_ROOT,
    input_path: Annotated[
        Path | None,
        typer.Option(
            "--input-path",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Clean DSPy rich-trace candidate JSONL gzip path.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="DSPy rich trace Parquet output path."),
    ] = None,
) -> None:
    clean_path = input_path or clean_eval_attempts_path_for(
        dspy_rich_trace_split_dir_for(corpus_root)
    )
    output_path = output or dspy_rich_attempts_path_for(corpus_root)
    attempts = [
        build_dspy_rich_trace_attempt(record)
        for record in iter_dspy_rich_trace_records(clean_path)
    ]
    row_count = write_dataframe_parquet(output_path, attempts)
    typer.echo(f"Wrote {row_count} DSPy rich trace attempts to {output_path}")


if __name__ == "__main__":
    app()
