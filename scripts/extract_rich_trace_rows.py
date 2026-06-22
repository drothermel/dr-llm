#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rich_trace_extract_common import (
    build_rich_trace_attempt,
    clean_records_path_for,
    iter_rich_trace_records,
    rich_attempts_path_for,
    rich_trace_split_dir_for,
    write_dataframe_parquet,
)

app = typer.Typer(
    help="Extract rich encoder-decoder trace rows from clean candidate records."
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
            help="Directory containing rich_trace_split/.",
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
            help="Clean rich-trace candidate JSONL gzip path.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Rich trace Parquet output path."),
    ] = None,
) -> None:
    clean_path = input_path or clean_records_path_for(
        rich_trace_split_dir_for(dump_dir)
    )
    output_path = output or rich_attempts_path_for(dump_dir)
    attempts = [
        build_rich_trace_attempt(record)
        for record in iter_rich_trace_records(clean_path)
    ]
    row_count = write_dataframe_parquet(output_path, attempts)
    typer.echo(f"Wrote {row_count} rich trace attempts to {output_path}")


if __name__ == "__main__":
    app()
